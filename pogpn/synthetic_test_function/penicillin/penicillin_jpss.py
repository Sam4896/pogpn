from typing import Dict, List, Optional

import torch

from ..base.dag_experiment_base import (
    DAGSyntheticTestFunction,
)
from ..penicillin.feature_extractor import (
    extract_simple_features_scipy,
)


class _PenicillinSDE:
    """Defines the drift for the Penicillin SDE."""

    def __init__(self):
        self.C_L_star = 8.26
        self.Y_xs = 0.45
        self.Y_xo = 0.04
        self.Y_ps = 0.90
        self.Y_po = 0.20
        self.K_1 = 1e-10
        self.K_2 = 7e-5
        self.m_X = 0.014
        self.m_o = 0.467
        self.alpha_1 = 0.143
        self.alpha_2 = 4e-7
        self.alpha_3 = 1e-4
        self.mu_X = 0.092
        self.K_X = 0.15
        self.mu_p = 0.005
        self.K_p = 2e-4
        self.K_I = 0.10
        self.p = 3
        self.K = 0.04
        self.k_g = 7e3
        self.E_g = 5100
        self.k_d = 1e33
        self.E_d = 50000
        self.rou_dot_C_p = 1000
        self.rou_c_dot_C_pc = 1000
        self.r_q1 = 60
        self.r_q2 = 1.6783e-4
        self.a = 1000
        self.b = 0.60
        self.alpha = 70
        self.beta = 0.4
        self.lambd = 2.5e-4
        self.gamma = 1e-5
        self.T_v = 273.0
        self.T_o = 373.0
        self.R = 1.9872

        # Robustness knobs
        self.V_MIN_EPS = 1e-6
        self.EXP_ARG_MIN = -300.0

    def f(
        self,
        y: torch.Tensor,
        T: torch.Tensor,
        F: torch.Tensor,
        s_f: torch.Tensor,
        H_: torch.Tensor,
    ) -> torch.Tensor:
        """Drift function for the Penicillin SDE."""
        P, V, X, S, CO2 = y.unbind(dim=-1)

        def pow10(x: torch.Tensor) -> torch.Tensor:
            return torch.pow(torch.tensor(10.0, device=x.device, dtype=x.dtype), x)

        H = pow10(-H_)
        F_loss = (
            self.lambd
            * V
            * (torch.exp(5.0 * ((T - self.T_o) / (self.T_v - self.T_o))) - 1.0)
        )
        dV_dt = F - F_loss
        mu_val = self.mu_X / (1.0 + self.K_1 / H + H / self.K_2)
        arg_g = torch.clamp(-self.E_g / (self.R * T), min=self.EXP_ARG_MIN)
        arg_d = torch.clamp(-self.E_d / (self.R * T), min=self.EXP_ARG_MIN)
        temp_term = self.k_g * torch.exp(arg_g) - self.k_d * torch.exp(arg_d)
        mu = mu_val * (S / (self.K_X * X + S)) * temp_term
        V_safe = torch.clamp(V, min=self.V_MIN_EPS)
        dX_dt = mu * X - (X / V_safe) * dV_dt
        mu_pp = self.mu_p * (S / (self.K_p + S + S.pow(2) / self.K_I))
        dS_dt = (
            -(mu / self.Y_xs) * X
            - (mu_pp / self.Y_ps) * X
            - self.m_X * X
            + (F * s_f / V_safe)
            - (S / V_safe) * dV_dt
        )
        dP_dt = (mu_pp * X) - self.K * P - (P / V_safe) * dV_dt
        dCO2_dt = self.alpha_1 * dX_dt + self.alpha_2 * X + self.alpha_3

        return torch.stack([dP_dt, dV_dt, dX_dt, dS_dt, dCO2_dt], dim=-1)


class _PenicillinSimulator:
    r"""Internal ODE simulator for the Penicillin process."""

    def __init__(
        self,
        observation_noise_std: Optional[float] = None,
        process_stochasticity_std: Optional[float] = None,
        n_steps: int = 500,
        step_size: float = 1.0,
        initial_p: float = 0.0,
    ) -> None:
        self.n_steps = int(n_steps)
        self.step_size = float(step_size)
        self.initial_p = float(initial_p)
        self.observation_noise_std = observation_noise_std
        self.CLAMP_NONNEG = True
        self.V_max = 180.0
        self.V_MIN_EPS = 1e-6

        self.sde = _PenicillinSDE()

        if process_stochasticity_std is None:
            self.process_stochasticity_std = 0.0
        else:
            self.process_stochasticity_std = float(process_stochasticity_std)
        self._process_noise_enabled = self.process_stochasticity_std > 0.0

    def _run_simulation(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        def _s1(t: torch.Tensor) -> torch.Tensor:
            return t.squeeze(-1) if t.ndim > 1 and t.size(-1) == 1 else t

        inputs = input_dict["inputs"]
        V, X, T, S, F, s_f, H_ = tuple(_s1(inputs[..., i]) for i in range(7))

        batch_shape = V.shape
        device, dtype = V.device, V.dtype

        P = torch.full_like(V, self.initial_p)
        CO2 = torch.zeros_like(V)
        t = torch.zeros_like(V)

        state_space_latent = torch.zeros(
            *batch_shape, self.n_steps, 5, device=device, dtype=dtype
        )
        alive = torch.ones_like(V, dtype=torch.bool)
        dt = torch.as_tensor(self.step_size, device=device, dtype=dtype)
        sqrt_dt = torch.sqrt(dt)

        if self._process_noise_enabled:
            d = 5  # Number of states
            Q = (self.process_stochasticity_std**2) * torch.eye(
                d, device=device, dtype=dtype
            )
            L = torch.linalg.cholesky(Q)

        for i in range(self.n_steps):
            y = torch.stack([P, V, X, S, CO2], dim=-1)  # (batch, 5)
            drift = self.sde.f(y, T, F, s_f, H_)  # (batch, 5)

            if self._process_noise_enabled:
                xi = torch.randn_like(y)  # (batch, 5)
                noise = (L @ xi.T).T
                dY = drift * dt + noise * sqrt_dt
            else:
                dY = drift * dt

            y_new = torch.where(alive.unsqueeze(-1), y + dY, y)

            if self.CLAMP_NONNEG:
                y_new[:, 0] = y_new[:, 0].clamp_min(0.0)  # P
                y_new[:, 2] = y_new[:, 2].clamp_min(0.0)  # X
                y_new[:, 3] = y_new[:, 3].clamp_min(0.0)  # S

            P, V, X, S, CO2 = y_new.unbind(dim=-1)
            t = torch.where(alive, t + dt, t)
            state_space_latent[..., i, :] = y_new

            finite_mask = y_new.isfinite().all(dim=-1)
            dP_dt = drift[:, 0]
            just_finished = (
                (V > self.V_max)
                | (S < 0.0)
                | (dP_dt < 1e-11)
                | (V <= self.V_MIN_EPS)
                | (~finite_mask)
            )
            alive &= ~just_finished
            if not alive.any():
                break

        # Calculate the actual number of steps that were simulated
        # Use the maximum time across all simulations to determine the relevant portion
        t_final_max = t.max().item()
        steps = (
            (t_final_max / dt)
            .round()
            .to(torch.long)
            .clamp(min=1, max=state_space_latent.shape[1])
        )

        return {
            "P_final": P,
            "V_final": V,
            "X_final": X,
            "S_final": S,
            "CO2_final": CO2,
            "t_final": t,
            "state_space": state_space_latent[:, :steps, :],
        }

    def _evaluate_true(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return self._run_simulation(input_dict)

    def _evaluate_noisy(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        latent = self._run_simulation(input_dict)
        sigma = float(self.observation_noise_std or 0.0)
        if sigma <= 0:
            return latent

        Y_latent = latent["state_space"]
        state_space_noisy = Y_latent + torch.randn_like(Y_latent) * sigma
        t_final = latent["t_final"]

        # Get the final values from the last time step of each simulation
        dt = torch.as_tensor(
            self.step_size, device=Y_latent.device, dtype=Y_latent.dtype
        )
        steps = (
            (t_final / dt).round().to(torch.long).clamp(min=1, max=Y_latent.shape[1])
        )
        idx = steps - 1
        b = torch.arange(Y_latent.shape[0], device=Y_latent.device)
        last = state_space_noisy[b, idx, :]
        noisy_P, noisy_V, noisy_X, noisy_S, noisy_CO2 = last.unbind(dim=-1)

        # Slice the state space to match the actual simulation length
        t_final_max = t_final.max().item()
        steps_max = (
            (t_final_max / dt)
            .round()
            .to(torch.long)
            .clamp(min=1, max=Y_latent.shape[1])
        )
        state_space_noisy = state_space_noisy[:, :steps_max, :]

        return {
            "P_final": noisy_P,
            "V_final": noisy_V,
            "X_final": noisy_X,
            "S_final": noisy_S,
            "CO2_final": noisy_CO2,
            "t_final": t_final,
            "state_space": state_space_noisy,
        }


class PenicillinJPSS(DAGSyntheticTestFunction):
    """Penicillin function."""

    def __init__(
        self,
        dim: int = 7,
        observation_noise_std: Optional[float] = None,
        process_stochasticity_std: Optional[float] = None,
        step_size: float = 1.0,
    ):
        if dim != 7:
            raise ValueError("dim must be 7")

        bounds = [
            (60.0, 120.0),
            (0.05, 18.0),
            (293.0, 303.0),
            (0.05, 18.0),
            (0.01, 0.50),
            (500.0, 700.0),
            (5.0, 6.5),
        ]
        root_node_indices_dict: Dict[str, List[int]] = {"inputs": list(range(dim))}
        observed_output_node_names: List[str] = [
            "final_observed_values",
            "combined_features",
        ]
        objective_node_name = "P_final"
        self._optimal_value = 14.5

        super().__init__(
            dim=dim,
            bounds=bounds,
            observed_output_node_names=observed_output_node_names,
            root_node_indices_dict=root_node_indices_dict,
            objective_node_name=objective_node_name,
            negate=False,
            process_stochasticity_std=process_stochasticity_std,
            observation_noise_std=observation_noise_std,
        )
        self.penicillin_simulator = _PenicillinSimulator(
            observation_noise_std=observation_noise_std,
            process_stochasticity_std=process_stochasticity_std,
            step_size=step_size,
        )
        self.step_size = step_size

    def _evaluate_true(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # This is called when `is_observation_noisy=True` because of inverted
        # logic in the base class. It should return observations with noise.
        output_dict = self.penicillin_simulator._evaluate_true(input_dict)
        return self._process_outputs(output_dict)

    def _evaluate_noisy(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # This is called when `is_observation_noisy=False` because of inverted
        # logic in the base class. It should return observations without noise.
        output_dict = self.penicillin_simulator._evaluate_noisy(input_dict)
        return self._process_outputs(output_dict)

    def _process_outputs(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        final_observed_values = self.get_final_observed_values(output_dict)
        combined_features = self.get_features(output_dict)

        processed_outputs = {
            "final_observed_values": final_observed_values,
            "combined_features": combined_features,
        }
        # Add final values for objective access (Uncomment if single task nodes need to be made)
        for k in ["P_final", "V_final", "X_final", "S_final", "CO2_final", "t_final"]:
            processed_outputs[k] = output_dict[k].unsqueeze(-1)

            # if "state_space" in output_dict:
            #     processed_outputs["state_space"] = output_dict["state_space"]

        return processed_outputs

    def get_final_observed_values(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return torch.stack(
            [
                output_dict["X_final"].view(-1),
                output_dict["V_final"].view(-1),
                output_dict["CO2_final"].view(-1),
                output_dict["t_final"].view(-1),
                # output_dict["P_final"].view(-1),
            ],
            dim=-1,
        )

    def get_features(self, output_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        step_size = self.penicillin_simulator.step_size

        exp_model_types = {
            "P": "convex",
            "V": "convex",
            "X": "concave",
            "CO2": "concave",
        }
        # No longer need to specify bounds, relying on new defaults in extractor
        exp_p0 = {"X": [300, 300, 0.01]}

        features_exp = extract_simple_features_scipy(
            output_dict,
            step_size,
            "exp_all",
            exp_p0=exp_p0,
            exp_model_types=exp_model_types,  # type: ignore
        )
        # features_lin = extract_simple_features_scipy(
        #     output_dict, step_size, "linear_all"
        # )

        return torch.stack(
            [
                # features_exp["P"]["b"].view(-1),
                features_exp["P"]["c"].view(-1),
                # features_exp["V"]["b"].view(-1),
                features_exp["V"]["c"].view(-1),
                # features_exp["X"]["b"].view(-1),
                # features_exp["X"]["c"].view(-1),
                # features_exp["CO2"]["b"].view(-1),
                # features_exp["S"]["peak"].view(-1),
                # features_exp["S"]["t_peak"].view(-1),
            ],
            dim=-1,
        )
