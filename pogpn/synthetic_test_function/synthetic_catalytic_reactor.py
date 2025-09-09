from typing import Dict, Optional, Tuple, List
import math
import torch
from .base.dag_experiment_base import DAGSyntheticTestFunction


class CatalyticBatchReactor(DAGSyntheticTestFunction):
    r"""Catalytic Batch Reactor.

    | idx | symbol  | meaning (units)                         | used by |
    |----:|---------|-----------------------------------------|---------|
    | 0   | `T`     | temperature [K]                         | `reaction_rate`, `energy_cost` |
    | 1   | `C_A`   | reactant-A concentration [mol L⁻¹]      | `reaction_rate` |
    | 2   | `m`     | mixing intensity [0-1]                  | `reaction_rate` |
    | 3   | `t`     | reaction time [s]                       | `yield_frac` |
    | 4   | `η_cat` | catalyst efficiency [1-2]               | `yield_frac` |
    | 5-8 | `P₁…P₄` | power inputs (voltage/pressure/…)        | `energy_cost` |
    | 9   | `φ_m`   | market value factor [-]                 | `net_output` |
    | 10  | `m_in`  | feed mass [kg]                          | `net_output` |
    | 11  | `φ_env` | environmental cost factor [-]           | `net_output` |

    ### Updated process equations

    \[
    \begin{aligned}
    r_\text{rate} &=
        A_0 \exp\!\Bigl[-\tfrac{E_a}{R\,T}\Bigr]\;
        C_A^{\,1+\;0.2\,m}\;
        m\;
        \bigl[1+0.1\sin(T/50)\bigr] \\[4pt]
    y_\text{frac} &=
        \operatorname{sat}_{[0,1]}\!
        \Bigl\{\,1-\exp[-\,r_\text{rate}\,t\,\eta_{\!cat}] +
        0.1\sin(2\pi t/50)\Bigr\} \\[4pt]
    E_\text{cost} &=
        \eta\!\sum_{i=1}^4 P_i^{2} +
        0.005\,(T-300)^2 +
        5\!\times\!10^{-4}\Bigl(\!\sum_{i=1}^4 P_i\Bigr)^{2} \\[4pt]
    \text{Net} &=
        (V_0 + 0.5\,\phi_m)\;y_\text{frac}\,m_\text{in}
        - e^{0.1\,\phi_\text{env}}\,E_\text{cost}.
    \end{aligned}
    \]

    *sat₍₀,₁₎ clamps to [0, 1].  Constants unchanged:*
    \(A_0=10^{3},\,E_a=50,\,R=8.314,\,\eta=10^{-2},\,V_0=5.\)
    """

    # ------------------------------------------------------------------ #
    # ctor identical except for docstring – omitted for brevity
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        dim: int = 12,
        bounds: Optional[List[Tuple[float, float]]] = None,
        observation_noise_std: Optional[float] = None,
        process_stochasticity_std: Optional[float] = None,
    ):
        if dim != 12:
            raise ValueError("CatalyticBatchReactor supports exactly 12 inputs.")
        if bounds is None:
            bounds = [(-10.0, 10.0) for _ in range(dim)]

        self.optimizers = None

        self.Ea, self.R, self.A0 = 50.0, 8.314, 1e3
        self.energy_cost_scale = 0.01
        self.base_product_value = 5.0

        root_node_indices_dict = {
            "x_reaction_rate": [0, 1, 2],
            "x_yield_frac": [3, 4],
            "x_energy_cost": [5, 6, 7, 8],
            "x_net_output": [9, 10, 11],
        }

        super().__init__(
            dim=dim,
            bounds=bounds,
            observed_output_node_names=[
                "reaction_rate",
                "yield_frac",
                "energy_cost",
                "net_output",
            ],
            root_node_indices_dict=root_node_indices_dict,
            objective_node_name="net_output",
            negate=False,
            process_stochasticity_std=process_stochasticity_std,
            observation_noise_std=observation_noise_std,
        )

    # ------------------------------------------------------------------ #
    #                         core simulator                             #
    # ------------------------------------------------------------------ #
    def _evaluate_true(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        xr = input_dict["x_reaction_rate"]  # [T, C_A, m]
        xy = input_dict["x_yield_frac"]  # [t, η_cat_raw]
        xe = input_dict["x_energy_cost"]  # P₁…P₄
        xn = input_dict["x_net_output"]  # [φ_m, m_in, φ_env]

        # ---------- reaction_rate ---------------------------------------
        T = xr[..., 0] + 300.0
        C_A = torch.clamp(xr[..., 1], min=0.01)
        m = torch.sigmoid(xr[..., 2])  # 0–1

        rate_base = self.A0 * torch.exp(-self.Ea / (self.R * T))
        mix_term = C_A ** (1 + 0.2 * m) * m  # nonlinear CA–mix coupling
        osc_term = 1.0 + 0.1 * torch.sin(T / 50.0)  # temp oscillation
        r_rate = rate_base * mix_term * osc_term
        if self.is_stochastic:
            r_rate = r_rate + torch.randn_like(r_rate) * self.process_stochasticity_std

        # ---------- yield_frac ------------------------------------------
        t = torch.clamp(xy[..., 0], min=0.1)
        eta_cat = 1.0 + torch.sigmoid(xy[..., 1])  # 1–2
        cyc = 0.1 * torch.sin(2 * math.pi * t / 50.0)  # periodic catalyst swing
        y_raw = 1.0 - torch.exp(-r_rate * t * eta_cat) + cyc
        y_frac = torch.clamp(y_raw, 0.0, 1.0)
        if self.is_stochastic:
            y_frac = y_frac + torch.randn_like(y_frac) * self.process_stochasticity_std

        # ---------- energy_cost -----------------------------------------
        P_vec = xe  # shape (...,4)
        quad = self.energy_cost_scale * torch.sum(P_vec**2, dim=-1)
        temp_pen = 0.005 * (T - 300.0) ** 2
        cross = 5e-4 * torch.sum(P_vec, dim=-1) ** 2
        e_cost = quad + temp_pen + cross
        if self.is_stochastic:
            e_cost = e_cost + torch.randn_like(e_cost) * self.process_stochasticity_std

        # ---------- net_output ------------------------------------------
        phi_m = xn[..., 0]
        m_in = torch.clamp(xn[..., 1], min=0.1)
        phi_env = xn[..., 2]

        product_val = self.base_product_value + 0.5 * phi_m
        cost_scale = torch.exp(0.1 * phi_env)

        net_out = product_val * y_frac * m_in - cost_scale * e_cost

        return {
            "reaction_rate": r_rate,
            "yield_frac": y_frac,
            "energy_cost": e_cost,
            "net_output": net_out,
        }

    def _evaluate_noisy(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        outs = self._evaluate_true(input_dict)
        if self.observation_noise_std and self.observation_noise_std > 0:
            for k, v in outs.items():
                outs[k] = v + torch.randn_like(v) * self.observation_noise_std
        return outs
