from .base.dag_experiment_base import DAGSyntheticTestFunction
from typing import Dict, Optional, Tuple, List
import torch
import torch.nn.functional as F


class ServiceNetworkPCDirectEnterprise(DAGSyntheticTestFunction):
    r"""
    ServiceNetworkPCDirect — Multi-service network with pricing/marketing, capacity,
    and routing; waits directly reduce realized throughput (abandonment/satisfaction).
    """

    def __init__(
        self,
        K: int = 5,
        S: int = 30,  # Number of services and stations
        dim: int = 190,  # Will be calculated based on K and S
        # noise (decoupled from time window)
        observation_noise_std: Optional[float] = 0.02,
        process_stochasticity_std: Optional[float] = 0.02,
        # system (not optimized)
        window_T: float = 1.0,
        output_as_rate: bool = True,
        # bounds (replicated per block if None)
        bounds: Optional[List[Tuple[float, float]]] = None,
        # demand & costs (fixed for legibility)
        lambda0=40.0,  # Increased for a robust, established business
        alpha=0.25,
        beta_m=0.20,
        # abandonment sensitivity (fixed for legibility)
        gamma_ab=0.1,
        kappa=2.5,  # Increased penalty to avoid severe congestion
        c_m=0.15,  # Decreased to reflect marketing efficiency
        c_mu=0.3,  # Decreased to reflect economies of scale
        hetero=True,
        # smoothness & heterogeneity
        soft_beta: float = 10.0,
    ):
        self.K, self.S = int(K), int(S)
        self.window_T = float(window_T)
        self.output_as_rate = bool(output_as_rate)

        if dim != (2 * self.K + self.S + self.K * self.S):
            raise ValueError(f"dim must be {2 * self.K + self.S + self.K * self.S}")

        if bounds is None:
            bounds = (
                [(12.0, 25.0)] * K  # prices p_k: higher starting point for profit
                + [(0.0, 3.0)] * K  # marketing m_k: limit costs
                + [(4.0, 10.0)]
                * S  # capacities μ_s: lower max capacity to reduce costs
                + [(-2.0, 2.0)]
                * (K * S)  # routing logits: maintains a wide range for flexibility
            )

        # index blocks
        idx_p = list(range(0, self.K))
        idx_m = list(range(self.K, 2 * self.K))
        idx_mu = list(range(2 * self.K, 2 * self.K + self.S))
        idx_z = list(range(2 * self.K + self.S, dim))

        observed_output_node_names: List[str] = ["y1", "y2", "y3"]
        root_node_indices_dict: Dict[str, List[int]] = {
            "p": idx_p,
            "m": idx_m,
            "mu": idx_mu,
            "z": idx_z,
        }
        objective_node_name = "y3"

        super().__init__(
            dim=dim,
            bounds=bounds,
            observed_output_node_names=observed_output_node_names,
            root_node_indices_dict=root_node_indices_dict,
            objective_node_name=objective_node_name,
            negate=False,  # maximize profit
            process_stochasticity_std=process_stochasticity_std,
            observation_noise_std=observation_noise_std,
        )

        self.lambda0 = float(lambda0)
        self.alpha = float(alpha)
        self.beta_m = float(beta_m)
        self.c_mu = float(c_mu)
        self.c_m = float(c_m)
        self.kappa = float(kappa)
        self.gamma_ab = float(gamma_ab)
        self.soft_beta = float(soft_beta)
        self.hetero = bool(hetero)
        self._eps = 1e-6

    # smooth min for elementwise tensors
    def _softmin(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        beta = torch.as_tensor(self.soft_beta, dtype=a.dtype, device=a.device)
        return (
            -torch.log(torch.exp(-beta * a) + torch.exp(-beta * b) + self._eps) / beta
        )

    def _make_hetero(self, K: int, device, dtype):
        k = torch.arange(K, device=device, dtype=dtype)
        lam_scale = 1.0 + 0.25 * torch.sin(0.7 * (k + 1))
        alpha_scale = 1.0 + 0.15 * torch.cos(0.9 * (k + 1))
        beta_scale = 1.0 + 0.10 * torch.sin(0.5 * (k + 2))
        return lam_scale, alpha_scale, beta_scale

    def _evaluate_true(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        p = input_dict["p"]  # [B,K]
        m = input_dict["m"]  # [B,K]
        mu = input_dict["mu"]  # [B,S]
        z = input_dict["z"]  # [B,K*S]

        B, K, S = p.shape[0], self.K, self.S
        z = z.view(B, K, S)
        w = torch.softmax(z, dim=-1)  # routing weights per service across stations

        # deterministic heterogeneity on elasticities/levels (for realism)
        lam_scale, alpha_scale, beta_scale = (1.0, 1.0, 1.0)
        if self.hetero:
            lam_scale, alpha_scale, beta_scale = self._make_hetero(K, p.device, p.dtype)

        # f1: demand per service (rate)
        lam = (
            self.lambda0
            * lam_scale
            * torch.exp(
                -(self.alpha * alpha_scale) * p + (self.beta_m * beta_scale) * m
            )
        )  # [B,K]

        # --- process stochasticity injected at f1 (customer uncertainty) ---
        if self.is_stochastic:
            s_proc = self.process_stochasticity_std
            lam = (lam + torch.randn_like(lam) * s_proc).clamp_min(self._eps)

        # f2: station loads and waits (computed from noisy lam so noise propagates)
        load = (w * lam.unsqueeze(-1)).sum(dim=1)  # [B,S]
        denom = (mu - load).clamp_min(self._eps)
        W_s = 1.0 / denom  # [B,S]
        y2_latent = W_s.mean(dim=1)  # [B]

        # capacity available to each service and base throughput
        cap_k = (w * mu.unsqueeze(1)).sum(dim=2)  # [B,K]
        tau0 = self._softmin(lam, cap_k)  # [B,K]

        # direct influence: abandonment via perceived wait per service
        Wk = (w * W_s.unsqueeze(1)).sum(dim=2)  # [B,K]
        tau = tau0 * torch.exp(-self.gamma_ab * Wk)  # [B,K]

        # profit rate Π
        overload_pen = F.softplus(load - mu).sum(dim=1)  # [B]
        profit_rate = (
            (p * tau).sum(dim=1)
            - self.c_mu * mu.sum(dim=1)
            - self.c_m * m.sum(dim=1)
            - self.kappa * overload_pen
        )

        # y1 rate from the (possibly noisy) lam; rate→total if requested
        y1_rate = lam.sum(dim=1)  # [B]
        if self.output_as_rate:
            y1, y2, y3 = y1_rate, y2_latent, profit_rate
        else:
            y1, y2, y3 = y1_rate * self.window_T, y2_latent, profit_rate * self.window_T

        return {"y1": y1, "y2": y2, "y3": y3}

    def _evaluate_noisy(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        out = self._evaluate_true(input_dict)
        s_obs = self.observation_noise_std
        if s_obs and s_obs > 0:
            for k in ("y1", "y2", "y3"):
                out[k] = out[k] + torch.randn_like(out[k]) * s_obs
        return out
