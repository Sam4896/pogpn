from typing import Dict, Optional, Tuple, List
import math
import torch
from .base.dag_experiment_base import DAGSyntheticTestFunction


class MultiStageLaunchSystem(DAGSyntheticTestFunction):
    r"""Three-stage launch system with environmental disturbance affecting final orbit insertion.

    Inputs:
    | Group       | Inputs          | Interpretation                                  |
    | ----------- | --------------- | ----------------------------------------------- |
    | Stage 1     | `x[0 : g1]`     | Booster config: fuel mix, angle, nozzle shape   |
    | Stage 2     | `x[g1 : g1+g2]` | Structural stage mass, material choices, layout |
    | Stage 3     | `x[...]`        | Final stage: vector control, inertial tuning    |
    | Environment | `x[remaining]`  | Wind speed, thermal variance, turbulence        |

    Realistic Behavior Modeled
    Exponential performance per stage: captures real-world nonlinear thrust/mass dynamics

    Synergy terms: captures inter-stage efficiency gains

    Environmental penalty: bounded, smooth loss (no discontinuity) using tanh

    Objective: Final mission score = payload delivery margin (e.g., delta-v surplus)

    \begin{align*}
    y_1 &= T_1 \cdot \exp(k_1 \cdot \|x_1\|) \\
    y_2 &= T_2 \cdot \exp(k_2 \cdot \|x_2\|) \cdot (1 + \alpha y_1) \\
    y_3 &= T_3 \cdot \exp(k_3 \cdot \|x_3\|) \cdot (1 + \beta y_2) \\
    y_4 &= A \cdot \tanh(\gamma \cdot \text{mean}(x_{\text{env}})) \\
    y_5 &= y_3 - y_4
    \end{align*}


    \tikzset{every node/.style={draw, minimum size=1.2cm}, node distance=1.7cm}
    \begin{tikzpicture}[->, thick]
    \node (x1) {x₁ (Stage 1)};
    \node (y1) [below of=x1] {y₁};

    \node (x2) [right of=x1] {x₂ (Stage 2)};
    \node (y2) [below of=x2] {y₂};
    \draw (x1) -- (y1); \draw (y1) -- (y2); \draw (x2) -- (y2);

    \node (x3) [right of=x2] {x₃ (Stage 3)};
    \node (y3) [below of=x3] {y₃};
    \draw (x3) -- (y3); \draw (y2) -- (y3);

    \node (xe) [right of=x3] {xₑ (Env)};
    \node (y4) [below of=xe] {y₄ (Env)};
    \draw (xe) -- (y4);

    \node (y5) [below of=y2, yshift=-3cm] {y₅ (Mission)};
    \draw (y3) -- (y5); \draw (y4) -- (y5);
    \end{tikzpicture}
    """

    def __init__(
        self,
        dim: int,
        negate: bool = True,
        observation_noise_std: Optional[float] = None,
        process_stochasticity_std: Optional[float] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ):
        if dim <= 10:
            raise ValueError("dim must be > 10 for this launch system model.")
        if bounds is None:
            bounds = [(-10.0, 10.0) for _ in range(dim)]

        # Partition dimensions into 3 stage inputs and 1 environmental group
        self.g1 = math.ceil(dim / 4)
        remaining = dim - self.g1
        self.g2 = math.ceil(remaining / 3)
        remaining -= self.g2
        self.g3 = math.ceil(remaining / 2)
        self.g_env = dim - self.g1 - self.g2 - self.g3  # remainder to env inputs

        # Constants modeled after typical thrust and structure dynamics
        self.T1, self.T2, self.T3 = 100.0, 80.0, 60.0  # thrust base values
        self.k1, self.k2, self.k3 = 0.05, 0.04, 0.03  # exponential growth rates
        self.synergy_12, self.synergy_23 = 0.02, 0.01  # amplification from stage i-1
        self.env_amp = 20.0
        self.env_sensitivity = 0.1

        observed_output_node_names = [
            "stage1_perf",
            "stage2_perf",
            "stage3_perf",
            "env_disturbance",
            "mission_score",
        ]
        root_node_indices = {"x": list(range(dim))}
        objective_node_name = "mission_score"

        super().__init__(
            dim=dim,
            bounds=bounds,
            observed_output_node_names=observed_output_node_names,
            root_node_indices=root_node_indices,
            objective_node_name=objective_node_name,
            negate=negate,
            process_stochasticity_std=process_stochasticity_std,
            observation_noise_std=observation_noise_std,
        )

    def _evaluate_true(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        x = input_dict["x"]

        # Stage 1 performance
        base1 = torch.norm(x[..., : self.g1], dim=-1) / math.sqrt(self.g1)
        stage1 = self.T1 * torch.exp(self.k1 * base1)
        stage1_perf = (
            stage1 + torch.randn_like(stage1) * self.process_stochasticity_std
            if self.is_stochastic
            else stage1
        )

        # Stage 2: synergy from stage 1
        x2 = x[..., self.g1 : self.g1 + self.g2]
        base2 = torch.norm(x2, dim=-1) / math.sqrt(self.g2)
        stage2 = (
            self.T2 * torch.exp(self.k2 * base2) * (1 + self.synergy_12 * stage1_perf)
        )
        stage2_perf = (
            stage2 + torch.randn_like(stage2) * self.process_stochasticity_std
            if self.is_stochastic
            else stage2
        )

        # Stage 3: synergy from stage 2
        start3 = self.g1 + self.g2
        x3 = x[..., start3 : start3 + self.g3]
        base3 = torch.norm(x3, dim=-1) / math.sqrt(self.g3)
        stage3 = (
            self.T3 * torch.exp(self.k3 * base3) * (1 + self.synergy_23 * stage2_perf)
        )
        stage3_perf = (
            stage3 + torch.randn_like(stage3) * self.process_stochasticity_std
            if self.is_stochastic
            else stage3
        )

        # Environment effect (e.g., wind, turbulence)
        start_env = self.g1 + self.g2 + self.g3
        if self.g_env > 0:
            x_env = x[..., start_env:]
            env_input = torch.mean(x_env, dim=-1)
        else:
            env_input = torch.zeros_like(x[..., 0])
        env_disturbance = self.env_amp * torch.tanh(self.env_sensitivity * env_input)
        if self.is_stochastic:
            env_disturbance += (
                torch.randn_like(env_disturbance) * self.process_stochasticity_std
            )

        # Final score = total thrust - environmental loss
        mission_score = stage3_perf - env_disturbance

        return {
            "stage1_perf": stage1_perf,
            "stage2_perf": stage2_perf,
            "stage3_perf": stage3_perf,
            "env_disturbance": env_disturbance,
            "mission_score": mission_score,
        }

    def _evaluate_noisy(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        output = self._evaluate_true(input_dict)
        for key in output:
            output[key] += torch.randn_like(output[key]) * self.observation_noise_std
        return output
