"""Bioethanol two-stage process as a DAG synthetic test function.

Implements a pretreatment stage followed by SSCF using a torch-based simulator.
"""

from typing import Dict, List, Optional, Tuple

import torch

from .base.dag_experiment_base import DAGSyntheticTestFunction


class _BioEthanolSimulator:
    """Torch simulator for a two-stage bioethanol process.

    Stage 1 (Pretreatment): Xylan -> Xylose -> Furfural with Arrhenius kinetics.
    Stage 2 (SSCF): Fed-batch saccharification and co-fermentation with simple rate laws.

    This is implemented fully in torch and vectorized across a batch of inputs.
    """

    def __init__(
        self,
        observation_noise_std: Optional[float] = None,
        process_stochasticity_std: Optional[float] = None,
        pretreatment_steps: int = 120,
        sscf_steps: int = 480,
        sscf_dt_h: float = 0.1,
    ) -> None:
        self.observation_noise_std = float(observation_noise_std or 0.0)
        self.process_stochasticity_std = float(process_stochasticity_std or 0.0)
        self.pretreatment_steps = int(pretreatment_steps)
        self.sscf_steps = int(sscf_steps)
        self.sscf_dt_h = float(sscf_dt_h)

        # Stoichiometry for pretreatment
        self.f_XYL = 1.136  # g xylose / g xylan
        self.f_FUR = 0.727  # g furfural / g xylose

        # Feedstock composition
        self.slurry_density_g_per_L = 1050.0
        self.xylan_mass_fraction = 0.20
        self.cellulose_mass_fraction = 0.35

        # SSCF constants (representative; see literature for exact values)
        self.total_volume_L = 3_596_000.0
        self.initial_fill_fraction = 0.10
        self.feed_duration_h_default = 12.0
        self.total_batch_time_h_default = 48.0

        # Hydrolysis (Sin et al. inspired placeholders)
        self.p_hyd = {
            "kr1": 22.3,
            "kr2": 7.18,
            "kr3": 285.5,
            "CE1_max": 0.06,
            "CE2_max": 0.01,
            "Kad1": 0.4,
            "Kad2": 0.1,
            "K1IG": 0.1,
            "K2IG": 0.04,
            "K3IG": 3.9,
            "K1IXy": 3.0,
            "K2IXy": 0.2,
            "K3IXy": 2.5,
            "K1IC": 0.3,
            "K1IEt": 3.0,
        }

        # Fermentation kinetics (representative)
        self.p_ferm = {
            "mu_max_G": 0.662,
            "mu_max_Xy": 0.190,
            "v_max_G": 2.005,
            "v_max_Xy": 0.250,
            "Y_XG": 0.11,  # gX produced per g glucose consumed
            "Y_XXy": 0.11,  # gX produced per g xylose consumed
            "m_G": 0.029,
            "m_Xy": 0.029,
            "KG": 1.66,
            "KXy": 5.38,
            "KEtG": 19.3,
            "KEtXy": 19.3,
            "CEt_max_G": 95.4,
            "CEt_max_Xy": 59.04,
            "CEt_prime_max_G": 103.0,
            "CEt_prime_max_Xy": 60.20,
        }

        # Furfural inhibition/detox (representative)
        self.p_inhib = {"K_if": 1.5, "k_detox": 0.15}

    @staticmethod
    def _arrhenius(
        a: float, ea: float, n: float, acid_wt_pct: torch.Tensor, t_c: torch.Tensor
    ) -> torch.Tensor:
        r = 8.314
        t_k = t_c + 273.15
        return a * (acid_wt_pct.clamp_min(1e-9) ** n) * torch.exp(-ea / (r * t_k))

    def _pretreatment(
        self,
        solids_loading_percent: torch.Tensor,
        pretreatment_temp_c: torch.Tensor,
        acid_conc_wt_percent: torch.Tensor,
        residence_time_min: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        # Initial concentrations from solids loading
        solids_loading_fraction = solids_loading_percent / 100.0
        initial_xylan_conc = (
            solids_loading_fraction
            * self.slurry_density_g_per_L
            * self.xylan_mass_fraction
        )

        # Lavarack-like parameters
        a1 = 1.87e17
        ea1 = 140000.0
        n1 = 1.13
        a2 = 2.36e13
        ea2 = 125000.0
        n2 = 1.01

        k1 = self._arrhenius(a1, ea1, n1, acid_conc_wt_percent, pretreatment_temp_c)
        k2 = self._arrhenius(a2, ea2, n2, acid_conc_wt_percent, pretreatment_temp_c)

        # Time discretization
        t_final = residence_time_min.clamp_min(1e-6)
        n_steps = self.pretreatment_steps
        # Per-sample dt so we keep vectorization; approximate using mean dt
        dt = (t_final / float(n_steps)).to(device=device, dtype=dtype)

        # State init
        xylan = initial_xylan_conc.clone()
        xylose = torch.zeros_like(xylan)
        furfural = torch.zeros_like(xylan)

        for _ in range(n_steps):
            dxylan_dt = -k1 * xylan
            dxylose_dt = self.f_XYL * k1 * xylan - k2 * xylose
            dfurfural_dt = self.f_FUR * k2 * xylose

            xylan = (xylan + dxylan_dt * dt).clamp_min(0.0)
            xylose = (xylose + dxylose_dt * dt).clamp_min(0.0)
            furfural = (furfural + dfurfural_dt * dt).clamp_min(0.0)

        return {
            "final_xylose_conc": xylose,
            "final_furfural_conc": furfural,
            "initial_cellulose_conc": solids_loading_fraction
            * self.slurry_density_g_per_L
            * self.cellulose_mass_fraction,
        }

    def _sscf(
        self,
        sscf_temp_c: torch.Tensor,
        enzyme_loading_g_per_g_cellulose: torch.Tensor,
        yeast_inoculum_gl: torch.Tensor,
        xylose_feed_in_g_per_l: torch.Tensor,
        furfural_feed_in_g_per_l: torch.Tensor,
        initial_cellulose_conc: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        # Initial conditions
        v0 = torch.full_like(
            yeast_inoculum_gl, self.total_volume_L * self.initial_fill_fraction
        )
        total_time_h = torch.full_like(
            yeast_inoculum_gl, self.total_batch_time_h_default
        )
        feed_duration_h = torch.full_like(
            yeast_inoculum_gl, self.feed_duration_h_default
        )
        feed_rate_l_h = (self.total_volume_L - v0) / feed_duration_h.clamp_min(1e-6)

        cs = initial_cellulose_conc.clone()  # cellulose (g/L)
        cg2 = torch.zeros_like(cs)  # oligomers
        cg1 = torch.zeros_like(cs)  # glucose
        cxy = torch.zeros_like(cs)  # xylose
        cx = yeast_inoculum_gl.clone()  # biomass
        cet = torch.zeros_like(cs)  # ethanol
        cef1 = enzyme_loading_g_per_g_cellulose * initial_cellulose_conc * 0.9
        cef2 = enzyme_loading_g_per_g_cellulose * initial_cellulose_conc * 0.1
        cfur = torch.zeros_like(cs)
        v = v0.clone()

        # Constants and modifiers
        # Growth/product inhibition by ethanol; furfural inhibition
        p_hyd = self.p_hyd
        p_ferm = self.p_ferm
        p_inhib = self.p_inhib

        # Time grid
        n_steps = min(
            int(torch.ceil(total_time_h / self.sscf_dt_h).max().item()), self.sscf_steps
        )
        dt = torch.full_like(cs, self.sscf_dt_h)

        for i in range(n_steps):
            t = dt * (i + 1)
            q_in = torch.where(
                t <= feed_duration_h, feed_rate_l_h, torch.zeros_like(feed_rate_l_h)
            )

            # Hydrolysis enzyme binding and rates
            r_reactivity = torch.where(
                cs > 1e-6, cs / cs.clamp_min(1e-6), torch.zeros_like(cs)
            )
            ceb1 = (p_hyd["CE1_max"] * p_hyd["Kad1"] * cs * cef1) / (
                1.0 + p_hyd["Kad1"] * cs
            )
            ceb2 = (p_hyd["CE2_max"] * p_hyd["Kad2"] * cs * cef2) / (
                1.0 + p_hyd["Kad2"] * cs
            )

            r1_denom = (
                1.0
                + (cg2 / p_hyd["K1IC"])
                + (cg1 / p_hyd["K1IG"])
                + (cxy / p_hyd["K1IXy"])
                + (cet / p_hyd["K1IEt"])
            )
            r1 = (p_hyd["kr1"] * ceb1 * r_reactivity * cs) / (r1_denom + 1e-6)
            r2_denom = 1.0 + (cg1 / p_hyd["K2IG"]) + (cxy / p_hyd["K2IXy"])
            r2 = (p_hyd["kr2"] * ceb1 * r_reactivity * cs) / (r2_denom + 1e-6)
            r3_denom = 1.0 + (cg1 / p_hyd["K3IG"]) + (cxy / p_hyd["K3IXy"])
            r3 = (p_hyd["kr3"] * ceb2 * cg2) / (r3_denom + 1e-6)

            # Fermentation growth and ethanol production
            growth_inhibition_g = (1.0 - cet / p_ferm["CEt_max_G"]).clamp_min(0.0)
            growth_inhibition_xy = (1.0 - cet / p_ferm["CEt_max_Xy"]).clamp_min(0.0)
            furfural_inhibition = p_inhib["K_if"] / (p_inhib["K_if"] + cfur + 1e-6)

            r_x_g = (
                (p_ferm["mu_max_G"] * cx * cg1 / (p_ferm["KG"] + cg1 + 1e-6))
                * growth_inhibition_g
                * furfural_inhibition
            )
            r_x_xy = (
                (p_ferm["mu_max_Xy"] * cx * cxy / (p_ferm["KXy"] + cxy + 1e-6))
                * growth_inhibition_xy
                * furfural_inhibition
            )
            r_x_tot = r_x_g + r_x_xy

            # Substrate consumption (biomass yield + maintenance)
            r_consum_g = -(r_x_g / (p_ferm["Y_XG"] + 1e-6)) - p_ferm["m_G"] * cx
            r_consum_xy = -(r_x_xy / (p_ferm["Y_XXy"] + 1e-6)) - p_ferm["m_Xy"] * cx

            # Ethanol production
            prod_inhibition_g = (1.0 - cet / p_ferm["CEt_prime_max_G"]).clamp_min(0.0)
            prod_inhibition_xy = (1.0 - cet / p_ferm["CEt_prime_max_Xy"]).clamp_min(0.0)
            r_et_g = (
                (p_ferm["v_max_G"] * cx * cg1 / (p_ferm["KG"] + cg1 + 1e-6))
                * prod_inhibition_g
                * furfural_inhibition
            )
            r_et_xy = (
                (p_ferm["v_max_Xy"] * cx * cxy / (p_ferm["KXy"] + cxy + 1e-6))
                * prod_inhibition_xy
                * furfural_inhibition
            )
            r_et_tot = r_et_g + r_et_xy

            # Furfural detox
            r_fur_detox = -self.p_inhib["k_detox"] * cx * cfur

            # Fed-batch balances
            v_safe = v.clamp_min(1e-6)

            dcs_dt = (q_in / v_safe) * (torch.zeros_like(cs) - cs) - r1 - r2
            dcg2_dt = (q_in / v_safe) * (torch.zeros_like(cs) - cg2) + r1 - r3
            dcg1_dt = (
                (q_in / v_safe) * (torch.zeros_like(cs) - cg1) + r2 + r3 + r_consum_g
            )
            dcxy_dt = (q_in / v_safe) * (xylose_feed_in_g_per_l - cxy) + r_consum_xy
            dcx_dt = (q_in / v_safe) * (torch.zeros_like(cs) - cx) + r_x_tot
            dcet_dt = (q_in / v_safe) * (torch.zeros_like(cs) - cet) + r_et_tot
            dcef1_dt = (q_in / v_safe) * (torch.zeros_like(cs) - cef1)
            dcef2_dt = (q_in / v_safe) * (torch.zeros_like(cs) - cef2)
            dcfur_dt = (q_in / v_safe) * (furfural_feed_in_g_per_l - cfur) + r_fur_detox
            dv_dt = q_in

            cs = (cs + dcs_dt * dt).clamp_min(0.0)
            cg2 = (cg2 + dcg2_dt * dt).clamp_min(0.0)
            cg1 = (cg1 + dcg1_dt * dt).clamp_min(0.0)
            cxy = (cxy + dcxy_dt * dt).clamp_min(0.0)
            cx = (cx + dcx_dt * dt).clamp_min(0.0)
            cet = (cet + dcet_dt * dt).clamp_min(0.0)
            cef1 = (cef1 + dcef1_dt * dt).clamp_min(0.0)
            cef2 = (cef2 + dcef2_dt * dt).clamp_min(0.0)
            cfur = (cfur + dcfur_dt * dt).clamp_min(0.0)
            v = (v + dv_dt * dt).clamp_min(1e-6)

        final_v = v
        final_cet = cet

        # Overall sugar potential (stoichiometric factors similar to dummy)
        total_sugar_potential = (
            initial_cellulose_conc * v0
            + feed_rate_l_h * feed_duration_h * torch.zeros_like(cs)
        ) * 1.111
        total_sugar_potential = total_sugar_potential + (
            torch.zeros_like(cs) * v0
            + feed_rate_l_h * feed_duration_h * xylose_feed_in_g_per_l
        )

        ethanol_yield = (final_cet * final_v) / (total_sugar_potential + 1e-6)
        vol_productivity = final_cet / total_time_h.clamp_min(1e-6)

        return {
            "final_ethanol_conc": final_cet,
            "ethanol_yield": ethanol_yield,
            "volumetric_productivity": vol_productivity,
            "final_V": final_v,
        }

    def _run(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = input_dict["inputs"]
        device, dtype = x.device, x.dtype

        (
            solids_loading_percent,
            pretreatment_temp_c,
            acid_conc_wt_percent,
            residence_time_min,
            sscf_temp_c,
            enzyme_loading_g_per_g_cellulose,
            yeast_inoculum_gl,
        ) = (x[..., i] for i in range(7))

        # Stage 1
        pret = self._pretreatment(
            solids_loading_percent=solids_loading_percent,
            pretreatment_temp_c=pretreatment_temp_c,
            acid_conc_wt_percent=acid_conc_wt_percent,
            residence_time_min=residence_time_min,
            device=device,
            dtype=dtype,
        )

        # Stage 2 (xylose and furfural feed are outputs of Stage 1)
        sscf = self._sscf(
            sscf_temp_c=sscf_temp_c,
            enzyme_loading_g_per_g_cellulose=enzyme_loading_g_per_g_cellulose,
            yeast_inoculum_gl=yeast_inoculum_gl,
            xylose_feed_in_g_per_l=pret["final_xylose_conc"],
            furfural_feed_in_g_per_l=pret["final_furfural_conc"],
            initial_cellulose_conc=pret["initial_cellulose_conc"],
            device=device,
            dtype=dtype,
        )

        return {
            "pretreatment_final_xylose_conc": pret["final_xylose_conc"],
            "pretreatment_final_furfural_conc": pret["final_furfural_conc"],
            "sscf_final_ethanol_conc": sscf["final_ethanol_conc"],
            "sscf_ethanol_yield": sscf["ethanol_yield"],
            "sscf_volumetric_productivity": sscf["volumetric_productivity"],
        }

    def _evaluate_true(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return self._run(input_dict)

    def _evaluate_noisy(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        out = self._run(input_dict)
        sigma = self.observation_noise_std
        if sigma > 0.0:
            for k in list(out.keys()):
                out[k] = out[k] + torch.randn_like(out[k]) * sigma
        return out


class BioEthanolProcess(DAGSyntheticTestFunction):
    """Two-stage bioethanol process as a DAG-style synthetic test function.

    Inputs (dim=7):
      0 solids_loading_percent [10, 40]
      1 pretreatment_temp_C [150, 200]
      2 pretreatment_acid_conc_wt_percent [0.5, 2.0]
      3 pretreatment_residence_time_min [1, 15]
      4 sscf_temp_C [30, 38]
      5 sscf_enzyme_loading_g_per_g_cellulose [0.005, 0.05]
      6 sscf_yeast_inoculum_gL [0.2, 5.0]
    """

    def __init__(
        self,
        dim: int = 7,
        observation_noise_std: Optional[float] = None,
        process_stochasticity_std: Optional[float] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        """Create the bioethanol DAG test function.

        Args:
            dim: Input dimension, must be 7 per definition.
            observation_noise_std: Additive observation noise std.
            process_stochasticity_std: Process noise std for simulator.
            bounds: Optional bounds for each input dimension.

        """
        if dim != 7:
            raise ValueError("dim must be 7")

        if bounds is None:
            bounds = [
                (10.0, 40.0),  # solids_loading_percent
                (150.0, 200.0),  # pretreatment_temp_C
                (0.5, 2.0),  # acid conc wt%
                (1.0, 15.0),  # residence time (min)
                (30.0, 38.0),  # sscf temp C
                (0.005, 0.05),  # enzyme loading (g/g cellulose)
                (0.2, 5.0),  # yeast inoculum (g/L)
            ]

        observed_output_node_names: List[str] = [
            "pretreatment_final_xylose_conc",
            "pretreatment_final_furfural_conc",
            "sscf_final_ethanol_conc",
            "sscf_ethanol_yield",
            "sscf_volumetric_productivity",
        ]
        root_node_indices_dict: Dict[str, List[int]] = {"inputs": list(range(dim))}
        objective_node_name = "sscf_final_ethanol_conc"

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

        self.simulator = _BioEthanolSimulator(
            observation_noise_std=observation_noise_std,
            process_stochasticity_std=process_stochasticity_std,
        )

    def _evaluate_true(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return self.simulator._evaluate_true(input_dict)

    def _evaluate_noisy(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return self.simulator._evaluate_noisy(input_dict)
