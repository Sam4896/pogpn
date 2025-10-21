from typing import Dict, Optional
import math
import numpy as np


FAILURE_MESP = 1000.0


class FullDynamicBiorefinerySDE:
    """Stochastic (SDE) variant of the bioethanol economic model using Euler–Maruyama.

    This mirrors the deterministic model structure but injects small process stochasticity
    into the ODEs for each subprocess (pretreatment, enzyme production, SSF), using the
    same noise level for all via a single `process_stochasticity_std`.

    Noise model (per step i):
        y_{i+1} = y_i + f(t_i, y_i) * dt + sigma * sqrt(dt) * N(0, I)

    The drift f(·) matches the original deterministic RHS. States are clamped to be
    non-negative after each step for physical realism.
    """

    def __init__(
        self,
        *,
        process_stochasticity_std: float = 0.01,
        pretreatment_em_steps: int = 400,
        enzyme_em_steps: int = 800,
        ssf_em_steps: int = 1200,
        rng: Optional[np.random.Generator] = None,
    ):
        # --- Stage 1: Pretreatment (Saeman Kinetics) Parameters ---
        self.PRE_R_GAS = 8.314e-3  # kJ/(mol*K)
        self.PRE_A1 = 1.0e15  # 1/min
        self.PRE_Ea1 = 140.0  # kJ/mol
        self.PRE_n1 = 1.1
        self.PRE_A2 = 1.0e14
        self.PRE_Ea2 = 130.0
        self.PRE_n2 = 1.0
        self.PRE_A3 = 1.0e13
        self.PRE_Ea3 = 150.0
        self.PRE_n3 = 1.1

        # --- Stage 2: Enzyme Production (Luedeking–Piret) Parameters ---
        self.ENZ_mu_max = 0.15  # 1/h
        self.ENZ_Ks = 0.5  # g/L
        self.ENZ_alpha = 0.05  # g_enzyme / g_biomass
        self.ENZ_beta = 0.005  # g_enzyme / (g_biomass * h)
        self.ENZ_Y_XS = 0.5  # g_biomass / g_substrate
        self.ENZ_maint = 0.01  # g_substrate / (g_biomass * h)

        # --- Stage 3: SSF (Monod/Inhibition) Parameters ---
        self.SSF_Vmax_hyd_base = 0.9  # g_cellulose / (L * h)
        self.SSF_Km_hyd = 15.0  # g/L
        self.SSF_E_to_Vmax_factor = 0.01

        self.SSF_mu_max_base = 0.4  # 1/h
        self.SSF_Ks_growth = 1.5  # g/L
        self.SSF_T_opt_growth = 36.0  # °C
        self.SSF_T_bw_growth = 10.0  # °C

        # Inhibition constants
        self.SSF_Ki_G_hyd = 5.0  # g/L
        self.SSF_Ki_E_hyd = 80.0  # g/L
        self.SSF_C_E_max_growth = 90.0  # g/L
        self.SSF_Ki_fur_growth = 2.5  # g/L
        self.SSF_Ki_fur_ferm = 5.0  # g/L

        # Stoichiometry
        self.SSF_Y_XG = 0.18  # g_biomass / g_glucose
        self.SSF_Y_EG = 0.45  # g_ethanol / g_glucose
        self.SSF_m_E = 0.02  # 1/h

        # --- Stage 4 & TEA Parameters ---
        self.PLANT_CAPACITY_KG_FEED_PER_YEAR = 80_000_000
        self.FIXED_COSTS_PER_YEAR = 120_000_000

        # Variable Costs
        self.COST_FEEDSTOCK_PER_KG = 0.08
        self.COST_ENZ_SUBSTRATE_PER_KG = 0.30
        self.COST_ACID_PER_KG = 0.05
        self.COST_BASE_PER_KG = 0.40
        self.COST_YEAST_PER_KG = 1.50
        self.COST_STEAM_PER_KG = 0.02

        # Distillation Energy Correlation
        self.DIST_A = 0.5
        self.DIST_B = 120.0
        self.DIST_C = -1.1

        # Conversions & Feedstock Composition
        self.ETHANOL_DENSITY_KG_L = 0.789
        self.INITIAL_XYLAN_FRAC = 0.22
        self.INITIAL_GLUCAN_FRAC = 0.35

        # Stochastic integration settings
        self.process_stochasticity_std = float(process_stochasticity_std)
        self.pretreatment_em_steps = int(pretreatment_em_steps)
        self.enzyme_em_steps = int(enzyme_em_steps)
        self.ssf_em_steps = int(ssf_em_steps)
        self.rng = rng if rng is not None else np.random.default_rng()

    @staticmethod
    def _is_finite(value):
        return not (math.isnan(value) or math.isinf(value))

    # --------------------------- EM Integrator --------------------------- #
    def _euler_maruyama(
        self,
        f,
        y0: np.ndarray,
        t0: float,
        t1: float,
        n_steps: int,
        sigma: float,
        args: tuple = (),
        clamp_nonneg_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        n_steps = max(1, int(n_steps))
        dt = (t1 - t0) / n_steps
        dt = max(dt, 1e-9)
        y = np.asarray(y0, dtype=float).copy()
        t = float(t0)
        dim = y.shape[0]

        for _ in range(n_steps):
            drift = np.asarray(f(t, y, *args), dtype=float)
            if not np.all(np.isfinite(drift)):
                raise ValueError("SDE drift produced non-finite values")
            if sigma > 0:
                noise = self.rng.normal(0.0, 1.0, size=dim) * math.sqrt(dt) * sigma
            else:
                noise = 0.0
            y = y + drift * dt + noise
            if clamp_nonneg_indices is not None and clamp_nonneg_indices.size > 0:
                y[clamp_nonneg_indices] = np.maximum(0.0, y[clamp_nonneg_indices])
            t += dt
        return y

    # ---------------------- Stage 1: Pretreatment ----------------------- #
    def _pre_ode_model(self, t, y, k1, k2, k3):
        """ODE system for Saeman kinetics."""
        C_Xylan, C_Xylose, C_Furfural, C_Glucan = y
        C_Xylan, C_Xylose, C_Glucan = (
            max(0.0, C_Xylan),
            max(0.0, C_Xylose),
            max(0.0, C_Glucan),
        )

        dC_Xylan_dt = -k1 * C_Xylan
        dC_Xylose_dt = k1 * C_Xylan - k2 * C_Xylose
        dC_Furfural_dt = k2 * C_Xylose
        dC_Glucan_dt = -k3 * C_Glucan
        return [dC_Xylan_dt, dC_Xylose_dt, dC_Furfural_dt, dC_Glucan_dt]

    def _solve_pretreatment(
        self, T_pretreat, t_pretreat, acid_conc_percent, solids_loading_pretreat
    ) -> Dict[str, float]:
        T_K = T_pretreat + 273.15

        k1 = (
            self.PRE_A1
            * math.exp(-self.PRE_Ea1 / (self.PRE_R_GAS * T_K))
            * (acid_conc_percent**self.PRE_n1)
        )
        k2 = (
            self.PRE_A2
            * math.exp(-self.PRE_Ea2 / (self.PRE_R_GAS * T_K))
            * (acid_conc_percent**self.PRE_n2)
        )
        k3 = (
            self.PRE_A3
            * math.exp(-self.PRE_Ea3 / (self.PRE_R_GAS * T_K))
            * (acid_conc_percent**self.PRE_n3)
        )

        # Initial concentrations based on solids loading (g/L)
        C_Xylan_0 = solids_loading_pretreat * self.INITIAL_XYLAN_FRAC
        C_Glucan_0 = solids_loading_pretreat * self.INITIAL_GLUCAN_FRAC
        y0 = np.array([C_Xylan_0, 0.0, 0.0, C_Glucan_0], dtype=float)

        yT = self._euler_maruyama(
            self._pre_ode_model,
            y0=y0,
            t0=0.0,
            t1=float(t_pretreat),
            n_steps=self.pretreatment_em_steps,
            sigma=self.process_stochasticity_std,
            args=(k1, k2, k3),
            clamp_nonneg_indices=np.array([0, 1, 2, 3], dtype=int),
        )

        C_Glucan_final = float(yT[3])
        C_Furfural_final = float(yT[2])

        return {
            "S_0_glucan_available": C_Glucan_final,
            "furfural_conc_g_L": C_Furfural_final,
        }

    # ------------------ Stage 2: Enzyme Production SDE ------------------ #
    def _enz_ode_model(self, t, y):
        C_Biomass, C_Product, C_Substrate = y
        C_Biomass, C_Substrate = max(0.0, C_Biomass), max(0.0, C_Substrate)

        mu = self.ENZ_mu_max * (C_Substrate / (self.ENZ_Ks + C_Substrate))
        r_growth = mu * C_Biomass
        r_product = (self.ENZ_alpha * mu + self.ENZ_beta) * C_Biomass
        r_substrate = (r_growth / self.ENZ_Y_XS) + self.ENZ_maint * C_Biomass

        dC_Biomass_dt = r_growth
        dC_Product_dt = r_product
        dC_Substrate_dt = -r_substrate
        return [dC_Biomass_dt, dC_Product_dt, dC_Substrate_dt]

    def _solve_enzyme_prod(self, t_enzyme_batch):
        # Initial: X_0, P_0, S_0 (g/L)
        y0_enz = np.array([0.1, 0.0, 50.0], dtype=float)

        yT = self._euler_maruyama(
            self._enz_ode_model,
            y0=y0_enz,
            t0=0.0,
            t1=float(t_enzyme_batch),
            n_steps=self.enzyme_em_steps,
            sigma=self.process_stochasticity_std,
            args=(),
            clamp_nonneg_indices=np.array([0, 1, 2], dtype=int),
        )

        final_product_g_L = float(yT[1])
        final_substrate_g_L = float(yT[2])
        total_substrate_used_g_L = max(0.0, 50.0 - final_substrate_g_L)

        if final_product_g_L <= 1e-6:
            return 1.0  # $/mg (penalty)

        cost_per_L_batch = total_substrate_used_g_L * (
            self.COST_ENZ_SUBSTRATE_PER_KG / 1000.0
        )
        cost_per_g_enzyme = cost_per_L_batch / max(final_product_g_L, 1e-12)
        return cost_per_g_enzyme / 1000.0  # $/mg

    # -------------------- Stage 3: SSF Dynamic SDE ---------------------- #
    def _ssf_ode_model(self, t, y, T, E_load_Vmax, C_furfural):
        C_C, C_G, C_X, C_E = y
        C_C, C_G, C_X, C_E = max(0.0, C_C), max(0.0, C_G), max(0.0, C_X), max(0.0, C_E)

        # 1. Temperature effect
        f_T = max(0.01, 1 - ((T - self.SSF_T_opt_growth) / self.SSF_T_bw_growth) ** 2)
        mu_max_T = self.SSF_mu_max_base * f_T

        # 2. Furfural inhibition
        f_inh_growth = max(0.01, (1 - C_furfural / self.SSF_Ki_fur_growth))
        f_inh_ferm = max(0.01, (1 - C_furfural / self.SSF_Ki_fur_ferm))

        # 3. Hydrolysis
        hyd_inhibition = max(
            0.0, (1 - C_G / self.SSF_Ki_G_hyd) * (1 - C_E / self.SSF_Ki_E_hyd)
        )
        v_hyd = E_load_Vmax * (C_C / (self.SSF_Km_hyd + C_C)) * hyd_inhibition

        # 4. Growth rate
        growth_inhibition = max(0.0, (1 - C_E / self.SSF_C_E_max_growth))
        mu_max_inhibited = mu_max_T * f_inh_growth
        mu = mu_max_inhibited * (C_G / (self.SSF_Ks_growth + C_G)) * growth_inhibition

        # 5. Fermentation rate
        v_ferm_specific = ((self.SSF_Y_EG * mu) + self.SSF_m_E) * f_inh_ferm
        v_ferm = v_ferm_specific * C_X

        # 6. Glucose consumption
        v_growth = mu * C_X
        v_gluc_cons = (v_growth / self.SSF_Y_XG) + (v_ferm / self.SSF_Y_EG)

        # Limit glucose drain to reasonable bound
        if t > 0:
            v_gluc_cons = min(v_gluc_cons, v_hyd + (C_G / 0.1))
        else:
            v_gluc_cons = min(v_gluc_cons, (C_G / 0.1))

        dC_C_dt = -v_hyd
        dC_G_dt = v_hyd - v_gluc_cons
        dC_X_dt = v_growth
        dC_E_dt = v_ferm
        return [dC_C_dt, dC_G_dt, dC_X_dt, dC_E_dt]

    def _solve_ssf(
        self, T_ssf, E_load_ssf, S_0_glucan, batch_time_ssf, C_furfural_initial, SSF_X_0
    ) -> Dict[str, float]:
        E_load_Vmax = E_load_ssf * self.SSF_E_to_Vmax_factor * S_0_glucan
        y0 = np.array([S_0_glucan, 0.0, SSF_X_0, 0.0], dtype=float)

        yT = self._euler_maruyama(
            self._ssf_ode_model,
            y0=y0,
            t0=0.0,
            t1=float(batch_time_ssf),
            n_steps=self.ssf_em_steps,
            sigma=self.process_stochasticity_std,
            args=(T_ssf, E_load_Vmax, C_furfural_initial),
            clamp_nonneg_indices=np.array([0, 1, 2, 3], dtype=int),
        )
        C_E_final = float(yT[3])
        return {"C_E_final_g_L": C_E_final}

    # ------------------ Stage 4: Distillation (Algebraic) --------------- #
    def _calculate_distillation_cost(self, C_E_final_g_L) -> float:
        if C_E_final_g_L < 10.0:
            return 10.0
        steam_needed_kg_per_L = self.DIST_A + self.DIST_B * (C_E_final_g_L**self.DIST_C)
        cost_per_L_etoh = steam_needed_kg_per_L * self.COST_STEAM_PER_KG
        return cost_per_L_etoh

    # ----------------------- Main Objective (MESP) ----------------------- #
    def calculate_mesp(
        self,
        T_pretreat,
        t_pretreat,
        acid_conc_percent,
        solids_loading_pretreat,
        t_enzyme_batch,
        S_0_loading_ssf,
        T_ssf,
        E_load_ssf,
        batch_time_ssf,
        SSF_X_0,
        return_details=False,
    ):
        details = {
            "S_0_glucan_ssf": None,
            "furfural_conc_ssf": None,
            "cost_per_mg_enzyme": None,
            "C_E_final_g_L": None,
            "MESP": FAILURE_MESP,
        }

        try:
            # 1) Pretreatment
            pre_results = self._solve_pretreatment(
                T_pretreat, t_pretreat, acid_conc_percent, solids_loading_pretreat
            )

            # 2) Dilution into SSF
            if solids_loading_pretreat <= 0:
                raise ValueError("solids_loading_pretreat must be > 0")
            if S_0_loading_ssf <= 0:
                raise ValueError("S_0_loading_ssf must be > 0")

            if solids_loading_pretreat < S_0_loading_ssf:
                dilution_factor = 1.0
            else:
                dilution_factor = S_0_loading_ssf / solids_loading_pretreat

            S_0_glucan_ssf = pre_results["S_0_glucan_available"] * dilution_factor
            furfural_conc_ssf = pre_results["furfural_conc_g_L"] * dilution_factor
            details["S_0_glucan_ssf"] = S_0_glucan_ssf
            details["furfural_conc_ssf"] = furfural_conc_ssf

            # 3) Enzyme production
            cost_per_mg_enzyme = self._solve_enzyme_prod(t_enzyme_batch)
            details["cost_per_mg_enzyme"] = cost_per_mg_enzyme

            # 4) SSF
            ssf_results = self._solve_ssf(
                T_ssf=T_ssf,
                E_load_ssf=E_load_ssf,
                S_0_glucan=S_0_glucan_ssf,
                batch_time_ssf=batch_time_ssf,
                C_furfural_initial=furfural_conc_ssf,
                SSF_X_0=SSF_X_0,
            )
            C_E_final = ssf_results["C_E_final_g_L"]
            details["C_E_final_g_L"] = C_E_final

            # 5) Distillation & TEA
            if C_E_final <= 1.0:
                if return_details:
                    return details
                return FAILURE_MESP

            variable_cost_distillation = self._calculate_distillation_cost(C_E_final)

            g_etoh_per_g_raw_ssf = C_E_final / S_0_loading_ssf
            L_etoh_per_kg_raw = g_etoh_per_g_raw_ssf / self.ETHANOL_DENSITY_KG_L
            total_annual_ethanol_L = (
                L_etoh_per_kg_raw * self.PLANT_CAPACITY_KG_FEED_PER_YEAR
            )

            if total_annual_ethanol_L <= 1.0:
                if return_details:
                    return details
                return FAILURE_MESP

            total_g_raw_per_year = self.PLANT_CAPACITY_KG_FEED_PER_YEAR * 1000
            cost_feedstock = (
                self.COST_FEEDSTOCK_PER_KG * self.PLANT_CAPACITY_KG_FEED_PER_YEAR
            )

            glucan_frac_in_ssf_solids = S_0_glucan_ssf / S_0_loading_ssf
            total_mg_enzyme_year = (
                E_load_ssf * glucan_frac_in_ssf_solids * total_g_raw_per_year
            )
            cost_enzyme = total_mg_enzyme_year * cost_per_mg_enzyme
            cost_distillation_annual = (
                variable_cost_distillation * total_annual_ethanol_L
            )
            total_kg_acid = (
                acid_conc_percent / 100.0
            ) * self.PLANT_CAPACITY_KG_FEED_PER_YEAR
            cost_acid = total_kg_acid * self.COST_ACID_PER_KG
            total_kg_base = total_kg_acid * (2 * 17.03) / 98.08
            cost_base = total_kg_base * self.COST_BASE_PER_KG
            total_L_ssf_per_year = total_g_raw_per_year / S_0_loading_ssf
            total_g_yeast_per_year = SSF_X_0 * total_L_ssf_per_year
            total_kg_yeast_per_year = total_g_yeast_per_year / 1000.0
            cost_yeast = total_kg_yeast_per_year * self.COST_YEAST_PER_KG

            TAC = (
                self.FIXED_COSTS_PER_YEAR
                + cost_feedstock
                + cost_enzyme
                + cost_distillation_annual
                + cost_acid
                + cost_base
                + cost_yeast
            )

            MESP = TAC / total_annual_ethanol_L
            details["MESP"] = MESP

            if return_details:
                return details
            return MESP

        except Exception:
            if return_details:
                return details
            return FAILURE_MESP


# --- END OF CLASS ---

# 1) Initialize the SDE simulator (small default stochasticity)
biorefinery_simulator = FullDynamicBiorefinerySDE(process_stochasticity_std=0.01)


# 2) Objective function for BO (returns details for convenience)
def bo_objective_function(
    T_pretreat,
    t_pretreat,
    acid_conc_percent,
    solids_loading_pretreat,
    t_enzyme_batch,
    S_0_loading_ssf,
    T_ssf,
    E_load_ssf,
    batch_time_ssf,
    SSF_X_0,
):
    return biorefinery_simulator.calculate_mesp(
        T_pretreat=T_pretreat,
        t_pretreat=t_pretreat,
        acid_conc_percent=acid_conc_percent,
        solids_loading_pretreat=solids_loading_pretreat,
        t_enzyme_batch=t_enzyme_batch,
        S_0_loading_ssf=S_0_loading_ssf,
        T_ssf=T_ssf,
        E_load_ssf=E_load_ssf,
        batch_time_ssf=batch_time_ssf,
        SSF_X_0=SSF_X_0,
        return_details=True,
    )


pbounds = {
    # Stage 1: Pretreatment
    "T_pretreat": (160.0, 195.0),  # °C
    "t_pretreat": (5.0, 20.0),  # minutes
    "acid_conc_percent": (0.5, 2.0),  # % w/w
    "solids_loading_pretreat": (150.0, 250.0),  # g/L
    # Stage 2: Enzyme Production
    "t_enzyme_batch": (48.0, 120.0),  # hours
    # Stage 3: SSF
    "S_0_loading_ssf": (100.0, 200.0),  # g/L
    "T_ssf": (32.0, 48.0),  # °C
    "E_load_ssf": (10.0, 40.0),  # mg/g
    "batch_time_ssf": (60.0, 144.0),  # hours
    "SSF_X_0": (0.5, 5.0),  # g/L
}


if __name__ == "__main__":
    print("Biorefinery SDE model is ready for Bayesian Optimization.")

    base_params = {
        "T_pretreat": 165.0,
        "t_pretreat": 10.0,
        "acid_conc_percent": 1.0,
        "solids_loading_pretreat": 200.0,
        "t_enzyme_batch": 72.0,
        "S_0_loading_ssf": 150.0,
        "T_ssf": 37.0,
        "E_load_ssf": 20.0,
        "batch_time_ssf": 96.0,
        "SSF_X_0": 1.0,
    }

    try:
        details = bo_objective_function(**base_params)
        print(
            "Baseline SDE run: MESP=$%.3f/L, C_E_final=%.2f g/L, furfural=%.2f g/L"
            % (
                details.get("MESP", FAILURE_MESP),
                details.get("C_E_final_g_L", float("nan")),
                details.get("furfural_conc_ssf", float("nan")),
            )
        )
    except Exception as e:
        print(f"[ERROR] Baseline SDE run failed: {e}")
