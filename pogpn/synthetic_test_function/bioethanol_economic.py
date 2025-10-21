# --- CITATIONS FOR THE MODELS ---
#
# 1. Overall Process Design & MESP Logic:
#    - Humbird, D., et al. (2011). "Process Design and Economics for
#      Biochemical Conversion of Lignocellulosic Biomass to Ethanol."
#      NREL Technical Report (NREL/TP-5100-47764).
#
# 2. Pretreatment Kinetics (Stage 1):
#    - Saeman, J. F. (1945). "Kinetics of wood saccharification at
#      high temperatures." *Industrial & Engineering Chemistry*.
#    - Aguilar, R., et al. (2002). "Kinetic study of the acid
#      hydrolysis of sugarcane bagasse." *Journal of Food Engineering*.
#
# 3. Enzyme Production Kinetics (Stage 2):
#    - Luedeking, R., & Piret, E. L. (1959). "A kinetic study of the
#      lactic acid fermentation." *Journal of Biochemical and
#      Microbiological Technology and Engineering*.
#
# 4. SSF Kinetics & Inhibition (Stage 3):
#    - Palmqvist, E., & Hahn-Hägerdal, B. (2000). "Fermentation of
#      lignocellulosic hydrolysates. II: inhibitors and mechanisms
#      of inhibition." *Bioresource Technology*.
#
# 5. Distillation Energy Correlation (Stage 4):
#    - Based on process simulations detailed in Humbird et al. (2011),
#      which show a strong inverse correlation between fermentation
#      titer and steam (energy) requirement.
#
# --- END OF CITATIONS ---
from scipy.integrate import solve_ivp
import math
from typing import Dict

FAILURE_MESP = 1000.0


class FullDynamicBiorefinery:
    """Simulate a multi-stage bioethanol process using citable DYNAMIC (ODE).

    models for Pretreatment and Enzyme Production, linked to the
    dynamic SSF model. This final version includes SSF_X_0 as a
    controllable BO variable.

    This class is intended to be the objective function for a
    Bayesian Optimization algorithm, which will seek to minimize the
    output of the `calculate_mesp` method.

    Stages:
    1. Pretreatment (Dynamic ODEs): Saeman Kinetics
    2. Enzyme Production (Dynamic ODEs): Luedeking-Piret Kinetics
    3. SSF (Dynamic ODEs): Monod/Inhibition Kinetics
    4. Distillation (Algebraic): Energy/Cost Correlation

    All stages are interconnected, and the final MESP calculation
    accounts for all capital and variable operating costs.
    """

    def __init__(self):
        """Initialize the model with citable kinetic and economic parameters.

        These are fixed parameters, not Bayesian Optimization (BO) variables.
        """
        # --- Stage 1: Pretreatment (Saeman Kinetics) Parameters ---
        # Citation: Saeman (1945); Aguilar et al. (2002)
        self.PRE_R_GAS = 8.314e-3  # kJ/(mol*K)
        # k = A * exp(-Ea / (R * T)) * [Acid %]^n
        # k1: Xylan -> Xylose
        self.PRE_A1 = 1.0e15  # 1/min
        self.PRE_Ea1 = 140.0  # kJ/mol
        self.PRE_n1 = 1.1  # Reaction order for acid
        # k2: Xylose -> Furfural
        self.PRE_A2 = 1.0e14
        self.PRE_Ea2 = 130.0
        self.PRE_n2 = 1.0
        # k3: Glucan -> Glucose (and HMF)
        self.PRE_A3 = 1.0e13
        self.PRE_Ea3 = 150.0
        self.PRE_n3 = 1.1

        # --- Stage 2: Enzyme Production (Luedeking-Piret) Parameters ---
        # Citation: Mandenius & Liden (2010); Luedeking & Piret (1959)
        self.ENZ_mu_max = 0.15  # 1/h (Max growth rate of T. reesei)
        self.ENZ_Ks = 0.5  # g/L (Substrate affinity)
        self.ENZ_alpha = 0.05  # g_enzyme / g_biomass (Growth-associated)
        self.ENZ_beta = 0.005  # g_enzyme / (g_biomass * h) (Non-growth-associated)
        self.ENZ_Y_XS = 0.5  # g_biomass / g_substrate
        self.ENZ_maint = 0.01  # g_substrate / (g_biomass * h)

        # --- Stage 3: SSF (Monod/Inhibition) Parameters ---
        # Citation: Palmqvist & Hahn-Hägerdal (2000); NREL 47764
        self.SSF_Vmax_hyd_base = 0.9  # g_cellulose / (L * h)
        self.SSF_Km_hyd = 15.0  # g/L (Michaelis-Menten)
        self.SSF_E_to_Vmax_factor = 0.01  # Converts (mg/g) to Vmax

        self.SSF_mu_max_base = 0.4  # 1/h (Max yeast growth)
        self.SSF_Ks_growth = 1.5  # g/L (Monod constant)
        self.SSF_T_opt_growth = 36.0  # °C (Optimal yeast temp)
        self.SSF_T_bw_growth = 10.0  # °C (Temp sensitivity)

        # Inhibition constants
        self.SSF_Ki_G_hyd = 5.0  # g/L (Glucose inhibition on hydrolysis)
        self.SSF_Ki_E_hyd = 80.0  # g/L (Ethanol inhibition on hydrolysis)
        self.SSF_C_E_max_growth = 90.0  # g/L (Max ethanol tolerance for yeast)
        self.SSF_Ki_fur_growth = 2.5  # g/L (Furfural inhibition on growth)
        self.SSF_Ki_fur_ferm = 5.0  # g/L (Furfural inhibition on fermentation)

        # Stoichiometry
        self.SSF_Y_XG = 0.18  # g_biomass / g_glucose
        self.SSF_Y_EG = 0.45  # g_ethanol / g_glucose
        self.SSF_m_E = 0.02  # 1/h (Maintenance coefficient)

        # --- Stage 4 & TEA Parameters ---
        # Citation: Humbird et al. (2011) (NREL 47764)
        self.PLANT_CAPACITY_KG_FEED_PER_YEAR = 80_000_000  # 80k metric tons
        self.FIXED_COSTS_PER_YEAR = 120_000_000  # $120M / year (CAPEX + Fixed OPEX)

        # Variable Costs
        self.COST_FEEDSTOCK_PER_KG = 0.08  # $ / kg raw biomass (e.g., corn stover)
        self.COST_ENZ_SUBSTRATE_PER_KG = 0.30  # $ / kg glucose for enzyme prod.
        self.COST_ACID_PER_KG = 0.05  # $ / kg H2SO4
        self.COST_BASE_PER_KG = 0.40  # $ / kg Ammonia (for neutralization)
        self.COST_YEAST_PER_KG = 1.50  # $ / kg dry yeast
        self.COST_STEAM_PER_KG = 0.02  # $ / kg steam for distillation

        # Distillation Energy Correlation
        self.DIST_A = 0.5  # Base steam load
        self.DIST_B = 120.0  # Scaling factor
        self.DIST_C = -1.1  # Inverse exponent (low titer = high cost)

        # Conversions & Feedstock Composition
        self.ETHANOL_DENSITY_KG_L = 0.789
        self.INITIAL_XYLAN_FRAC = 0.22  # % xylan in raw biomass
        self.INITIAL_GLUCAN_FRAC = 0.35  # % glucan in raw biomass

    @staticmethod
    def _is_finite(value):
        return not (math.isnan(value) or math.isinf(value))

    # --- STAGE 1: PRETREATMENT DYNAMIC MODEL ---

    def _pre_ode_model(self, t, y, k1, k2, k3):
        """ODE system for Saeman kinetics."""
        try:
            C_Xylan, C_Xylose, C_Furfural, C_Glucan = y
            C_Xylan, C_Xylose, C_Glucan = (
                max(0, C_Xylan),
                max(0, C_Xylose),
                max(0, C_Glucan),
            )

            dC_Xylan_dt = -k1 * C_Xylan
            dC_Xylose_dt = k1 * C_Xylan - k2 * C_Xylose
            dC_Furfural_dt = k2 * C_Xylose
            dC_Glucan_dt = -k3 * C_Glucan  # Simplified: tracks glucan remaining

            derivs = [dC_Xylan_dt, dC_Xylose_dt, dC_Furfural_dt, dC_Glucan_dt]
            if not all(self._is_finite(v) for v in derivs):
                print(
                    f"[DEBUG] PRE rhs non-finite at t={t}: y={y}, k1={k1}, k2={k2}, k3={k3}, derivs={derivs}"
                )
                raise ValueError("PRE ODE RHS produced non-finite derivative")

            return derivs
        except Exception as e:
            print(f"[ERROR] _pre_ode_model exception at t={t}, y={y}: {e}")
            raise

    def _solve_pretreatment(
        self, T_pretreat, t_pretreat, acid_conc_percent, solids_loading_pretreat
    ) -> Dict[str, float]:
        """Solves the dynamic pretreatment simulation.

        Args:
            T_pretreat: The pretreatment temperature in Celsius.
            t_pretreat: The pretreatment time in minutes.
            acid_conc_percent: The acid concentration in percent.
            solids_loading_pretreat: The solids loading in percent.

        Returns:
            A dictionary containing the final glucan concentration and furfural concentration.
            "S_0_glucan_available": The final glucan concentration in g/L.
            "furfural_conc_g_L": The final furfural concentration in g/L.

        """
        try:
            T_K = T_pretreat + 273.15

            # Calculate Arrhenius rates (k) based on T and Acid
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

            if not all(self._is_finite(v) for v in [k1, k2, k3]):
                print(
                    f"[DEBUG] Pretreatment rates not finite: k1={k1}, k2={k2}, k3={k3}"
                )
                raise ValueError("Non-finite pretreatment rate constant")

            # Initial concentrations based on solids loading (g/L)
            C_Xylan_0 = solids_loading_pretreat * self.INITIAL_XYLAN_FRAC
            C_Glucan_0 = solids_loading_pretreat * self.INITIAL_GLUCAN_FRAC
            y0 = [C_Xylan_0, 0.0, 0.0, C_Glucan_0]  # Xylan, Xylose, Furfural, Glucan

            sol = solve_ivp(
                self._pre_ode_model,
                [0, t_pretreat],
                y0,
                method="RK45",
                args=(k1, k2, k3),
            )

            if not sol.success:
                print(f"[DEBUG] Pretreatment solver failed: {sol.message}")
                raise RuntimeError("Pretreatment ODE solver failed")

            C_Glucan_final = sol.y[3, -1]
            C_Furfural_final = sol.y[2, -1]

            if not all(self._is_finite(v) for v in [C_Glucan_final, C_Furfural_final]):
                print(
                    f"[DEBUG] Pretreatment outputs not finite: C_Glucan_final={C_Glucan_final}, "
                    f"C_Furfural_final={C_Furfural_final}"
                )
                raise ValueError("Non-finite pretreatment outputs")

            return {
                "S_0_glucan_available": C_Glucan_final,  # g/L in the pre-hydrolysate
                "furfural_conc_g_L": C_Furfural_final,  # g/L in the pre-hydrolysate
            }
        except Exception as e:
            print(
                f"[ERROR] _solve_pretreatment failed with inputs: T={T_pretreat}, t={t_pretreat}, "
                f"acid%={acid_conc_percent}, solids={solids_loading_pretreat}; error={e}"
            )
            raise

    # --- STAGE 2: ENZYME PRODUCTION DYNAMIC MODEL ---

    def _enz_ode_model(self, t, y):
        """ODE system for Luedeking-Piret kinetics in a batch."""
        try:
            C_Biomass, C_Product, C_Substrate = y
            C_Biomass, C_Substrate = max(0, C_Biomass), max(0, C_Substrate)

            mu = self.ENZ_mu_max * (C_Substrate / (self.ENZ_Ks + C_Substrate))

            r_growth = mu * C_Biomass
            r_product = (self.ENZ_alpha * mu + self.ENZ_beta) * C_Biomass
            r_substrate = (r_growth / self.ENZ_Y_XS) + self.ENZ_maint * C_Biomass

            if not all(
                self._is_finite(v) for v in [mu, r_growth, r_product, r_substrate]
            ):
                print(
                    f"[DEBUG] ENZ rates non-finite at t={t}: y={y}, mu={mu}, rg={r_growth}, rp={r_product}, rs={r_substrate}"
                )
                raise ValueError("ENZ rates non-finite")

            dC_Biomass_dt = r_growth
            dC_Product_dt = r_product
            dC_Substrate_dt = -r_substrate
            derivs = [dC_Biomass_dt, dC_Product_dt, dC_Substrate_dt]
            if not all(self._is_finite(v) for v in derivs):
                print(f"[DEBUG] ENZ rhs non-finite at t={t}: derivs={derivs}")
                raise ValueError("ENZ ODE RHS produced non-finite derivative")
            return derivs
        except Exception as e:
            print(f"[ERROR] _enz_ode_model exception at t={t}, y={y}: {e}")
            raise

    def _solve_enzyme_prod(self, t_enzyme_batch):
        """Solves the enzyme production batch simulation.

        Args:
            t_enzyme_batch: The enzyme production time in minutes.

        Returns:
            The variable cost ($/mg) for the enzyme.
            "cost_per_g_enzyme": The cost per gram of enzyme in $/g.

        """
        try:
            y0_enz = [0.1, 0.0, 50.0]  # X_0, P_0, S_0 (g/L)
            sol = solve_ivp(
                self._enz_ode_model, [0, t_enzyme_batch], y0_enz, method="RK45"
            )

            if not sol.success:
                print(f"[DEBUG] Enzyme solver failed: {sol.message}")
                raise RuntimeError("Enzyme production ODE solver failed")

            final_product_g_L = sol.y[1, -1]
            total_substrate_used_g_L = y0_enz[2] - sol.y[2, -1]

            if not all(
                self._is_finite(v)
                for v in [final_product_g_L, total_substrate_used_g_L]
            ):
                print(
                    f"[DEBUG] Enzyme outputs not finite: P_final={final_product_g_L}, "
                    f"substrate_used={total_substrate_used_g_L}"
                )
                raise ValueError("Non-finite enzyme outputs")

            if final_product_g_L <= 1e-6:
                return 1.0  # High penalty cost ($/mg)

            # Cost ($/L) = substrate cost
            cost_per_L_batch = total_substrate_used_g_L * (
                self.COST_ENZ_SUBSTRATE_PER_KG / 1000.0
            )

            # Final cost in $/g of enzyme
            cost_per_g_enzyme = cost_per_L_batch / final_product_g_L

            if not self._is_finite(cost_per_g_enzyme):
                print(
                    f"[DEBUG] Enzyme cost not finite: cost_per_g_enzyme={cost_per_g_enzyme}"
                )
                raise ValueError("Non-finite enzyme cost")

            # Return cost in $/mg
            return cost_per_g_enzyme / 1000.0
        except Exception as e:
            print(
                f"[ERROR] _solve_enzyme_prod failed with t_enzyme_batch={t_enzyme_batch}; error={e}"
            )
            raise

    # --- STAGE 3: SSF DYNAMIC MODEL ---

    def _ssf_ode_model(self, t, y, T, E_load_Vmax, C_furfural):
        """Define the system of ODEs for SSF, including inhibitor effects.

        Args:
            t: The time in minutes.
            y: The initial conditions for the ODEs.
            T: The temperature in Celsius.
            E_load_Vmax: The enzyme load in Vmax units.
            C_furfural: The furfural concentration in g/L.
        """
        try:
            C_C, C_G, C_X, C_E = y
            C_C, C_G, C_X, C_E = max(0, C_C), max(0, C_G), max(0, C_X), max(0, C_E)

            # 1. Temperature Effect on Yeast
            f_T = max(
                0.01, 1 - ((T - self.SSF_T_opt_growth) / self.SSF_T_bw_growth) ** 2
            )
            mu_max_T = self.SSF_mu_max_base * f_T

            # 2. Furfural Inhibition Effect
            f_inh_growth = max(0.01, (1 - C_furfural / self.SSF_Ki_fur_growth))
            f_inh_ferm = max(0.01, (1 - C_furfural / self.SSF_Ki_fur_ferm))

            # 3. Hydrolysis Rate (Cellulose -> Glucose)
            hyd_inhibition = max(
                0, (1 - C_G / self.SSF_Ki_G_hyd) * (1 - C_E / self.SSF_Ki_E_hyd)
            )
            v_hyd = E_load_Vmax * (C_C / (self.SSF_Km_hyd + C_C)) * hyd_inhibition

            # 4. Yeast Specific Growth Rate (mu)
            growth_inhibition = max(0, (1 - C_E / self.SSF_C_E_max_growth))
            mu_max_inhibited = mu_max_T * f_inh_growth
            mu = (
                mu_max_inhibited
                * (C_G / (self.SSF_Ks_growth + C_G))
                * growth_inhibition
            )

            # 5. Fermentation Rate (Glucose -> Ethanol)
            v_ferm_specific = ((self.SSF_Y_EG * mu) + self.SSF_m_E) * f_inh_ferm
            v_ferm = v_ferm_specific * C_X

            # 6. Glucose Consumption Rate
            v_growth = mu * C_X
            v_gluc_cons = (v_growth / self.SSF_Y_XG) + (v_ferm / self.SSF_Y_EG)
            v_gluc_cons = min(
                v_gluc_cons, v_hyd + (C_G / 0.1) if t > 0 else (C_G / 0.1)
            )

            key_vals = [
                f_T,
                mu_max_T,
                f_inh_growth,
                f_inh_ferm,
                hyd_inhibition,
                v_hyd,
                growth_inhibition,
                mu,
                v_ferm_specific,
                v_ferm,
                v_growth,
                v_gluc_cons,
            ]
            if not all(self._is_finite(v) for v in key_vals):
                print(
                    f"[DEBUG] SSF rates non-finite at t={t}: y={y}, T={T}, E_load_Vmax={E_load_Vmax}, furf={C_furfural}, vals={key_vals}"
                )
                raise ValueError("SSF rates non-finite")

            # --- System of ODEs ---
            dC_C_dt = -v_hyd
            dC_G_dt = v_hyd - v_gluc_cons
            dC_X_dt = v_growth
            dC_E_dt = v_ferm

            derivs = [dC_C_dt, dC_G_dt, dC_X_dt, dC_E_dt]
            if not all(self._is_finite(v) for v in derivs):
                print(f"[DEBUG] SSF rhs non-finite at t={t}: derivs={derivs}")
                raise ValueError("SSF ODE RHS produced non-finite derivative")
            return derivs
        except Exception as e:
            print(
                f"[ERROR] _ssf_ode_model exception at t={t}, y={y}, T={T}, E={E_load_Vmax}, furf={C_furfural}: {e}"
            )
            raise

    def _solve_ssf(
        self, T_ssf, E_load_ssf, S_0_glucan, batch_time_ssf, C_furfural_initial, SSF_X_0
    ) -> Dict[str, float]:
        """Solves the dynamic SSF simulation.

        Args:
            T_ssf: The temperature in Celsius.
            E_load_ssf: The enzyme load in mg/g.
            S_0_glucan: The initial glucan concentration in g/L.
            batch_time_ssf: The batch time in minutes.
            C_furfural_initial: The initial furfural concentration in g/L.
            SSF_X_0: The initial yeast concentration in g/L.

        Returns:
            A dictionary containing the final ethanol concentration in g/L.
            "C_E_final_g_L": The final ethanol concentration in g/L.

        """
        try:
            # Map E_load (mg/g) to Vmax for the model
            E_load_Vmax = E_load_ssf * self.SSF_E_to_Vmax_factor * S_0_glucan

            if not self._is_finite(E_load_Vmax):
                print(
                    f"[DEBUG] SSF E_load_Vmax not finite: E_load_ssf={E_load_ssf}, "
                    f"S_0_glucan={S_0_glucan}, factor={self.SSF_E_to_Vmax_factor}"
                )
                raise ValueError("Non-finite SSF E_load_Vmax")

            # Set initial conditions
            # y0 = [C_Cellulose, C_Glucose, C_Yeast, C_Ethanol]
            y0 = [S_0_glucan, 0.0, SSF_X_0, 0.0]  # <-- SSF_X_0 is used here

            if not all(self._is_finite(v) for v in y0):
                print(f"[DEBUG] SSF initial conditions not finite: y0={y0}")
                raise ValueError("Non-finite SSF initial conditions")

            sol = solve_ivp(
                self._ssf_ode_model,
                [0, batch_time_ssf],
                y0,
                method="RK45",
                args=(T_ssf, E_load_Vmax, C_furfural_initial),
            )

            if not sol.success:
                print(f"[DEBUG] SSF solver failed: {sol.message}")
                raise RuntimeError("SSF ODE solver failed")

            C_E_final = sol.y[3, -1]  # Final ethanol concentration (g/L)

            if not self._is_finite(C_E_final):
                print(f"[DEBUG] SSF ethanol output not finite: C_E_final={C_E_final}")
                raise ValueError("Non-finite SSF ethanol output")

            return {"C_E_final_g_L": C_E_final}
        except Exception as e:
            print(
                f"[ERROR] _solve_ssf failed with inputs: T={T_ssf}, E_load={E_load_ssf}, S0={S_0_glucan}, "
                f"t={batch_time_ssf}, furfural={C_furfural_initial}, X0={SSF_X_0}; error={e}"
            )
            raise

    # --- STAGE 4: DISTILLATION (ALGEBRAIC) ---

    def _calculate_distillation_cost(self, C_E_final_g_L) -> float:
        """Calculate the distillation energy cost based on final ethanol titer.

        Args:
            C_E_final_g_L: The final ethanol concentration in g/L.

        Returns:
            The distillation energy cost in $/L.

        """
        if not self._is_finite(C_E_final_g_L):
            print(
                f"[DEBUG] Distillation input not finite: C_E_final_g_L={C_E_final_g_L}"
            )
            raise ValueError("Non-finite distillation input")

        if C_E_final_g_L < 10.0:  # Below ~1% ethanol, cost is prohibitive
            return 10.0  # Return a high penalty cost ($/L)

        # steam_kg / L_EtOH = A + B * (Titer)^C
        steam_needed_kg_per_L = self.DIST_A + self.DIST_B * (C_E_final_g_L**self.DIST_C)

        if not self._is_finite(steam_needed_kg_per_L):
            print(
                f"[DEBUG] Distillation steam calc not finite: A={self.DIST_A}, B={self.DIST_B}, "
                f"C={self.DIST_C}, titer={C_E_final_g_L}, steam={steam_needed_kg_per_L}"
            )
            raise ValueError("Non-finite distillation steam requirement")

        cost_per_L_etoh = steam_needed_kg_per_L * self.COST_STEAM_PER_KG
        if not self._is_finite(cost_per_L_etoh):
            print(
                f"[DEBUG] Distillation cost not finite: cost_per_L_etoh={cost_per_L_etoh}"
            )
            raise ValueError("Non-finite distillation cost")
        return cost_per_L_etoh

    # --- MAIN OBJECTIVE FUNCTION FOR BO ---

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
        """Calculate the main objective function for the Bayesian Optimization."""
        # Prepare detail placeholders (default penalty MESP)
        details = {
            "S_0_glucan_ssf": None,
            "furfural_conc_ssf": None,
            "cost_per_mg_enzyme": None,
            "C_E_final_g_L": None,
            "MESP": FAILURE_MESP,
        }

        try:
            # --- 1. Run Pretreatment ODE Model (Stage 1) ---
            pre_results = self._solve_pretreatment(
                T_pretreat, t_pretreat, acid_conc_percent, solids_loading_pretreat
            )

            # --- 2. Calculate Dilution & SSF Inputs ---
            if solids_loading_pretreat <= 0:
                raise ValueError(
                    f"solids_loading_pretreat must be > 0, got {solids_loading_pretreat}"
                )
            if S_0_loading_ssf <= 0:
                raise ValueError(f"S_0_loading_ssf must be > 0, got {S_0_loading_ssf}")

            if solids_loading_pretreat < S_0_loading_ssf:
                dilution_factor = 1.0  # Cannot "undilute"
            else:
                dilution_factor = S_0_loading_ssf / solids_loading_pretreat

            S_0_glucan_ssf = pre_results["S_0_glucan_available"] * dilution_factor
            furfural_conc_ssf = pre_results["furfural_conc_g_L"] * dilution_factor
            details["S_0_glucan_ssf"] = S_0_glucan_ssf
            details["furfural_conc_ssf"] = furfural_conc_ssf

            if not all(self._is_finite(v) for v in [S_0_glucan_ssf, furfural_conc_ssf]):
                raise ValueError(
                    f"Non-finite Stage 1/2 interface: S0={S_0_glucan_ssf}, Furfural={furfural_conc_ssf}"
                )

            # --- 3. Run Enzyme Production ODE Model (Stage 2) ---
            cost_per_mg_enzyme = self._solve_enzyme_prod(t_enzyme_batch)
            details["cost_per_mg_enzyme"] = cost_per_mg_enzyme

            if not self._is_finite(cost_per_mg_enzyme):
                raise ValueError(
                    f"Non-finite Stage 2 enzyme cost: {cost_per_mg_enzyme}"
                )

            # --- 4. Run SSF Dynamic Model (Stage 3) ---
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

            if not self._is_finite(C_E_final):
                raise ValueError(f"Non-finite Stage 3 ethanol output: {C_E_final}")

            # --- 5. Run Distillation Cost Model (Stage 4) ---
            if C_E_final <= 1.0:
                print(
                    f"[WARNING] Ethanol titer too low ({C_E_final:.2f} g/L). Run failed."
                )
                if return_details:
                    return details
                return FAILURE_MESP

            variable_cost_distillation = self._calculate_distillation_cost(C_E_final)

            # --- 6. Calculate Final MESP (TEA) ---
            g_etoh_per_g_raw_ssf = C_E_final / S_0_loading_ssf
            # Corrected conversion: L/kg = (g/g) / (kg/L)
            L_etoh_per_kg_raw = g_etoh_per_g_raw_ssf / self.ETHANOL_DENSITY_KG_L
            total_annual_ethanol_L = (
                L_etoh_per_kg_raw * self.PLANT_CAPACITY_KG_FEED_PER_YEAR
            )

            if total_annual_ethanol_L <= 1.0:
                print(
                    f"[WARNING] Total annual ethanol L <= 1 ({total_annual_ethanol_L:.2f}). Run failed."
                )
                if return_details:
                    return details
                return FAILURE_MESP

            # --- Calculate Total Annualized Costs (TAC) ---
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

            if not self._is_finite(MESP):
                raise ValueError(
                    f"Final MESP non-finite: TAC={TAC}, L_total={total_annual_ethanol_L}"
                )

            if return_details:
                return details
            return MESP

        except Exception as e:
            print(f"[ERROR] calculate_mesp failed: {e}. Inputs: {locals()}")
            if return_details:
                return details
            return FAILURE_MESP


# --- END OF CLASS ---

# 1. Initialize the complete, interconnected system
biorefinery_simulator = FullDynamicBiorefinery()


# 2. Define the objective function for the BO to MINIMIZE
#    (Note: Some BO libraries maximize, so you would return -mesp)
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
    try:
        mesp_or_details = biorefinery_simulator.calculate_mesp(
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
        # mesp_or_details is a dict when return_details=True
        mesp = mesp_or_details.get("MESP", FAILURE_MESP)
        if mesp == FAILURE_MESP:
            print(
                f"[INFO] MESP returned {FAILURE_MESP}; inputs were: "
                f"T_pretreat={T_pretreat}, t_pretreat={t_pretreat}, acid%={acid_conc_percent}, "
                f"solids_pre={solids_loading_pretreat}, t_enzyme={t_enzyme_batch}, "
                f"S0_ssf={S_0_loading_ssf}, T_ssf={T_ssf}, E_load={E_load_ssf}, "
                f"batch_time={batch_time_ssf}, X0={SSF_X_0}"
            )
        return mesp_or_details
    except Exception as e:
        print(
            f"[ERROR] bo_objective_function failed with inputs: T_pretreat={T_pretreat}, "
            f"t_pretreat={t_pretreat}, acid%={acid_conc_percent}, solids_pre={solids_loading_pretreat}, "
            f"t_enzyme={t_enzyme_batch}, S0_ssf={S_0_loading_ssf}, T_ssf={T_ssf}, E_load={E_load_ssf}, "
            f"batch_time={batch_time_ssf}, X0={SSF_X_0}; error={e}"
        )
        raise


pbounds = {
    # Stage 1: Pretreatment
    "T_pretreat": (160.0, 195.0),  # °C
    "t_pretreat": (5.0, 20.0),  # minutes
    "acid_conc_percent": (0.5, 2.0),  # % w/w
    "solids_loading_pretreat": (150.0, 250.0),  # g/L (15-25% solids)
    # Stage 2: Enzyme Production
    "t_enzyme_batch": (48.0, 120.0),  # hours
    # Stage 3: SSF
    "S_0_loading_ssf": (100.0, 200.0),  # g/L (10-20% solids)
    "T_ssf": (32.0, 48.0),  # °C
    "E_load_ssf": (10.0, 40.0),  # mg enzyme / g glucan
    "batch_time_ssf": (60.0, 144.0),  # hours
    "SSF_X_0": (0.5, 5.0),  # g/L (Initial Yeast)
}

print("Biorefinery model is ready for Bayesian Optimization.")

# --- Example Call to Test the Model ---
print("\n--- Testing Model with Example Setpoints ---")

# Define a set of 10 "baseline" parameters
base_params = {
    "T_pretreat": 175.0,
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

# Test the function
try:
    example_details = bo_objective_function(**base_params)
    if isinstance(example_details, dict):
        print(
            f"Baseline intermediates: S_0_glucan_ssf={example_details.get('S_0_glucan_ssf')}, "
            f"furfural_conc_ssf={example_details.get('furfural_conc_ssf')}, "
            f"cost_per_mg_enzyme={example_details.get('cost_per_mg_enzyme')}, "
            f"C_E_final_g_L={example_details.get('C_E_final_g_L')}, "
            f"MESP=${example_details.get('MESP'):.3f} / L"
        )
        if example_details.get("MESP") == FAILURE_MESP:
            print(
                "[INFO] Baseline run returned {FAILURE_MESP}; see diagnostics above for the failing stage."
            )
    else:
        print(f"MESP at baseline conditions: ${example_details:.3f} / L")
except Exception as e:
    print(f"[ERROR] Baseline run failed: {e}")


print("\n--- Testing Model with 'MILDER' Setpoints ---")
# Let's try a MILDER pretreatment to reduce inhibition
milder_params = base_params.copy()
milder_params["T_pretreat"] = 165.0  # <-- Milder Temp
milder_params["E_load_ssf"] = 25.0  # <-- Add a bit more enzyme

example_details_mild = bo_objective_function(**milder_params)
print(f"MILDER RUN (MESP=${example_details_mild.get('MESP'):.3f}/L)")
print(
    f"  -> Furfural: {example_details_mild.get('furfural_conc_ssf'):.2f} g/L (Inhibition Ki_growth = {biorefinery_simulator.SSF_Ki_fur_growth} g/L)"
)
print(f"  -> C_E_final: {example_details_mild.get('C_E_final_g_L'):.2f} g/L")


print("\n--- Testing Model with 'ROBUST' Setpoints ---")
# Let's try the original harsh pretreatment, but with a high yeast inoculum
robust_params = base_params.copy()
robust_params["SSF_X_0"] = 5.0  # <-- 5x more yeast

example_details_robust = bo_objective_function(**robust_params)
print(f"ROBUST RUN (MESP=${example_details_robust.get('MESP'):.3f}/L)")
print(
    f"  -> Furfural: {example_details_robust.get('furfural_conc_ssf'):.2f} g/L (Inhibition Ki_growth = {biorefinery_simulator.SSF_Ki_fur_growth} g/L)"
)
print(f"  -> C_E_final: {example_details_robust.get('C_E_final_g_L'):.2f} g/L")
