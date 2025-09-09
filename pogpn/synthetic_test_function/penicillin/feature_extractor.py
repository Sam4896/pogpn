import torch
from typing import Dict, List, Literal, Optional, Tuple
import warnings
import torch
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.stats import linregress


@torch.no_grad()
def extract_simple_features_scipy(
    sim_out: Dict[str, torch.Tensor],
    step_size: float,
    strategy: Literal["mixed", "exp_all", "linear_all"] = "mixed",
    ignore_peak_first_n: int = 0,
    exp_bounds: Optional[Dict[str, Tuple[List[float], List[float]]]] = None,
    exp_p0: Optional[Dict[str, List[float]]] = None,
    exp_model_types: Optional[Dict[str, Literal["convex", "concave"]]] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    import numpy as np

    # Suppress OptimizeWarning, which can be noisy if fits aren't perfect
    warnings.simplefilter("ignore", OptimizeWarning)

    Y = sim_out["state_space"]
    t_final = sim_out["t_final"]
    B, T, _ = Y.shape
    device, dtype = Y.device, Y.dtype
    dt = float(step_size)

    steps = (t_final / dt).round().to(torch.long).clamp(min=2, max=T)

    def want_exp(name: str) -> bool:
        if strategy == "exp_all":
            return True
        if strategy == "linear_all":
            return False
        return name in ("P", "V")

    def fit_linear_np(t: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
        slope, intercept, _, _, _ = linregress(t, y)
        yhat = slope * t + intercept
        rmse = np.sqrt(np.mean((yhat - y) ** 2))
        return float(slope), float(intercept), float(rmse)  # type: ignore

    def fit_exp_np(
        t: np.ndarray,
        y: np.ndarray,
        model_type: Literal["convex", "concave"],
        bounds: Tuple[List[float], List[float]],
        p0_user: Optional[List[float]] = None,
    ) -> Tuple[float, float, float, float]:
        if model_type == "concave":
            # Model: a - b * exp(-c * t) for increasing, concave growth
            def model(tt, a, b, c):
                return a - b * np.exp(-c * tt)

            p0 = p0_user
            if not p0:
                a0 = float(np.max(y))
                b0 = a0 - y[0]
                y_shift = np.maximum(a0 - y, 1e-8)
                try:
                    slope, intercept, _, _, _ = linregress(t, np.log(y_shift))
                    c0 = -float(slope)  # type: ignore
                    p0 = [a0, b0 if b0 > 0 else float(np.exp(intercept)), c0]  # type: ignore
                except ValueError:
                    p0 = [a0, b0, 0.01]
        else:  # convex
            # Model: a + b * exp(c * t) for convex growth/decay
            def model(tt, a, b, c):
                return a + b * np.exp(c * tt)

            p0 = p0_user
            if not p0:
                is_decay = y[-1] < y[0]
                if is_decay:  # Convex decay (e.g., Volume)
                    a0 = float(y[-1])  # Asymptote
                    b0 = float(y[0] - a0)
                    y_shift = np.maximum(y - a0, 1e-8)
                    try:
                        slope, _, _, _, _ = linregress(t, np.log(y_shift))
                        c0 = float(slope)  # type: ignore
                        p0 = [a0, b0, c0]
                    except ValueError:
                        p0 = [a0, b0, -0.01]
                else:  # Convex growth (e.g., Penicillin)
                    min_y = np.min(y)
                    y_shift = np.maximum(y - min_y, 1e-8)
                    try:
                        if np.any(y_shift > 1e-7):
                            slope, intercept, _, _, _ = linregress(t, np.log(y_shift))
                            p0 = [min_y, float(np.exp(intercept)), float(slope)]  # type: ignore
                        else:
                            p0 = [y[0], 0, 0]
                    except ValueError:
                        p0 = [y[0], (y[-1] - y[0]) / 2, 0.01]

        try:
            popt, _ = curve_fit(model, t, y, p0=p0, bounds=bounds, maxfev=20000)
            a, b, c = [float(v) for v in popt]
        except Exception:
            m, q, rmse_lin = fit_linear_np(t, y)
            return q, m, 0.0, rmse_lin

        yhat = model(t, a, b, c)
        rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
        return a, b, c, rmse

    out_np: Dict[str, Dict[str, list]] = {
        "P": {"a": [], "b": [], "c": [], "rmse": []},
        "V": {"a": [], "b": [], "c": [], "rmse": []},
        "X": {"a": [], "b": [], "c": [], "rmse": []},
        "CO2": {"a": [], "b": [], "c": [], "rmse": []},
        "S": {"peak": [], "t_peak": []},
    }

    Yn = Y.detach().cpu().numpy()
    steps_n = steps.cpu().numpy()
    dt64 = np.float64(dt)

    for i in range(B):
        n = int(steps_n[i])
        t = np.arange(n, dtype=np.float64) * dt64

        series = {
            "P": Yn[i, :n, 0],
            "V": Yn[i, :n, 1],
            "X": Yn[i, :n, 2],
            "CO2": Yn[i, :n, 4],
        }
        for name, y in series.items():
            if want_exp(name):
                model_type = (exp_model_types or {}).get(name, "convex")

                default_bounds_table = {
                    "P": ([-np.inf, 0, 1e-5], [np.max(y), np.inf, 1.0]),
                    "V": ([np.min(y) * 0.8, 0, -5.0], [np.max(y), np.inf, -1e-5]),
                    "X": ([np.min(y), 0, 1e-5], [np.max(y) * 1.5, np.inf, 1.0]),
                    "CO2": ([np.min(y), 0, 1e-5], [np.max(y) * 1.5, np.inf, 1.0]),
                }
                current_bounds = default_bounds_table.get(
                    name, ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
                )

                if exp_bounds and name in exp_bounds:
                    current_bounds = exp_bounds[name]

                p0_for_fit = (exp_p0 or {}).get(name)

                a, b, c, rmse = fit_exp_np(
                    t,
                    y,
                    model_type=model_type,
                    bounds=current_bounds,
                    p0_user=p0_for_fit,
                )
                out_np[name]["a"].append(a)
                out_np[name]["b"].append(b)
                out_np[name]["c"].append(c)
                out_np[name]["rmse"].append(rmse)
            else:
                slope, intercept, rmse = fit_linear_np(t, y)
                out_np[name].setdefault("slope", []).append(slope)
                out_np[name].setdefault("intercept", []).append(intercept)
                out_np[name]["rmse"].append(rmse)

        S = Yn[i, :n, 3]
        start = min(ignore_peak_first_n, max(n - 1, 0))
        if start >= S.shape[0]:
            start = 0
        j_rel = int(np.argmax(S[start:]))
        j = start + j_rel
        out_np["S"]["peak"].append(float(S[j]))
        out_np["S"]["t_peak"].append(float(j * dt64))

    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for state, d in out_np.items():
        packed: Dict[str, torch.Tensor] = {}
        for k, vals in d.items():
            if not vals:
                continue
            arr = np.asarray(vals, dtype=np.float64)
            packed[k] = torch.tensor(arr, device=device, dtype=dtype)
        out[state] = packed
    return out
