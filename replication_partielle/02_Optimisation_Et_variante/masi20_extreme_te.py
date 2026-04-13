"""
=============================================================================
MASI 20 – Réplication Extrême : Objectif TE < 0.20% (k=6, 7, 8)
=============================================================================
Technique : "Exhaustive Deep Search"
Puisque l'univers est petit (20 actions), on peut tester toutes les 
combinaisons C(20, k) possibles à chaque rebalancing.

1. Proxy Fast OLS : Permet de tester ~100k combos en quelques secondes.
2. Deep Optimization : On prend le Top 5 des combos OLS et on applique
   `min_te_shrinkage` (Ledoit-Wolf) pour obtenir les poids finaux.

Configuration : Walk-Forward strict, Rebal = 5 jours, Lookback = 126 jours.
=============================================================================
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

warnings.filterwarnings("ignore")
np.random.seed(42)

# ============================================================================
# CONFIG
# ============================================================================
BASE_PATH = r"c:\Users\PC\Downloads\Stage\répliaction_partielle"

FILES = {
    "2020-2021": "20-21.xlsx",
    "2021-2022": "21-22.xlsx",
    "2022-2023": "22-23.xlsx",
    "2023-2024": "23-24.xlsx",
    "2024-2025": "24-25.xlsx",
    "2025-2026": "25-26-SANSANOMALIE.xlsx",
}
FLOTTANT_FILE = "Flottant_plafonnements_masi20.xlsx"

K_VALUES = [8, 7, 6]
REBAL_DAYS = [5]
LOOKBACK = 126

OUTPUT_FILE = os.path.join(BASE_PATH, "resultats_extreme_te.xlsx")


# ============================================================================
# MODULE 1 : CHARGEMENT
# ============================================================================
def load_all_data():
    print("=" * 60)
    print("CHARGEMENT DES DONNÉES")
    print("=" * 60)

    yearly_data = {}
    for year_label, filename in FILES.items():
        path = os.path.join(BASE_PATH, filename)
        df = pd.read_excel(path, header=10)
        df = df.dropna(axis=1, how="all")
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        index_col = [c for c in df.columns if "MSEMSI20" in str(c)]
        index_col = index_col[0] if index_col else df.columns[1]
        stock_cols = [c for c in df.columns if c != "Date" and c != index_col]

        for col in [index_col] + stock_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        rets_df = df[[index_col] + stock_cols]
        rets = np.log(rets_df / rets_df.shift(1)).iloc[1:]
        rets = rets.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-0.10, 0.10)

        mcap_matrix = df[stock_cols].values

        yearly_data[year_label] = {
            "r_index": rets[index_col].values,
            "r_stocks": rets[stock_cols].values,
            "stock_cols": stock_cols,
            "mcap_matrix": mcap_matrix[1:],
        }
    
    df_flottant = pd.read_excel(os.path.join(BASE_PATH, FLOTTANT_FILE))
    return yearly_data, df_flottant


# ============================================================================
# MODULE 2 : OPTIMISATION DEEP (Ledoit-Wolf Shrinkage)
# ============================================================================
def optim_min_te_shrinkage(r_index, r_sub):
    """
    Optimisation lourde TE avec matrice de covariance Shrinkage (Ledoit-Wolf).
    Résout analytiquement un QP sous contrainte w>=0, sum(w)=1.
    """
    k = r_sub.shape[1]
    w0 = np.ones(k) / k
    T = r_sub.shape[0]

    if T < k + 2:
        return w0

    try:
        data = np.column_stack([r_sub, r_index])
        lw = LedoitWolf().fit(data)
        cov_shrunk = lw.covariance_

        Sigma_ss = cov_shrunk[:k, :k]
        sigma_si = cov_shrunk[:k, k]

        def obj(w): return w @ Sigma_ss @ w - 2 * w @ sigma_si
        def grad(w): return 2 * Sigma_ss @ w - 2 * sigma_si

        bounds = [(0, 1)] * k
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        res = minimize(obj, w0, jac=grad, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12})
        if res.success:
            w = np.maximum(res.x, 0)
            return w / w.sum() if w.sum() > 0 else w0
    except Exception:
        pass

    # Fallback standard TE si SLSQP échoue
    def obj_fallback(w): return np.std(r_sub @ w - r_index)
    res = minimize(obj_fallback, w0, method="SLSQP", bounds=bounds, constraints=cons)
    if res.success:
        w = np.maximum(res.x, 0)
        return w / w.sum() if w.sum() > 0 else w0
    return w0


# ============================================================================
# MAIN ENGINE : EXHAUSTIVE DEEP SEARCH
# ============================================================================
def run_extreme_te(yearly_data, df_flottant):
    print("\n" + "=" * 60)
    print("MOTEUR WALK-FORWARD — EXHAUSTIVE DEEP SEARCH")
    print("=" * 60)

    # Pré-calcul du nombre de combinaisons
    import math
    n_stocks = 20
    combos_counts = {k: math.comb(n_stocks, k) for k in K_VALUES}
    for k, c in combos_counts.items():
        print(f"  k={k} : {c:,} combinaisons possibles par rebalancing")

    all_results = []
    t0_global = time.time()

    for year_label, ydata in yearly_data.items():
        r_index = ydata["r_index"]
        r_stocks = ydata["r_stocks"]
        stock_cols = ydata["stock_cols"]
        
        T = len(r_index)
        print(f"\n  📅 {year_label}: {T} jours")

        for k in K_VALUES:
            for rebal in REBAL_DAYS:
                lb_size = LOOKBACK
                if lb_size >= T:
                    lb_size = T // 2

                all_diffs = []
                best_subsets_history = []
                t = lb_size

                # Boucle de Walk-Forward
                while t < T:
                    # 1. Fenêtre passée stricte
                    t0_w = max(0, t - lb_size)
                    r_idx_past = r_index[t0_w:t]
                    r_stk_past = r_stocks[t0_w:t]

                    # 2. BRUTE FORCE : Évaluation OLS ultra-rapide de toutes les combinaisons
                    # Proxy : w = inv(X'X)X'y (sans contrainte d'égalité stricte sum(w)=1 pour la vitesse)
                    # puis TE = std(Xw - y)
                    best_candidates = []
                    
                    # Pre-calculate X'y and X'X components for speed?
                    # We just run lstsq directly, numpy is heavily optimized in C.
                    idx_combinations = list(combinations(range(n_stocks), k))
                    
                    # Optimization trick: Instead of std for 100k subsets, use sum of squared residuals
                    # which is proportional to variance (and thus std).
                    # RSS = ||y - Xw||^2
                    
                    # On stocke les (TE_proxy, indices_combinaison)
                    results_proxy = []
                    for c in idx_combinations:
                        r_sub = r_stk_past[:, c]
                        try:
                            # OLS Fast
                            w, residuals, rank, s = np.linalg.lstsq(r_sub, r_idx_past, rcond=None)
                            # Clip to positive (long-only proxy)
                            w = np.maximum(w, 0)
                            s_w = w.sum()
                            if s_w > 0:
                                w = w / s_w
                            else:
                                w = np.ones(k) / k
                                
                            err = r_sub @ w - r_idx_past
                            rss = np.sum(err ** 2)
                            results_proxy.append((rss, c))
                        except Exception:
                            continue
                    
                    # Tri par erreur croissante, prendre le Top 5
                    results_proxy.sort(key=lambda x: x[0])
                    top5_combos = [res[1] for res in results_proxy[:5]]
                    
                    # 3. DEEP OPTIMIZATION sur le Top 5
                    # On applique l'optimisation rigoureuse Ledoit-Wolf sur les 5 finalistes
                    best_final_te = np.inf
                    best_final_c = top5_combos[0]
                    best_final_w = None
                    
                    for c in top5_combos:
                        r_sub = r_stk_past[:, c]
                        w = optim_min_te_shrinkage(r_idx_past, r_sub)
                        err = r_sub @ w - r_idx_past
                        te_exact = np.std(err)
                        
                        if te_exact < best_final_te:
                            best_final_te = te_exact
                            best_final_c = c
                            best_final_w = w
                            
                    best_subsets_history.append(best_final_c)

                    # 4. ÉVALUATION OOS du vainqueur
                    t_end = min(t + rebal, T)
                    r_idx_oos = r_index[t:t_end]
                    r_sub_oos = r_stocks[t:t_end, :][:, best_final_c]

                    if len(r_idx_oos) == 0:
                        break

                    r_port = r_sub_oos @ best_final_w
                    all_diffs.extend((r_port - r_idx_oos).tolist())
                    t += rebal

                # Calcul des métriques annuelles OOS
                if len(all_diffs) < 3:
                    te, corr, beta, cov_val = np.nan, np.nan, np.nan, np.nan
                else:
                    diffs = np.array(all_diffs)
                    te = np.std(diffs)
                    n = len(diffs)
                    r_idx_eval = r_index[lb_size:lb_size + n]
                    r_port_eval = r_idx_eval + diffs

                    if len(r_idx_eval) > 2:
                        cov_mat = np.cov(r_port_eval, r_idx_eval)
                        cov_val = cov_mat[0, 1]
                        var_idx = cov_mat[1, 1]
                        beta = cov_val / var_idx if var_idx > 0 else np.nan
                        corr = cov_val / (np.std(r_port_eval, ddof=1) * np.std(r_idx_eval, ddof=1))
                    else:
                        corr, beta, cov_val = np.nan, np.nan, np.nan

                # Statistiques sur la stabilité de la sélection
                selection_names = []
                for c in set(best_subsets_history):
                    count = best_subsets_history.count(c)
                    names = "+".join(stock_cols[i][:4] for i in c)
                    selection_names.append(f"{names} (x{count})")

                all_results.append({
                    "Année": year_label,
                    "k": k,
                    "Sélection": "exhaustive_deep",
                    "Optimisation": "min_te_shrinkage",
                    "Rebal_jours": rebal,
                    "Lookback": "rolling_126",
                    "TE": te,
                    "Corr": corr,
                    "Beta": beta,
                    "Stabilité": " | ".join(selection_names[:3]) + "..."
                })
                print(f"    ✅ k={k:<2d} | TE OOS: {te:.6f} ({te*100:.3f}%) | Corr: {corr:.3f}")

    elapsed = time.time() - t0_global
    print(f"  ⏱️  Total Extreme TE en {elapsed:.1f}s")
    return pd.DataFrame(all_results)

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("╔" + "═" * 55 + "╗")
    print("║  MASI 20 — EXHAUSTIVE DEEP SEARCH (k=6, 7, 8)        ║")
    print("╚" + "═" * 55 + "╝")

    yearly_data, df_flottant = load_all_data()

    print("\n✅ Données chargées. Lancement Exhaustive Search...")
    df_res = run_extreme_te(yearly_data, df_flottant)

    # Synthèse globale par k
    df_res["TE_pct"] = df_res["TE"] * 100
    grp = df_res.groupby("k")["TE_pct"].mean().sort_index(ascending=False)
    
    print("\n" + "=" * 60)
    print("SYNTHÈSE EXTREME TE (Moyenne 6 ans)")
    print("=" * 60)
    for k, mean_te in grp.items():
        print(f"  k={int(k)} actions : TE Moyen = {mean_te:.3f}%")
        
    df_res.to_excel(OUTPUT_FILE, index=False)
    print(f"\n✅ Résultats complets exportés dans : {OUTPUT_FILE}")
