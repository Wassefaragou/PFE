"""
=============================================================================
MASI 20 – Réplication Avancée : Objectif TE < 0.20%
=============================================================================
Techniques avancées pour réduire le Tracking Error :

  1. Shrinkage Ledoit-Wolf (covariance robuste)
  2. EWMA (pondération exponentielle)
  3. Rebalancing haute fréquence (1-3 jours)
  4. Sélection ensemble (vote de 4 méthodes)
  5. Optimisation TE régularisée
  6. Fenêtre glissante optimale (126j)

Approche UNIQUE : Walk-Forward strict (la plus rigoureuse)
=============================================================================
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Lasso, ElasticNet, Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
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

K_VALUES = [9, 10]  # Seuls k viables pour TE < 0.20%
REBAL_DAYS = [1, 2, 3, 5]  # Plus fréquent
LOOKBACK_MODES = ["rolling_126", "rolling_63", "expanding"]

SELECTION_METHODS = [
    "elastic_net", "lasso", "greedy_te", "top_mcap", "ensemble_vote",
]
OPTIM_METHODS = [
    "min_te",
    "min_te_shrinkage",
    "min_te_ewma",
    "min_te_regularized",
    "min_te_shrinkage_ewma",
]

OUTPUT_FILE = os.path.join(BASE_PATH, "resultats_advanced_te.xlsx")


# ============================================================================
# MODULE 1 : CHARGEMENT (identique à v2)
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
        df = df[[c for c in df.columns if not str(c).startswith("Unnamed")]]
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        index_col = [c for c in df.columns if "MSEMSI20" in str(c)]
        index_col = index_col[0] if index_col else df.columns[1]
        stock_cols = [c for c in df.columns if c != "Date" and c != index_col]

        for col in [index_col] + stock_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Rendements logarithmiques
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
        print(f"  ✅ {year_label}: {len(rets)} jours, {len(stock_cols)} actions")

    df_flottant = pd.read_excel(os.path.join(BASE_PATH, FLOTTANT_FILE))
    print(f"  ✅ Table flottant: {len(df_flottant)} lignes")
    return yearly_data, df_flottant


# ============================================================================
# MODULE 2 : SÉLECTION (5 méthodes dont ensemble)
# ============================================================================
def select_top_mcap(r_index, r_stocks, mean_mcap, k):
    return np.argsort(mean_mcap)[-k:][::-1].tolist()

def select_top_corr(r_index, r_stocks, mean_mcap, k):
    corrs = np.array([np.corrcoef(r_index, r_stocks[:, i])[0, 1]
                      for i in range(r_stocks.shape[1])])
    corrs = np.nan_to_num(corrs, nan=-1)
    return np.argsort(corrs)[-k:][::-1].tolist()

def select_greedy_te(r_index, r_stocks, mean_mcap, k):
    n = r_stocks.shape[1]
    selected = []
    for _ in range(k):
        best_te, best_j = np.inf, 0
        for j in range(n):
            if j in selected:
                continue
            cand = selected + [j]
            r_sub = r_stocks[:, cand]
            try:
                w = np.linalg.lstsq(r_sub, r_index, rcond=None)[0]
                w = np.maximum(w, 0)
                s = w.sum()
                w = w / s if s > 0 else np.ones(len(cand)) / len(cand)
            except Exception:
                w = np.ones(len(cand)) / len(cand)
            te = np.std(r_sub @ w - r_index)
            if te < best_te:
                best_te, best_j = te, j
        selected.append(best_j)
    return selected

def select_lasso(r_index, r_stocks, mean_mcap, k):
    best = list(range(k))
    for alpha in np.logspace(-6, -1, 15):
        m = Lasso(alpha=alpha, max_iter=2000, positive=True)
        m.fit(r_stocks, r_index)
        nz = np.where(m.coef_ > 1e-8)[0]
        if len(nz) >= k:
            return np.argsort(m.coef_)[-k:][::-1].tolist()
        elif len(nz) > 0:
            best = nz.tolist()
    if len(best) < k:
        corrs = np.nan_to_num(np.array([np.corrcoef(r_index, r_stocks[:, i])[0, 1]
                                         for i in range(r_stocks.shape[1])]), nan=-1)
        for i in np.argsort(corrs)[::-1]:
            if i not in best:
                best.append(i)
            if len(best) == k:
                break
    return best[:k]

def select_elastic_net(r_index, r_stocks, mean_mcap, k):
    best = list(range(k))
    for alpha in np.logspace(-6, -1, 15):
        m = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=2000, positive=True)
        m.fit(r_stocks, r_index)
        nz = np.where(m.coef_ > 1e-8)[0]
        if len(nz) >= k:
            return np.argsort(m.coef_)[-k:][::-1].tolist()
        elif len(nz) > 0:
            best = nz.tolist()
    if len(best) < k:
        corrs = np.nan_to_num(np.array([np.corrcoef(r_index, r_stocks[:, i])[0, 1]
                                         for i in range(r_stocks.shape[1])]), nan=-1)
        for i in np.argsort(corrs)[::-1]:
            if i not in best:
                best.append(i)
            if len(best) == k:
                break
    return best[:k]

def select_ensemble_vote(r_index, r_stocks, mean_mcap, k):
    """
    Vote majoritaire : chaque méthode vote pour ses k titres.
    On prend les k titres ayant reçu le plus de votes.
    En cas d'égalité, on préfère les actions avec la plus forte corrélation.
    """
    n = r_stocks.shape[1]
    votes = np.zeros(n)

    # 4 méthodes votantes (les meilleures de v2)
    methods = [select_elastic_net, select_lasso, select_greedy_te, select_top_mcap]
    for method in methods:
        try:
            selected = method(r_index, r_stocks, mean_mcap, k)
            for idx in selected:
                votes[idx] += 1
        except Exception:
            pass

    # Départager par corrélation
    corrs = np.array([np.corrcoef(r_index, r_stocks[:, i])[0, 1]
                      for i in range(n)])
    corrs = np.nan_to_num(corrs, nan=-1)

    # Score composite : votes * 10 + corrélation (votes dominent)
    score = votes * 10 + corrs
    return np.argsort(score)[-k:][::-1].tolist()


SEL_FUNCS = {
    "elastic_net": select_elastic_net,
    "lasso": select_lasso,
    "greedy_te": select_greedy_te,
    "top_mcap": select_top_mcap,
    "ensemble_vote": select_ensemble_vote,
}

def run_selection(method, r_index, r_stocks, mean_mcap, k):
    try:
        idx = SEL_FUNCS[method](r_index, r_stocks, mean_mcap, k)
        idx = [i for i in idx if 0 <= i < r_stocks.shape[1]][:k]
        if len(idx) < 2:
            idx = select_top_corr(r_index, r_stocks, mean_mcap, k)
    except Exception:
        idx = select_top_corr(r_index, r_stocks, mean_mcap, k)
    return idx


# ============================================================================
# MODULE 3 : OPTIMISATION AVANCÉE (5 méthodes)
# ============================================================================

def optim_min_te(r_index, r_sub, caps=None, mcaps=None):
    """Baseline : minimisation du TE par SLSQP."""
    k = r_sub.shape[1]
    w0 = np.ones(k) / k
    def obj(w): return np.std(r_sub @ w - r_index)
    bounds = [(0, 1)] * k
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    try:
        res = minimize(obj, w0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12})
        if res.success:
            w = np.maximum(res.x, 0)
            return w / w.sum() if w.sum() > 0 else w0
    except Exception:
        pass
    return w0


def optim_min_te_shrinkage(r_index, r_sub, caps=None, mcaps=None):
    """
    Optimisation TE avec matrice de covariance Shrinkage (Ledoit-Wolf).
    Résout analytiquement : w* = Σ^{-1} Σ_{si} / (1' Σ^{-1} Σ_{si})
    où Σ est la covariance shrinkée et Σ_{si} les covariances stock-index.
    """
    k = r_sub.shape[1]
    w0 = np.ones(k) / k
    T = r_sub.shape[0]

    if T < k + 2:
        return w0

    try:
        # Construire matrice augmentée [stocks | index]
        data = np.column_stack([r_sub, r_index])
        lw = LedoitWolf().fit(data)
        cov_shrunk = lw.covariance_

        # Extraire sous-matrices
        Sigma_ss = cov_shrunk[:k, :k]  # Cov stocks-stocks
        sigma_si = cov_shrunk[:k, k]   # Cov stocks-index

        # Poids optimaux via QP : min w'Σ_ss w - 2w'σ_si
        # sous contraintes w>=0, sum(w)=1
        def obj(w):
            return w @ Sigma_ss @ w - 2 * w @ sigma_si

        def grad(w):
            return 2 * Sigma_ss @ w - 2 * sigma_si

        bounds = [(0, 1)] * k
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        res = minimize(obj, w0, jac=grad, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12})
        if res.success:
            w = np.maximum(res.x, 0)
            return w / w.sum() if w.sum() > 0 else w0
    except Exception:
        pass

    return optim_min_te(r_index, r_sub, caps, mcaps)


def optim_min_te_ewma(r_index, r_sub, caps=None, mcaps=None):
    """
    Optimisation TE avec pondération exponentielle (EWMA).
    Les observations récentes ont plus de poids.
    Halflife = 21 jours (~1 mois de trading).
    """
    k = r_sub.shape[1]
    w0 = np.ones(k) / k
    T = r_sub.shape[0]

    if T < 5:
        return w0

    try:
        # Calcul des poids EWMA
        halflife = 21
        lam = np.exp(-np.log(2) / halflife)
        ewma_weights = np.array([lam ** (T - 1 - t) for t in range(T)])
        ewma_weights /= ewma_weights.sum()
        sqrt_w = np.sqrt(ewma_weights)

        # Pondérer les données
        r_sub_w = r_sub * sqrt_w[:, None]
        r_idx_w = r_index * sqrt_w

        def obj(w):
            diffs = r_sub_w @ w - r_idx_w
            return np.sqrt(np.sum(diffs ** 2))

        bounds = [(0, 1)] * k
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        res = minimize(obj, w0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12})
        if res.success:
            w = np.maximum(res.x, 0)
            return w / w.sum() if w.sum() > 0 else w0
    except Exception:
        pass

    return optim_min_te(r_index, r_sub, caps, mcaps)


def optim_min_te_regularized(r_index, r_sub, caps=None, mcaps=None):
    """
    Min TE + régularisation Ridge vers poids égaux.
    Objectif : min ||r_sub @ w - r_index||² + λ * ||w - w_eq||²
    λ choisi pour stabiliser les poids (0.01).
    """
    k = r_sub.shape[1]
    w0 = np.ones(k) / k
    w_eq = np.ones(k) / k
    lam = 0.01  # Force de régularisation

    try:
        def obj(w):
            te_sq = np.mean((r_sub @ w - r_index) ** 2)
            reg = lam * np.sum((w - w_eq) ** 2)
            return te_sq + reg

        bounds = [(0, 1)] * k
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        res = minimize(obj, w0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12})
        if res.success:
            w = np.maximum(res.x, 0)
            return w / w.sum() if w.sum() > 0 else w0
    except Exception:
        pass

    return optim_min_te(r_index, r_sub, caps, mcaps)


def optim_min_te_shrinkage_ewma(r_index, r_sub, caps=None, mcaps=None):
    """
    Combinaison : Covariance Ledoit-Wolf + pondération EWMA.
    La meilleure technique attendue.
    """
    k = r_sub.shape[1]
    w0 = np.ones(k) / k
    T = r_sub.shape[0]

    if T < k + 2:
        return w0

    try:
        # 1. Poids EWMA
        halflife = 21
        lam = np.exp(-np.log(2) / halflife)
        ewma_weights = np.array([lam ** (T - 1 - t) for t in range(T)])
        ewma_weights /= ewma_weights.sum()
        sqrt_w = np.sqrt(ewma_weights)

        # 2. Données pondérées
        r_sub_w = r_sub * sqrt_w[:, None]
        r_idx_w = r_index * sqrt_w

        # 3. Covariance Shrinkage sur données pondérées
        data_w = np.column_stack([r_sub_w, r_idx_w])
        lw = LedoitWolf().fit(data_w)
        cov_shrunk = lw.covariance_

        Sigma_ss = cov_shrunk[:k, :k]
        sigma_si = cov_shrunk[:k, k]

        # 4. QP avec gradient analytique
        def obj(w):
            return w @ Sigma_ss @ w - 2 * w @ sigma_si

        def grad(w):
            return 2 * Sigma_ss @ w - 2 * sigma_si

        bounds = [(0, 1)] * k
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        res = minimize(obj, w0, jac=grad, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12})
        if res.success:
            w = np.maximum(res.x, 0)
            return w / w.sum() if w.sum() > 0 else w0
    except Exception:
        pass

    return optim_min_te_shrinkage(r_index, r_sub, caps, mcaps)


OPT_FUNCS = {
    "min_te": optim_min_te,
    "min_te_shrinkage": optim_min_te_shrinkage,
    "min_te_ewma": optim_min_te_ewma,
    "min_te_regularized": optim_min_te_regularized,
    "min_te_shrinkage_ewma": optim_min_te_shrinkage_ewma,
}


# ============================================================================
# WALK-FORWARD AVANCÉ
# ============================================================================
def run_walk_forward_advanced(yearly_data, df_flottant):
    """
    Walk-forward avec lookback modes :
      - rolling_126 : fenêtre glissante 126j (~6 mois)
      - rolling_63  : fenêtre glissante 63j (~3 mois)
      - expanding   : toutes les données passées
    """
    print("\n" + "=" * 60)
    print("WALK-FORWARD AVANCÉ — Objectif TE < 0.20%")
    print("=" * 60)

    total = len(FILES) * len(K_VALUES) * len(SELECTION_METHODS) * \
            len(OPTIM_METHODS) * len(REBAL_DAYS) * len(LOOKBACK_MODES)
    print(f"  Combinaisons totales: {total}")

    all_results = []
    count = 0
    t0 = time.time()

    for year_label, ydata in yearly_data.items():
        r_index = ydata["r_index"]
        r_stocks = ydata["r_stocks"]
        stock_cols = ydata["stock_cols"]
        mcap_matrix = ydata["mcap_matrix"]

        T = len(r_index)

        # Flottant / plafonnement
        df_fl = df_flottant[df_flottant["Année"] == year_label]
        flot_dict = dict(zip(df_fl["Valeur"], df_fl["Flottant"]))
        plaf_dict = dict(zip(df_fl["Valeur"], df_fl["Plafonnement"]))

        print(f"\n  📅 {year_label}: {T} jours")

        for lb_mode in LOOKBACK_MODES:
            for k in K_VALUES:
                for sel in SELECTION_METHODS:
                    for opt in OPTIM_METHODS:
                        for rebal in REBAL_DAYS:
                            count += 1

                            # Déterminer le lookback initial
                            if lb_mode == "rolling_126":
                                lb_size = 126
                            elif lb_mode == "rolling_63":
                                lb_size = 63
                            else:
                                lb_size = 60  # minimum pour expanding

                            if lb_size >= T:
                                lb_size = T // 2

                            all_diffs = []
                            t = lb_size

                            while t < T:
                                # Fenêtre passée
                                if lb_mode == "expanding":
                                    t0_w = 0
                                else:
                                    t0_w = max(0, t - lb_size)

                                r_idx_past = r_index[t0_w:t]
                                r_stk_past = r_stocks[t0_w:t]
                                mcap_past = mcap_matrix[t0_w:t]
                                mean_mcap_past = mcap_past.mean(axis=0)

                                # 1. SÉLECTION
                                selected_idx = run_selection(
                                    sel, r_idx_past, r_stk_past,
                                    mean_mcap_past, k
                                )

                                sel_names = [stock_cols[i] for i in selected_idx]
                                caps = np.array([plaf_dict.get(s, 1.0) for s in sel_names])
                                flotts = np.array([flot_dict.get(s, 1.0) for s in sel_names])
                                mcaps_sel = mean_mcap_past[selected_idx]
                                mcaps_float = mcaps_sel * flotts

                                # 2. OPTIMISATION
                                r_sub_past = r_stk_past[:, selected_idx]
                                w = OPT_FUNCS[opt](
                                    r_idx_past, r_sub_past,
                                    caps=caps, mcaps=mcaps_float
                                )

                                # 3. ÉVALUATION OOS
                                t_end = min(t + rebal, T)
                                r_idx_oos = r_index[t:t_end]
                                r_sub_oos = r_stocks[t:t_end, :][:, selected_idx]

                                if len(r_idx_oos) == 0:
                                    break

                                r_port = r_sub_oos @ w
                                all_diffs.extend((r_port - r_idx_oos).tolist())
                                t += rebal

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

                            all_results.append({
                                "Année": year_label,
                                "k": k,
                                "Sélection": sel,
                                "Optimisation": opt,
                                "Rebal_jours": rebal,
                                "Lookback": lb_mode,
                                "TE": te,
                                "Corr": corr,
                                "Beta": beta,
                                "Cov": cov_val,
                                "Actions": ", ".join(sel_names) if len(all_diffs) >= 3 else "",
                            })

                            if count % 500 == 0:
                                e = time.time() - t0
                                pct = count / total * 100
                                print(f"    [{pct:5.1f}%] {count}/{total} — {e:.0f}s")

        print(f"    ✅ {year_label} terminé")

    elapsed = time.time() - t0
    print(f"  ⏱️  Total: {count} combos en {elapsed:.1f}s")
    return pd.DataFrame(all_results)


# ============================================================================
# RAPPORT & SYNTHÈSE
# ============================================================================
def generate_report(df_all):
    print("\n" + "=" * 60)
    print("RAPPORT AVANCÉ")
    print("=" * 60)

    grp = ["k", "Sélection", "Optimisation", "Rebal_jours", "Lookback"]
    df_agg = df_all.groupby(grp).agg(
        TE_moyen=("TE", "mean"),
        TE_median=("TE", "median"),
        TE_max=("TE", "max"),
        TE_std=("TE", "std"),
        Corr_moy=("Corr", "mean"),
        Beta_moy=("Beta", "mean"),
    ).reset_index().sort_values("TE_moyen")

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as w:
        df_all.to_excel(w, sheet_name="Brut", index=False)
        df_agg.to_excel(w, sheet_name="Agrégé", index=False)
        df_agg.head(30).to_excel(w, sheet_name="Top30", index=False)

        # Par dimension
        for dim in ["Sélection", "Optimisation", "Rebal_jours", "Lookback", "k"]:
            agg = df_all.groupby(dim)["TE"].agg(["mean", "median", "min", "max"]).sort_values("mean")
            agg.to_excel(w, sheet_name=f"Par_{dim[:8]}")

        # Heatmap Sel × Opt
        hm = df_all.groupby(["Sélection", "Optimisation"])["TE"].mean().reset_index()
        hm.pivot(index="Sélection", columns="Optimisation", values="TE").to_excel(w, sheet_name="HM_SelOpt")

        # Heatmap Lookback × Rebal
        hm2 = df_all.groupby(["Lookback", "Rebal_jours"])["TE"].mean().reset_index()
        hm2.pivot(index="Lookback", columns="Rebal_jours", values="TE").to_excel(w, sheet_name="HM_LbRebal")

        # Par année (check annuel)
        year_check = df_all.groupby("Année")["TE"].agg(["mean", "median", "min", "max"]).sort_values("mean")
        year_check.to_excel(w, sheet_name="ParAnnée")

        # Combos avec TE < 0.002 par année
        df_agg_year = df_all.groupby(grp + ["Année"]).agg(TE=("TE", "mean")).reset_index()
        sub_002 = df_agg[df_agg["TE_moyen"] < 0.002]
        sub_002.to_excel(w, sheet_name="Sub02pct", index=False)

    print(f"  📄 Rapport: {OUTPUT_FILE}")
    return df_agg


def print_synthesis(df_all, df_agg):
    print("\n" + "=" * 60)
    print("SYNTHÈSE — OBJECTIF TE < 0.20%")
    print("=" * 60)

    n_sub = len(df_agg[df_agg["TE_moyen"] < 0.002])
    n_total = len(df_agg)
    print(f"\n  🎯 {n_sub}/{n_total} combinaisons avec TE moyen < 0.20%")

    if n_sub > 0:
        print(f"\n  🏆 TOP 10 MEILLEURES COMBINAISONS:")
        for i, (_, r) in enumerate(df_agg.head(10).iterrows()):
            print(f"     {i+1}. k={int(r['k'])}, sel={r['Sélection']:<16s}, "
                  f"opt={r['Optimisation']:<22s}, rebal={int(r['Rebal_jours']):2d}j, "
                  f"lb={r['Lookback']:<12s} → TE={r['TE_moyen']:.6f}")

    # Vérification par année
    print(f"\n  📅 TE moyen PAR ANNÉE (top 10 configs):")
    top_configs = df_agg.head(10)
    grp_cols = ["k", "Sélection", "Optimisation", "Rebal_jours", "Lookback"]
    for _, cfg in top_configs.head(3).iterrows():
        mask = True
        for col in grp_cols:
            mask = mask & (df_all[col] == cfg[col])
        sub = df_all[mask]
        print(f"\n     Config: k={int(cfg['k'])}, {cfg['Sélection']}, "
              f"{cfg['Optimisation']}, rebal={int(cfg['Rebal_jours'])}j, {cfg['Lookback']}")
        for _, row in sub.iterrows():
            status = "✅" if row["TE"] < 0.002 else ("⚠️" if row["TE"] < 0.003 else "❌")
            print(f"       {row['Année']}: TE={row['TE']:.6f} {status}")

    # Par dimension
    print(f"\n  📌 TE moyen par SÉLECTION:")
    for s, te in df_all.groupby("Sélection")["TE"].mean().sort_values().items():
        print(f"     {s:<20s} → TE = {te:.6f}")

    print(f"\n  📌 TE moyen par OPTIMISATION:")
    for o, te in df_all.groupby("Optimisation")["TE"].mean().sort_values().items():
        print(f"     {o:<22s} → TE = {te:.6f}")

    print(f"\n  📌 TE moyen par LOOKBACK:")
    for lb, te in df_all.groupby("Lookback")["TE"].mean().sort_values().items():
        print(f"     {lb:<15s} → TE = {te:.6f}")

    print(f"\n  📌 TE moyen par REBAL:")
    for f, te in df_all.groupby("Rebal_jours")["TE"].mean().sort_values().items():
        print(f"     {int(f):3d} jours → TE = {te:.6f}")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("╔" + "═" * 55 + "╗")
    print("║  MASI 20 — Réplication Avancée (TE < 0.20%)          ║")
    print("╚" + "═" * 55 + "╝")

    yearly_data, df_flottant = load_all_data()

    print("\n✅ Données chargées. Lancement Walk-Forward avancé...")
    df_wf = run_walk_forward_advanced(yearly_data, df_flottant)

    df_agg = generate_report(df_wf)
    print_synthesis(df_wf, df_agg)

    print("\n✅ TERMINÉ — Aucun in-sample dans l'évaluation")
