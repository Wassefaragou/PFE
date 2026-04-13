"""
=============================================================================
MASI 20 – Réplication Partielle Exhaustive (Optimisé)
=============================================================================
Grid Search: 8 sélections × 5 optims × 6 freq × 9 k-values = 2160/an
Architecture optimisée: sélection précomputée sur données complètes,
seule l'optimisation des poids est refaite à chaque rééquilibrage.

Évaluation: Tracking Error NON annualisé = std(r_portfolio - r_index)
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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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

K_VALUES = list(range(2, 11))           # 2 à 10 titres
REBAL_DAYS = [5, 10, 15, 21, 42, 63]   # ~1 sem à ~3 mois

SELECTION_METHODS = [
    "top_mcap", "top_corr", "greedy_te", "lasso",
    "elastic_net", "pca", "random_forest", "clustering",
]
OPTIM_METHODS = [
    "min_te", "min_te_constrained", "ols", "ridge", "prop_mcap",
]

OUTPUT_FILE = os.path.join(BASE_PATH, "resultats_replication.xlsx")


# ============================================================================
# MODULE 1 : CHARGEMENT & PRÉPARATION
# ============================================================================
def load_all_data():
    """Charge toutes les données annuelles et la table flottant."""
    print("=" * 60)
    print("MODULE 1 : Chargement des données")
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
        if not index_col:
            index_col = [df.columns[1]]
        index_col = index_col[0]
        stock_cols = [c for c in df.columns if c != "Date" and c != index_col]

        # Convertir en numérique
        for col in [index_col] + stock_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Rendements simples
        returns = df[[index_col] + stock_cols].pct_change().iloc[1:]
        returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-0.5, 0.5)

        r_index = returns[index_col].values
        r_stocks = returns[stock_cols].values

        # MarketCap moyenne
        mean_mcap = df[stock_cols].mean().values

        yearly_data[year_label] = {
            "r_index": r_index,
            "r_stocks": r_stocks,
            "stock_cols": stock_cols,
            "mean_mcap": mean_mcap,
            "df": df,
        }
        print(f"  ✅ {year_label}: {len(r_index)} jours, {len(stock_cols)} actions")

    # Table flottant
    df_flottant = pd.read_excel(os.path.join(BASE_PATH, FLOTTANT_FILE))
    print(f"  ✅ Table flottant: {len(df_flottant)} lignes")

    return yearly_data, df_flottant


# ============================================================================
# MODULE 2 : MÉTHODES DE SÉLECTION (8)
# Chaque méthode retourne une liste d'indices (positions dans stock_cols)
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
            # Min-TE poids rapide via pseudo-inverse
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
    for alpha in np.logspace(-6, -1, 20):
        m = Lasso(alpha=alpha, max_iter=3000, positive=True)
        m.fit(r_stocks, r_index)
        nz = np.where(m.coef_ > 1e-8)[0]
        if len(nz) >= k:
            best = np.argsort(m.coef_)[-k:][::-1].tolist()
            break
        elif len(nz) > 0:
            best = nz.tolist()
    if len(best) < k:
        corrs = np.array([np.corrcoef(r_index, r_stocks[:, i])[0, 1]
                          for i in range(r_stocks.shape[1])])
        corrs = np.nan_to_num(corrs, nan=-1)
        for i in np.argsort(corrs)[::-1]:
            if i not in best:
                best.append(i)
            if len(best) == k:
                break
    return best[:k]


def select_elastic_net(r_index, r_stocks, mean_mcap, k):
    best = list(range(k))
    for alpha in np.logspace(-6, -1, 20):
        m = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=3000, positive=True)
        m.fit(r_stocks, r_index)
        nz = np.where(m.coef_ > 1e-8)[0]
        if len(nz) >= k:
            best = np.argsort(m.coef_)[-k:][::-1].tolist()
            break
        elif len(nz) > 0:
            best = nz.tolist()
    if len(best) < k:
        corrs = np.array([np.corrcoef(r_index, r_stocks[:, i])[0, 1]
                          for i in range(r_stocks.shape[1])])
        corrs = np.nan_to_num(corrs, nan=-1)
        for i in np.argsort(corrs)[::-1]:
            if i not in best:
                best.append(i)
            if len(best) == k:
                break
    return best[:k]


def select_pca(r_index, r_stocks, mean_mcap, k):
    nc = min(3, r_stocks.shape[1])
    pca = PCA(n_components=nc)
    pca.fit(r_stocks)
    importance = np.zeros(r_stocks.shape[1])
    for i in range(nc):
        importance += np.abs(pca.components_[i]) * pca.explained_variance_ratio_[i]
    return np.argsort(importance)[-k:][::-1].tolist()


def select_random_forest(r_index, r_stocks, mean_mcap, k):
    rf = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42, n_jobs=1)
    rf.fit(r_stocks, r_index)
    return np.argsort(rf.feature_importances_)[-k:][::-1].tolist()


def select_clustering(r_index, r_stocks, mean_mcap, k):
    n = r_stocks.shape[1]
    if k >= n:
        return list(range(n))
    scaler = StandardScaler()
    r_scaled = scaler.fit_transform(r_stocks.T)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(r_scaled)
    selected = []
    for c in range(k):
        members = np.where(labels == c)[0]
        if len(members) == 0:
            continue
        corrs = [np.corrcoef(r_index, r_stocks[:, m])[0, 1] for m in members]
        corrs = [c if not np.isnan(c) else -1 for c in corrs]
        selected.append(members[np.argmax(corrs)])
    while len(selected) < k:
        for i in range(n):
            if i not in selected:
                selected.append(i)
                break
    return selected[:k]


SEL_FUNCS = {
    "top_mcap": select_top_mcap,
    "top_corr": select_top_corr,
    "greedy_te": select_greedy_te,
    "lasso": select_lasso,
    "elastic_net": select_elastic_net,
    "pca": select_pca,
    "random_forest": select_random_forest,
    "clustering": select_clustering,
}


# ============================================================================
# MODULE 3 : OPTIMISATION DES POIDS (5)
# ============================================================================
def optim_min_te(r_index, r_sub, caps=None, mcaps=None):
    k = r_sub.shape[1]
    w0 = np.ones(k) / k

    def obj(w):
        return np.std(r_sub @ w - r_index)

    bounds = [(0, 1)] * k
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    try:
        res = minimize(obj, w0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
        if res.success:
            w = np.maximum(res.x, 0)
            return w / w.sum() if w.sum() > 0 else w0
    except Exception:
        pass
    return w0


def optim_min_te_constrained(r_index, r_sub, caps=None, mcaps=None):
    k = r_sub.shape[1]
    w0 = np.ones(k) / k
    if caps is None:
        caps = np.ones(k)

    def obj(w):
        return np.std(r_sub @ w - r_index)

    bounds = [(0, min(c, 1)) for c in caps]
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    try:
        res = minimize(obj, w0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
        if res.success:
            w = np.maximum(res.x, 0)
            return w / w.sum() if w.sum() > 0 else w0
    except Exception:
        pass
    return w0


def optim_ols(r_index, r_sub, caps=None, mcaps=None):
    k = r_sub.shape[1]
    try:
        m = LinearRegression(fit_intercept=False)
        m.fit(r_sub, r_index)
        w = np.maximum(m.coef_, 0)
        return w / w.sum() if w.sum() > 0 else np.ones(k) / k
    except Exception:
        return np.ones(k) / k


def optim_ridge(r_index, r_sub, caps=None, mcaps=None):
    k = r_sub.shape[1]
    try:
        m = Ridge(alpha=1.0, fit_intercept=False)
        m.fit(r_sub, r_index)
        w = np.maximum(m.coef_, 0)
        return w / w.sum() if w.sum() > 0 else np.ones(k) / k
    except Exception:
        return np.ones(k) / k


def optim_prop_mcap(r_index, r_sub, caps=None, mcaps=None):
    k = r_sub.shape[1]
    if mcaps is not None and len(mcaps) == k:
        w = mcaps / mcaps.sum()
    else:
        w = np.ones(k) / k
    return w


OPT_FUNCS = {
    "min_te": optim_min_te,
    "min_te_constrained": optim_min_te_constrained,
    "ols": optim_ols,
    "ridge": optim_ridge,
    "prop_mcap": optim_prop_mcap,
}


# ============================================================================
# MODULE 4 : ÉVALUATION D'UNE COMBINAISON (sélection + optim + rebal)
# ============================================================================
def evaluate_combo(r_index, r_stocks, selected_idx, opt_method,
                   rebal_period, caps, mcaps_float):
    """
    Évalue une combinaison (sélection déjà faite) :
    Rolling window avec rééquilibrage.
    Retourne (TE_non_annualisé, corrélation).
    """
    T = len(r_index)
    lookback = max(60, rebal_period * 2)
    if lookback >= T:
        lookback = T // 2

    opt_func = OPT_FUNCS[opt_method]
    all_diffs = []

    t = lookback
    while t < T:
        # In-sample
        t0 = max(0, t - lookback)
        r_idx_is = r_index[t0:t]
        r_sub_is = r_stocks[t0:t][:, selected_idx]

        # Poids
        w = opt_func(r_idx_is, r_sub_is, caps=caps, mcaps=mcaps_float)

        # Out-of-sample
        t_end = min(t + rebal_period, T)
        r_idx_oos = r_index[t:t_end]
        r_sub_oos = r_stocks[t:t_end][:, selected_idx]
        if len(r_idx_oos) == 0:
            break

        r_port = r_sub_oos @ w
        all_diffs.extend((r_port - r_idx_oos).tolist())
        t += rebal_period

    if len(all_diffs) < 5:
        return np.nan, np.nan

    diffs = np.array(all_diffs)
    te = np.std(diffs)

    # Corrélation portfolio vs indice (reconstruction)
    n = len(diffs)
    r_idx_eval = r_index[lookback:lookback + n]
    r_port_eval = r_idx_eval + diffs
    if len(r_idx_eval) > 2:
        corr = np.corrcoef(r_idx_eval, r_port_eval)[0, 1]
    else:
        corr = np.nan

    return te, corr


# ============================================================================
# MODULE 5 : GRID SEARCH PRINCIPAL
# ============================================================================
def run_grid_search(yearly_data, df_flottant):
    """
    Architecture optimisée:
    1. Pour chaque année et chaque k, précomputer les sélections (8 méthodes)
    2. Pour chaque sélection, tester les 5 optims × 6 fréquences
    """
    print("\n" + "=" * 60)
    print("MODULE 5 : GRID SEARCH")
    print("=" * 60)

    total = len(FILES) * len(K_VALUES) * len(SELECTION_METHODS) * len(OPTIM_METHODS) * len(REBAL_DAYS)
    print(f"  Total combinaisons: {total}")

    all_results = []
    count = 0
    t0 = time.time()

    for year_label, ydata in yearly_data.items():
        r_index = ydata["r_index"]
        r_stocks = ydata["r_stocks"]
        stock_cols = ydata["stock_cols"]
        mean_mcap = ydata["mean_mcap"]

        # Flottant / plafonnement
        df_fl = df_flottant[df_flottant["Année"] == year_label]
        flot_dict = dict(zip(df_fl["Valeur"], df_fl["Flottant"]))
        plaf_dict = dict(zip(df_fl["Valeur"], df_fl["Plafonnement"]))

        print(f"\n  📅 {year_label} ({len(r_index)} jours)")

        for k in K_VALUES:
            # STEP 1: Précomputer TOUTES les sélections pour ce k
            selections = {}
            for sel in SELECTION_METHODS:
                try:
                    idx = SEL_FUNCS[sel](r_index, r_stocks, mean_mcap, k)
                    # S'assurer qu'on a bien k indices valides
                    idx = [i for i in idx if 0 <= i < r_stocks.shape[1]][:k]
                    if len(idx) < 2:
                        idx = list(range(min(k, r_stocks.shape[1])))
                except Exception:
                    idx = list(range(min(k, r_stocks.shape[1])))
                selections[sel] = idx

            # STEP 2: Pour chaque sélection, tester toutes les optims et fréquences
            for sel, selected_idx in selections.items():
                sel_names = [stock_cols[i] for i in selected_idx]
                caps = np.array([plaf_dict.get(s, 1.0) for s in sel_names])
                flotts = np.array([flot_dict.get(s, 1.0) for s in sel_names])
                mcaps_sel = mean_mcap[selected_idx]
                mcaps_float = mcaps_sel * flotts

                for opt in OPTIM_METHODS:
                    for rebal in REBAL_DAYS:
                        count += 1
                        te, corr = evaluate_combo(
                            r_index, r_stocks, selected_idx,
                            opt, rebal, caps, mcaps_float
                        )
                        all_results.append({
                            "Année": year_label,
                            "k": k,
                            "Sélection": sel,
                            "Optimisation": opt,
                            "Rebal_jours": rebal,
                            "TE": te,
                            "Corr": corr,
                            "Actions": ", ".join(sel_names),
                        })

                        if count % 1000 == 0:
                            e = time.time() - t0
                            pct = count / total * 100
                            print(f"    [{pct:5.1f}%] {count}/{total} — {e:.0f}s")

        print(f"    ✅ {year_label} terminé")

    elapsed = time.time() - t0
    print(f"\n  ⏱️  Grid search: {count} combos en {elapsed:.1f}s")
    return pd.DataFrame(all_results)


# ============================================================================
# MODULE 7 : RAPPORT EXCEL
# ============================================================================
def generate_report(df_res):
    print("\n" + "=" * 60)
    print("MODULE 7 : Rapport Excel")
    print("=" * 60)

    # Agrégation par combinaison (moyenne sur les années)
    grp = ["k", "Sélection", "Optimisation", "Rebal_jours"]
    df_agg = df_res.groupby(grp).agg(
        TE_moyen=("TE", "mean"),
        TE_median=("TE", "median"),
        TE_max=("TE", "max"),
        TE_std=("TE", "std"),
        Corr_moy=("Corr", "mean"),
    ).reset_index().sort_values("TE_moyen")

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as w:
        # 1. Brut
        df_res.to_excel(w, sheet_name="Brut", index=False)

        # 2. Agrégé
        df_agg.to_excel(w, sheet_name="Agrégé", index=False)

        # 3. Top 30
        df_agg.head(30).to_excel(w, sheet_name="Top_30", index=False)

        # 4. Par Sélection
        df_res.groupby("Sélection").agg(
            TE_moyen=("TE", "mean"), TE_med=("TE", "median"), TE_max=("TE", "max")
        ).sort_values("TE_moyen").to_excel(w, sheet_name="Par_Sélection")

        # 5. Par Optimisation
        df_res.groupby("Optimisation").agg(
            TE_moyen=("TE", "mean"), TE_med=("TE", "median"), TE_max=("TE", "max")
        ).sort_values("TE_moyen").to_excel(w, sheet_name="Par_Optimisation")

        # 6. Par Fréquence
        df_res.groupby("Rebal_jours").agg(
            TE_moyen=("TE", "mean"), TE_med=("TE", "median"), TE_max=("TE", "max")
        ).sort_values("TE_moyen").to_excel(w, sheet_name="Par_Fréquence")

        # 7. Par k
        df_res.groupby("k").agg(
            TE_moyen=("TE", "mean"), TE_med=("TE", "median"), TE_max=("TE", "max")
        ).sort_values("TE_moyen").to_excel(w, sheet_name="Par_k")

        # 8. Meilleure combinaison détaillée
        if len(df_agg) > 0:
            b = df_agg.iloc[0]
            best_detail = df_res[
                (df_res["k"] == b["k"]) &
                (df_res["Sélection"] == b["Sélection"]) &
                (df_res["Optimisation"] == b["Optimisation"]) &
                (df_res["Rebal_jours"] == b["Rebal_jours"])
            ]
            best_detail.to_excel(w, sheet_name="Meilleure_Combi", index=False)

        # 9. Heatmap Sélection × Optimisation
        hm1 = df_res.groupby(["Sélection", "Optimisation"])["TE"].mean().reset_index()
        hm1.pivot(index="Sélection", columns="Optimisation", values="TE")\
           .to_excel(w, sheet_name="Heatmap_Sel_x_Opt")

        # 10. Heatmap k × Fréquence
        hm2 = df_res.groupby(["k", "Rebal_jours"])["TE"].mean().reset_index()
        hm2.pivot(index="k", columns="Rebal_jours", values="TE")\
           .to_excel(w, sheet_name="Heatmap_k_x_Freq")

        # 11. Par Année
        df_res.groupby("Année").agg(
            TE_moyen=("TE", "mean"), TE_med=("TE", "median"), TE_max=("TE", "max")
        ).to_excel(w, sheet_name="Par_Année")

    print(f"  📄 Rapport: {OUTPUT_FILE}")
    return df_agg


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("╔════════════════════════════════════════════════════╗")
    print("║  MASI 20 — Réplication Partielle Exhaustive       ║")
    print("║  8 sélections × 5 optims × 6 freq × 9 k          ║")
    print("╚════════════════════════════════════════════════════╝\n")

    # 1. Chargement
    yearly_data, df_flottant = load_all_data()

    # 2. Grid search
    df_res = run_grid_search(yearly_data, df_flottant)

    # 3. Rapport
    df_agg = generate_report(df_res)

    # 4. Synthèse console
    print("\n" + "=" * 60)
    print("SYNTHÈSE")
    print("=" * 60)

    valid = df_agg.dropna(subset=["TE_moyen"])
    if len(valid) > 0:
        b = valid.iloc[0]
        print(f"\n  🏆 MEILLEURE COMBINAISON:")
        print(f"     k = {int(b['k'])}")
        print(f"     Sélection    = {b['Sélection']}")
        print(f"     Optimisation = {b['Optimisation']}")
        print(f"     Rebalancing  = {int(b['Rebal_jours'])} jours")
        print(f"     TE moyen     = {b['TE_moyen']:.6f}")
        print(f"     TE médian    = {b['TE_median']:.6f}")
        print(f"     TE max       = {b['TE_max']:.6f}")
        print(f"     Corr moyenne = {b['Corr_moy']:.4f}")

        print(f"\n  📊 Top 10:")
        for i, (_, r) in enumerate(valid.head(10).iterrows()):
            print(f"     {i+1:2d}. k={int(r['k'])}, sel={r['Sélection']:<16s}, "
                  f"opt={r['Optimisation']:<22s}, rebal={int(r['Rebal_jours']):2d}j "
                  f"→ TE={r['TE_moyen']:.6f}")

    print("\n  📌 Par méthode de sélection:")
    for s, te in df_res.groupby("Sélection")["TE"].mean().sort_values().items():
        print(f"     {s:<20s} → TE = {te:.6f}")

    print("\n  📌 Par optimisation:")
    for o, te in df_res.groupby("Optimisation")["TE"].mean().sort_values().items():
        print(f"     {o:<22s} → TE = {te:.6f}")

    print("\n  📌 Par fréquence:")
    for f, te in df_res.groupby("Rebal_jours")["TE"].mean().sort_values().items():
        print(f"     {int(f):3d} jours → TE = {te:.6f}")

    print("\n  📌 Par k:")
    for kv, te in df_res.groupby("k")["TE"].mean().sort_values().items():
        print(f"     k={int(kv):2d} → TE = {te:.6f}")

    print("\n" + "=" * 60)
    print("✅ TERMINÉ")
    print("=" * 60)


if __name__ == "__main__":
    main()
