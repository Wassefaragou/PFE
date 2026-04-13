"""
=============================================================================
MASI 20 – Réplication Partielle Exhaustive v2 (STRICT OUT-OF-SAMPLE)
=============================================================================
AUCUN in-sample : La sélection ET l'optimisation utilisent UNIQUEMENT
des données passées. Deux approches :

  APPROCHE 1 – TRAIN/TEST SPLIT (50/50)
    • Première moitié de l'année = TRAIN (sélection + calibration poids)
    • Seconde moitié = TEST (évaluation pure OOS, rebalancing des poids
      uniquement sur train étendu ou rolling passé)

  APPROCHE 2 – WALK-FORWARD
    • À chaque date de rééquilibrage, sélection ET optimisation sur
      les données passées uniquement (lookback window)
    • Évaluation sur la période suivante

Évaluation : Tracking Error NON annualisé = std(r_portfolio - r_index)
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

K_VALUES = list(range(2, 11))
REBAL_DAYS = [5, 10, 15, 21, 42, 63]

SELECTION_METHODS = [
    "beta_weight", "beta_corr",
]
OPTIM_METHODS = [
    "min_te", "min_te_constrained", "ols", "ridge", "prop_mcap",
]

OUTPUT_FILE = os.path.join(BASE_PATH, "resultats_remedy.xlsx")


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
        
        # Nettoyage : plafonner les rendements extrêmes (ex: erreurs de données)
        # On définit un seuil strict à +/- 10% pour les log returns
        rets = rets.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-0.10, 0.10)

        # MarketCap brutes (pour la sélection top_mcap)
        mcap_matrix = df[stock_cols].values  # (T_raw, N)

        yearly_data[year_label] = {
            "r_index": rets[index_col].values,
            "r_stocks": rets[stock_cols].values,
            "stock_cols": stock_cols,
            "mcap_matrix": mcap_matrix[1:],  # aligner avec rendements
        }
        print(f"  ✅ {year_label}: {len(rets)} jours, {len(stock_cols)} actions")

    df_flottant = pd.read_excel(os.path.join(BASE_PATH, FLOTTANT_FILE))
    print(f"  ✅ Table flottant: {len(df_flottant)} lignes")
    return yearly_data, df_flottant


# ============================================================================
# MODULE 2 : SÉLECTION (8 méthodes)
# Toutes prennent (r_index, r_stocks, mean_mcap, k) sur une fenêtre donnée
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
    # Compléter par corrélation
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

def select_pca(r_index, r_stocks, mean_mcap, k):
    nc = min(3, r_stocks.shape[1])
    pca = PCA(n_components=nc)
    pca.fit(r_stocks)
    imp = np.zeros(r_stocks.shape[1])
    for i in range(nc):
        imp += np.abs(pca.components_[i]) * pca.explained_variance_ratio_[i]
    return np.argsort(imp)[-k:][::-1].tolist()

def select_random_forest(r_index, r_stocks, mean_mcap, k):
    rf = RandomForestRegressor(n_estimators=30, max_depth=3, random_state=42, n_jobs=1)
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
        corrs = [x if not np.isnan(x) else -1 for x in corrs]
        selected.append(members[np.argmax(corrs)])
    while len(selected) < k:
        for i in range(n):
            if i not in selected:
                selected.append(i)
                break
    return selected[:k]




def select_beta_weight(r_index, r_stocks, mean_mcap, k):
    var_m = np.var(r_index)
    if var_m == 0:
        betas = np.zeros(r_stocks.shape[1])
    else:
        betas = []
        for i in range(r_stocks.shape[1]):
            cov = np.cov(r_stocks[:, i], r_index)[0, 1]
            betas.append(cov / var_m)
        betas = np.array(betas)
    
    # Poids = Mcap relative
    weights = mean_mcap / mean_mcap.sum() if mean_mcap.sum() > 0 else np.zeros_like(mean_mcap)
    
    # Score = Beta * Poids
    scores = betas * weights
    return np.argsort(scores)[-k:][::-1].tolist()

def select_beta_corr(r_index, r_stocks, mean_mcap, k):
    var_m = np.var(r_index)
    if var_m == 0:
        betas = np.zeros(r_stocks.shape[1])
    else:
        betas = []
        for i in range(r_stocks.shape[1]):
            cov = np.cov(r_stocks[:, i], r_index)[0, 1]
            betas.append(cov / var_m)
        betas = np.array(betas)
        
    corrs = np.array([np.corrcoef(r_index, r_stocks[:, i])[0, 1]
                      for i in range(r_stocks.shape[1])])
    corrs = np.nan_to_num(corrs, nan=0)
    
    # Score = Beta * Correlation
    scores = betas * corrs
    return np.argsort(scores)[-k:][::-1].tolist()

SEL_FUNCS = {
    "top_mcap": select_top_mcap,
    "top_corr": select_top_corr,
    "greedy_te": select_greedy_te,
    "lasso": select_lasso,
    "elastic_net": select_elastic_net,
    "pca": select_pca,
    "random_forest": select_random_forest,
    "clustering": select_clustering,
    "beta_weight": select_beta_weight,
    "beta_corr": select_beta_corr,
}


def run_selection(method, r_index, r_stocks, mean_mcap, k):
    """Sélection avec fallback."""
    try:
        idx = SEL_FUNCS[method](r_index, r_stocks, mean_mcap, k)
        idx = [i for i in idx if 0 <= i < r_stocks.shape[1]][:k]
        if len(idx) < 2:
            idx = select_top_corr(r_index, r_stocks, mean_mcap, k)
    except Exception:
        idx = select_top_corr(r_index, r_stocks, mean_mcap, k)
    return idx


# ============================================================================
# MODULE 3 : OPTIMISATION DES POIDS (5 méthodes)
# ============================================================================
def optim_min_te(r_index, r_sub, caps=None, mcaps=None):
    k = r_sub.shape[1]
    w0 = np.ones(k) / k
    def obj(w): return np.std(r_sub @ w - r_index)
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
    def obj(w): return np.std(r_sub @ w - r_index)
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
        w = mcaps / mcaps.sum() if mcaps.sum() > 0 else np.ones(k) / k
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
# APPROCHE 1 : TRAIN/TEST SPLIT (50/50)
# ============================================================================
# Sélection faite UNE FOIS sur TRAIN. Poids recalibrés sur TRAIN.
# Évaluation PURE OOS sur TEST avec rebalancing des poids uniquement
# recalculés sur la partie TRAIN (pas d'accès aux données TEST).
# ============================================================================
def run_train_test_split(yearly_data, df_flottant):
    """
    Split 50/50.
    - TRAIN : sélection + optimisation → poids fixés
    - TEST  : évaluation OOS avec rééquilibrage (poids re-optimisés
              sur TRAIN étendu jusqu'au point de rebalancing passé,
              mais sélection fixée sur TRAIN uniquement)
    """
    print("\n" + "=" * 60)
    print("APPROCHE 1 : TRAIN/TEST SPLIT (50/50)")
    print("=" * 60)

    total = len(FILES) * len(K_VALUES) * len(SELECTION_METHODS) * \
            len(OPTIM_METHODS) * len(REBAL_DAYS)
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
        split = T // 2  # 50/50

        # TRAIN / TEST
        r_idx_train = r_index[:split]
        r_stk_train = r_stocks[:split]
        mcap_train = mcap_matrix[:split]
        mean_mcap_train = mcap_train.mean(axis=0)

        r_idx_test = r_index[split:]
        r_stk_test = r_stocks[split:]

        # Flottant / plafonnement
        df_fl = df_flottant[df_flottant["Année"] == year_label]
        flot_dict = dict(zip(df_fl["Valeur"], df_fl["Flottant"]))
        plaf_dict = dict(zip(df_fl["Valeur"], df_fl["Plafonnement"]))

        print(f"\n  📅 {year_label}: {T} jours (train={split}, test={T-split})")

        for k in K_VALUES:
            # Sélection sur TRAIN uniquement
            selections = {}
            for sel in SELECTION_METHODS:
                idx = run_selection(sel, r_idx_train, r_stk_train, mean_mcap_train, k)
                selections[sel] = idx

            for sel, selected_idx in selections.items():
                sel_names = [stock_cols[i] for i in selected_idx]
                caps = np.array([plaf_dict.get(s, 1.0) for s in sel_names])
                flotts = np.array([flot_dict.get(s, 1.0) for s in sel_names])
                mcaps_sel = mean_mcap_train[selected_idx]
                mcaps_float = mcaps_sel * flotts

                for opt in OPTIM_METHODS:
                    for rebal in REBAL_DAYS:
                        count += 1
                        # Optimisation des poids sur TRAIN
                        r_sub_train = r_stk_train[:, selected_idx]
                        w_train = OPT_FUNCS[opt](
                            r_idx_train, r_sub_train,
                            caps=caps, mcaps=mcaps_float
                        )

                        # Évaluation OOS sur TEST avec rebalancing
                        # À chaque rebalancing, on re-optimise les poids sur
                        # toutes les données PASSÉES (train + test passé)
                        T_test = len(r_idx_test)
                        all_diffs = []
                        t = 0

                        while t < T_test:
                            # Données disponibles = train + test[0:t]
                            r_idx_avail = np.concatenate([r_idx_train, r_idx_test[:t]]) if t > 0 else r_idx_train
                            r_stk_avail = np.concatenate([r_stk_train, r_stk_test[:t]]) if t > 0 else r_stk_train

                            r_sub_avail = r_stk_avail[:, selected_idx]

                            # Re-optimiser poids sur données disponibles
                            if len(r_sub_avail) > 5:
                                w = OPT_FUNCS[opt](
                                    r_idx_avail, r_sub_avail,
                                    caps=caps, mcaps=mcaps_float
                                )
                            else:
                                w = w_train

                            # Évaluer OOS sur [t, t+rebal]
                            t_end = min(t + rebal, T_test)
                            r_idx_oos = r_idx_test[t:t_end]
                            r_sub_oos = r_stk_test[t:t_end, :][:, selected_idx]

                            if len(r_idx_oos) == 0:
                                break

                            r_port = r_sub_oos @ w
                            all_diffs.extend((r_port - r_idx_oos).tolist())
                            t += rebal

                        if len(all_diffs) < 3:
                            te, corr, beta, cov = np.nan, np.nan, np.nan, np.nan
                        else:
                            diffs = np.array(all_diffs)
                            te = np.std(diffs)
                            r_idx_eval = r_idx_test[:len(diffs)]
                            r_port_eval = r_idx_eval + diffs
                            
                            if len(r_idx_eval) > 2:
                                cov_mat = np.cov(r_port_eval, r_idx_eval) # Sample cov
                                cov = cov_mat[0, 1]
                                var_idx = cov_mat[1, 1]
                                beta = cov / var_idx if var_idx > 0 else np.nan
                                corr = cov / (np.std(r_port_eval, ddof=1) * np.std(r_idx_eval, ddof=1))
                            else:
                                corr, beta, cov = np.nan, np.nan, np.nan

                        all_results.append({
                            "Approche": "Train/Test",
                            "Année": year_label,
                            "k": k,
                            "Sélection": sel,
                            "Optimisation": opt,
                            "Rebal_jours": rebal,
                            "TE": te,
                            "Corr": corr,
                            "Beta": beta,
                            "Cov": cov,
                            "Actions": ", ".join(sel_names),
                        })

                        if count % 1000 == 0:
                            e = time.time() - t0
                            pct = count / total * 100
                            print(f"    [{pct:5.1f}%] {count}/{total} — {e:.0f}s")

        print(f"    ✅ {year_label} terminé")

    elapsed = time.time() - t0
    print(f"  ⏱️  Approche 1: {count} combos en {elapsed:.1f}s")
    return pd.DataFrame(all_results)


# ============================================================================
# APPROCHE 2 : WALK-FORWARD (le plus rigoureux)
# ============================================================================
# À chaque date de rééquilibrage : sélection ET optimisation faites
# UNIQUEMENT sur le lookback window passé. Zéro accès au futur.
# ============================================================================
def run_walk_forward(yearly_data, df_flottant):
    """
    Walk-forward complet. À chaque date de rééquilibrage :
    1. Sélection sur données passées (lookback window)
    2. Optimisation sur données passées (même window)
    3. Évaluation OOS sur la période suivante
    """
    print("\n" + "=" * 60)
    print("APPROCHE 2 : WALK-FORWARD COMPLET")
    print("=" * 60)

    total = len(FILES) * len(K_VALUES) * len(SELECTION_METHODS) * \
            len(OPTIM_METHODS) * len(REBAL_DAYS)
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

        for k in K_VALUES:
            for sel in SELECTION_METHODS:
                for opt in OPTIM_METHODS:
                    for rebal in REBAL_DAYS:
                        count += 1
                        lookback = max(60, rebal * 2)
                        if lookback >= T:
                            lookback = T // 2

                        all_diffs = []
                        t = lookback

                        while t < T:
                            # === Fenêtre PASSÉE uniquement ===
                            t0_w = max(0, t - lookback)
                            r_idx_past = r_index[t0_w:t]
                            r_stk_past = r_stocks[t0_w:t]
                            mcap_past = mcap_matrix[t0_w:t]
                            mean_mcap_past = mcap_past.mean(axis=0)

                            # 1. SÉLECTION sur données passées
                            selected_idx = run_selection(
                                sel, r_idx_past, r_stk_past,
                                mean_mcap_past, k
                            )

                            sel_names = [stock_cols[i] for i in selected_idx]
                            caps = np.array([plaf_dict.get(s, 1.0) for s in sel_names])
                            flotts = np.array([flot_dict.get(s, 1.0) for s in sel_names])
                            mcaps_sel = mean_mcap_past[selected_idx]
                            mcaps_float = mcaps_sel * flotts

                            # 2. OPTIMISATION sur données passées
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
                            te, corr, beta, cov = np.nan, np.nan, np.nan, np.nan
                        else:
                            diffs = np.array(all_diffs)
                            te = np.std(diffs)
                            n = len(diffs)
                            r_idx_eval = r_index[lookback:lookback + n]
                            r_port_eval = r_idx_eval + diffs
                            
                            if len(r_idx_eval) > 2:
                                cov_mat = np.cov(r_port_eval, r_idx_eval)
                                cov = cov_mat[0, 1]
                                var_idx = cov_mat[1, 1]
                                beta = cov / var_idx if var_idx > 0 else np.nan
                                corr = cov / (np.std(r_port_eval, ddof=1) * np.std(r_idx_eval, ddof=1))
                            else:
                                corr, beta, cov = np.nan, np.nan, np.nan

                        all_results.append({
                            "Approche": "Walk-Forward",
                            "Année": year_label,
                            "k": k,
                            "Sélection": sel,
                            "Optimisation": opt,
                            "Rebal_jours": rebal,
                            "TE": te,
                            "Corr": corr,
                            "Beta": beta,
                            "Cov": cov,
                            "Actions": ", ".join(sel_names),
                        })

                        if count % 500 == 0:
                            e = time.time() - t0
                            pct = count / total * 100
                            print(f"    [{pct:5.1f}%] {count}/{total} — {e:.0f}s")

        print(f"    ✅ {year_label} terminé")

    elapsed = time.time() - t0
    print(f"  ⏱️  Approche 2: {count} combos en {elapsed:.1f}s")
    return pd.DataFrame(all_results)


# ============================================================================
# MODULE RAPPORT EXCEL
# ============================================================================
def generate_report(df_all):
    """Génère le rapport avec comparaison des deux approches."""
    print("\n" + "=" * 60)
    print("RAPPORT EXCEL")
    print("=" * 60)

    grp = ["Approche", "k", "Sélection", "Optimisation", "Rebal_jours"]
    df_agg = df_all.groupby(grp).agg(
        TE_moyen=("TE", "mean"),
        TE_median=("TE", "median"),
        TE_max=("TE", "max"),
        TE_std=("TE", "std"),
        Corr_moy=("Corr", "mean"),
    ).reset_index().sort_values(["Approche", "TE_moyen"])

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as w:
        # 1. Brut
        df_all.to_excel(w, sheet_name="Brut", index=False)

        # 2. Agrégé
        df_agg.to_excel(w, sheet_name="Agrégé", index=False)

        # 3. Top 30 par approche
        for app in df_agg["Approche"].unique():
            sub = df_agg[df_agg["Approche"] == app].head(30)
            name = "Top30_" + ("TT" if "Train" in str(app) else "WF")
            sub.to_excel(w, sheet_name=name, index=False)

        # 4. Par dimension — pour chaque approche
        for app in df_all["Approche"].unique():
            sub = df_all[df_all["Approche"] == app]
            tag = "TT" if "Train" in str(app) else "WF"

            sub.groupby("Sélection")["TE"].agg(["mean", "median", "max"])\
               .sort_values("mean").to_excel(w, sheet_name=f"Sel_{tag}")

            sub.groupby("Optimisation")["TE"].agg(["mean", "median", "max"])\
               .sort_values("mean").to_excel(w, sheet_name=f"Opt_{tag}")

            sub.groupby("Rebal_jours")["TE"].agg(["mean", "median", "max"])\
               .sort_values("mean").to_excel(w, sheet_name=f"Freq_{tag}")

            sub.groupby("k")["TE"].agg(["mean", "median", "max"])\
               .sort_values("mean").to_excel(w, sheet_name=f"K_{tag}")

            sub.groupby("Année")["TE"].agg(["mean", "median", "max"])\
               .to_excel(w, sheet_name=f"Année_{tag}")

        # 5. Comparaison directe
        cmp = df_all.groupby(["Approche", "Sélection"]).agg(
            TE_moyen=("TE", "mean")
        ).reset_index()
        cmp_piv = cmp.pivot(index="Sélection", columns="Approche", values="TE_moyen")
        cmp_piv.to_excel(w, sheet_name="Comparaison_Sel")

        cmp2 = df_all.groupby(["Approche", "Optimisation"]).agg(
            TE_moyen=("TE", "mean")
        ).reset_index()
        cmp2_piv = cmp2.pivot(index="Optimisation", columns="Approche", values="TE_moyen")
        cmp2_piv.to_excel(w, sheet_name="Comparaison_Opt")

        # 6. Heatmaps
        for app in df_all["Approche"].unique():
            sub = df_all[df_all["Approche"] == app]
            tag = "TT" if "Train" in str(app) else "WF"

            hm = sub.groupby(["Sélection", "Optimisation"])["TE"].mean().reset_index()
            hm.pivot(index="Sélection", columns="Optimisation", values="TE")\
              .to_excel(w, sheet_name=f"HM_SelOpt_{tag}")

            hm2 = sub.groupby(["k", "Rebal_jours"])["TE"].mean().reset_index()
            hm2.pivot(index="k", columns="Rebal_jours", values="TE")\
               .to_excel(w, sheet_name=f"HM_kFreq_{tag}")

        # 7. Meilleure combi par approche
        for app in df_agg["Approche"].unique():
            sub_agg = df_agg[df_agg["Approche"] == app]
            tag = "TT" if "Train" in str(app) else "WF"
            if len(sub_agg) > 0:
                b = sub_agg.iloc[0]
                best_detail = df_all[
                    (df_all["Approche"] == app) &
                    (df_all["k"] == b["k"]) &
                    (df_all["Sélection"] == b["Sélection"]) &
                    (df_all["Optimisation"] == b["Optimisation"]) &
                    (df_all["Rebal_jours"] == b["Rebal_jours"])
                ]
                best_detail.to_excel(w, sheet_name=f"Best_{tag}", index=False)

    print(f"  📄 Rapport: {OUTPUT_FILE}")
    return df_agg


# ============================================================================
# SYNTHÈSE CONSOLE
# ============================================================================
def print_synthesis(df_all, df_agg):
    print("\n" + "=" * 60)
    print("SYNTHÈSE FINALE")
    print("=" * 60)

    for app in ["Train/Test", "Walk-Forward"]:
        sub_agg = df_agg[df_agg["Approche"] == app].reset_index(drop=True)
        sub = df_all[df_all["Approche"] == app]

        if len(sub_agg) == 0:
            continue

        tag = "TRAIN/TEST" if "Train" in app else "WALK-FORWARD"
        print(f"\n  ={'=' * 50}")
        print(f"  {'🔷' if 'Train' in app else '🔶'} APPROCHE: {tag}")
        print(f"  ={'=' * 50}")

        b = sub_agg.iloc[0]
        print(f"\n  🏆 MEILLEURE COMBINAISON:")
        print(f"     k = {int(b['k'])}")
        print(f"     Sélection    = {b['Sélection']}")
        print(f"     Optimisation = {b['Optimisation']}")
        print(f"     Rebalancing  = {int(b['Rebal_jours'])} jours")
        print(f"     TE moyen     = {b['TE_moyen']:.6f}")
        print(f"     TE médian    = {b['TE_median']:.6f}")
        print(f"     TE max       = {b['TE_max']:.6f}")
        print(f"     Corr moyenne = {b['Corr_moy']:.4f}")

        print(f"\n  📊 Top 5:")
        for i, (_, r) in enumerate(sub_agg.head(5).iterrows()):
            print(f"     {i+1}. k={int(r['k'])}, sel={r['Sélection']:<16s}, "
                  f"opt={r['Optimisation']:<22s}, rebal={int(r['Rebal_jours']):2d}j "
                  f"→ TE={r['TE_moyen']:.6f}")

        print(f"\n  📌 Par sélection:")
        for s, te in sub.groupby("Sélection")["TE"].mean().sort_values().items():
            print(f"     {s:<20s} → TE = {te:.6f}")

        print(f"\n  📌 Par optimisation:")
        for o, te in sub.groupby("Optimisation")["TE"].mean().sort_values().items():
            print(f"     {o:<22s} → TE = {te:.6f}")

        print(f"\n  📌 Par fréquence:")
        for f, te in sub.groupby("Rebal_jours")["TE"].mean().sort_values().items():
            print(f"     {int(f):3d} jours → TE = {te:.6f}")

        print(f"\n  📌 Par k:")
        for kv, te in sub.groupby("k")["TE"].mean().sort_values().items():
            print(f"     k={int(kv):2d} → TE = {te:.6f}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("╔════════════════════════════════════════════════════════╗")
    print("║  MASI 20 — Réplication Partielle v2                   ║")
    print("║  STRICT OUT-OF-SAMPLE (zéro in-sample)                ║")
    print("║  Approche 1: Train/Test 50/50                         ║")
    print("║  Approche 2: Walk-Forward complet                     ║")
    print("║  8 sélections × 5 optims × 6 freq × 9 k              ║")
    print("╚════════════════════════════════════════════════════════╝\n")

    # Chargement
    yearly_data, df_flottant = load_all_data()

    # Approche 1 : Train/Test Split
    df_tt = run_train_test_split(yearly_data, df_flottant)

    # Approche 2 : Walk-Forward
    df_wf = run_walk_forward(yearly_data, df_flottant)

    # Fusion
    df_all = pd.concat([df_tt, df_wf], ignore_index=True)

    # Rapport Excel
    df_agg = generate_report(df_all)

    # Synthèse console
    print_synthesis(df_all, df_agg)

    print("\n" + "=" * 60)
    print("✅ TERMINÉ — Aucun in-sample dans l'évaluation")
    print("=" * 60)


if __name__ == "__main__":
    main()
