"""
AUDIT APPROFONDI — Vérifie chaque aspect du script masi20_replication_v2.py
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Lasso, ElasticNet, Ridge, LinearRegression

BASE_PATH = r"c:\Users\PC\Downloads\Stage\répliaction_partielle"

errors_found = []
warnings_found = []

def ERROR(msg): errors_found.append(f"❌ ERREUR: {msg}"); print(f"❌ ERREUR: {msg}")
def WARN(msg): warnings_found.append(f"⚠️ ATTENTION: {msg}"); print(f"⚠️ ATTENTION: {msg}")
def OK(msg): print(f"  ✅ {msg}")

# ============================================================================
# TEST 1 : CHARGEMENT DES DONNÉES
# ============================================================================
print("=" * 70)
print("TEST 1 : CHARGEMENT DES DONNÉES")
print("=" * 70)

FILES = {
    "2020-2021": "20-21.xlsx", "2021-2022": "21-22.xlsx",
    "2022-2023": "22-23.xlsx", "2023-2024": "23-24.xlsx",
    "2024-2025": "24-25.xlsx", "2025-2026": "25-26-SANSANOMALIE.xlsx",
}

for year_label, filename in FILES.items():
    path = os.path.join(BASE_PATH, filename)
    df = pd.read_excel(path, header=10)
    df = df.dropna(axis=1, how="all")
    df = df[[c for c in df.columns if not str(c).startswith("Unnamed")]]
    
    # Vérifier que Date existe
    if "Date" not in df.columns:
        ERROR(f"{year_label}: Pas de colonne 'Date'")
    else:
        OK(f"{year_label}: Date trouvée")
    
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Vérifier l'index MASI 20
    index_col = [c for c in df.columns if "MSEMSI20" in str(c)]
    if not index_col:
        WARN(f"{year_label}: Pas de colonne MSEMSI20 → utilise colonne 1 comme index")
        index_col = df.columns[1]
    else:
        index_col = index_col[0]
        OK(f"{year_label}: Index trouvé = '{index_col}'")
    
    stock_cols = [c for c in df.columns if c != "Date" and c != index_col]
    
    # Nombre d'actions
    if len(stock_cols) != 20:
        WARN(f"{year_label}: {len(stock_cols)} actions (attendu 20)")
    else:
        OK(f"{year_label}: 20 actions trouvées")
    
    # Vérifier les valeurs numériques
    for col in [index_col] + stock_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    nan_count = df[stock_cols].isna().sum().sum()
    if nan_count > 0:
        WARN(f"{year_label}: {nan_count} valeurs NaN dans les données MarketCap")
    
    # Vérifier les rendements
    rets = df[[index_col] + stock_cols].pct_change().iloc[1:]
    rets = rets.replace([np.inf, -np.inf], np.nan)
    
    inf_count = rets.isna().sum().sum()
    if inf_count > 0:
        WARN(f"{year_label}: {inf_count} valeurs inf/NaN après pct_change (sera rempli par 0)")
    
    rets = rets.fillna(0).clip(-0.5, 0.5)
    
    # Vérifier que les rendements sont dans une plage raisonnable
    r_idx = rets[index_col].values
    r_stk = rets[stock_cols].values
    
    print(f"  📊 {year_label}: Rendement index → mean={r_idx.mean():.6f}, std={r_idx.std():.6f}, min={r_idx.min():.4f}, max={r_idx.max():.4f}")
    
    # Un rendement quotidien moyen > 5% serait suspect
    for i, col in enumerate(stock_cols):
        mean_ret = r_stk[:, i].mean()
        if abs(mean_ret) > 0.05:
            WARN(f"{year_label}: {col} rendement moyen suspect = {mean_ret:.4f}")
    
    # Vérifier alignement mcap / rendements
    mcap_matrix = df[stock_cols].values
    mcap_aligned = mcap_matrix[1:]  # aligné avec rendements
    if mcap_aligned.shape[0] != r_stk.shape[0]:
        ERROR(f"{year_label}: mcap_matrix mal alignée ({mcap_aligned.shape[0]} vs {r_stk.shape[0]})")
    else:
        OK(f"{year_label}: mcap alignée ({mcap_aligned.shape[0]} jours)")

# ============================================================================
# TEST 2 : VÉRIFICATION DES MÉTHODES DE SÉLECTION
# ============================================================================
print("\n" + "=" * 70)
print("TEST 2 : VÉRIFICATION DES MÉTHODES DE SÉLECTION")
print("=" * 70)

# Utiliser une année spécifique pour les tests
path = os.path.join(BASE_PATH, "23-24.xlsx")
df = pd.read_excel(path, header=10)
df = df.dropna(axis=1, how="all")
df = df[[c for c in df.columns if not str(c).startswith("Unnamed")]]
df["Date"] = pd.to_datetime(df["Date"])
index_col = [c for c in df.columns if "MSEMSI20" in str(c)][0]
stock_cols = [c for c in df.columns if c != "Date" and c != index_col]
for col in [index_col] + stock_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
rets = df[[index_col] + stock_cols].pct_change().iloc[1:]
rets = rets.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-0.5, 0.5)
r_index = rets[index_col].values
r_stocks = rets[stock_cols].values
mcap_matrix = df[stock_cols].values[1:]
mean_mcap = mcap_matrix.mean(axis=0)

# Utiliser TRAIN (50%)
split = len(r_index) // 2
r_idx_train = r_index[:split]
r_stk_train = r_stocks[:split]
mean_mcap_train = mcap_matrix[:split].mean(axis=0)

for k in [3, 5, 10]:
    print(f"\n  --- k={k} ---")
    
    # top_mcap
    idx = np.argsort(mean_mcap_train)[-k:][::-1].tolist()
    if len(idx) != k: ERROR(f"top_mcap k={k}: retourne {len(idx)} actions")
    if len(set(idx)) != k: ERROR(f"top_mcap k={k}: doublons dans la sélection")
    if any(i < 0 or i >= 20 for i in idx): ERROR(f"top_mcap k={k}: index hors bornes")
    OK(f"top_mcap k={k}: {idx}")
    
    # top_corr
    corrs = np.array([np.corrcoef(r_idx_train, r_stk_train[:, i])[0, 1] for i in range(20)])
    corrs = np.nan_to_num(corrs, nan=-1)
    idx = np.argsort(corrs)[-k:][::-1].tolist()
    if len(idx) != k: ERROR(f"top_corr k={k}: retourne {len(idx)} actions")
    OK(f"top_corr k={k}: {idx}, corrs={[f'{corrs[i]:.4f}' for i in idx]}")
    
    # lasso
    best = list(range(k))
    for alpha in np.logspace(-6, -1, 15):
        m = Lasso(alpha=alpha, max_iter=2000, positive=True)
        m.fit(r_stk_train, r_idx_train)
        nz = np.where(m.coef_ > 1e-8)[0]
        if len(nz) >= k:
            best = np.argsort(m.coef_)[-k:][::-1].tolist()
            break
    if len(best) != k: ERROR(f"lasso k={k}: retourne {len(best)} actions")
    OK(f"lasso k={k}: {best}")
    
    # elastic_net
    best_en = list(range(k))
    for alpha in np.logspace(-6, -1, 15):
        m = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=2000, positive=True)
        m.fit(r_stk_train, r_idx_train)
        nz = np.where(m.coef_ > 1e-8)[0]
        if len(nz) >= k:
            best_en = np.argsort(m.coef_)[-k:][::-1].tolist()
            break
    if len(best_en) != k: ERROR(f"elastic_net k={k}: retourne {len(best_en)} actions")
    OK(f"elastic_net k={k}: {best_en}")

# ============================================================================
# TEST 3 : VÉRIFICATION DES MÉTHODES D'OPTIMISATION
# ============================================================================
print("\n" + "=" * 70)
print("TEST 3 : VÉRIFICATION DES MÉTHODES D'OPTIMISATION")
print("=" * 70)

k = 5
# Sélectionner les 5 plus grosses market cap (sur train)
selected_idx = np.argsort(mean_mcap_train)[-k:][::-1].tolist()
r_sub = r_stk_train[:, selected_idx]

# min_te
w0 = np.ones(k) / k
def obj(w): return np.std(r_sub @ w - r_idx_train)
bounds = [(0, 1)] * k
cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 300})
w_min_te = np.maximum(res.x, 0)
w_min_te = w_min_te / w_min_te.sum()

print(f"  min_te: w={w_min_te.round(4)}, sum={w_min_te.sum():.6f}")
if abs(w_min_te.sum() - 1.0) > 1e-6: ERROR("min_te: poids ne somment pas à 1")
else: OK("min_te: poids somment à 1")
if any(w < -1e-10 for w in w_min_te): ERROR("min_te: poids négatifs")
else: OK("min_te: poids ≥ 0")

te_train = np.std(r_sub @ w_min_te - r_idx_train)
te_equal = np.std(r_sub @ w0 - r_idx_train)
print(f"  TE(min_te)={te_train:.6f}, TE(equal_weight)={te_equal:.6f}")
if te_train > te_equal:
    WARN("min_te donne un TE supérieur à equal weight sur le train — optimisation possiblement bloquée")
else:
    OK(f"min_te réduit le TE de {te_equal:.6f} à {te_train:.6f} (-{(1-te_train/te_equal)*100:.1f}%)")

# OLS
m = LinearRegression(fit_intercept=False)
m.fit(r_sub, r_idx_train)
w_ols = np.maximum(m.coef_, 0)
w_ols_sum = w_ols.sum()
if w_ols_sum > 0:
    w_ols = w_ols / w_ols_sum
else:
    ERROR("OLS: tous les poids sont négatifs")
    w_ols = np.ones(k) / k

print(f"  OLS: w={w_ols.round(4)}, sum={w_ols.sum():.6f}")
if abs(w_ols.sum() - 1.0) > 1e-6: ERROR("OLS: poids ne somment pas à 1")
else: OK("OLS: poids somment à 1")

# Ridge
m = Ridge(alpha=1.0, fit_intercept=False)
m.fit(r_sub, r_idx_train)
w_ridge = np.maximum(m.coef_, 0)
w_ridge_sum = w_ridge.sum()
if w_ridge_sum > 0:
    w_ridge = w_ridge / w_ridge_sum
print(f"  Ridge: w={w_ridge.round(4)}, sum={w_ridge.sum():.6f}")
OK("Ridge OK")

# prop_mcap
mcaps_float = mean_mcap_train[selected_idx]
w_mcap = mcaps_float / mcaps_float.sum()
print(f"  prop_mcap: w={w_mcap.round(4)}, sum={w_mcap.sum():.6f}")
if abs(w_mcap.sum() - 1.0) > 1e-6: ERROR("prop_mcap: poids ne somment pas à 1")
else: OK("prop_mcap: poids somment à 1")

# ============================================================================
# TEST 4 : VÉRIFICATION OOS — TRAIN/TEST
# ============================================================================
print("\n" + "=" * 70)
print("TEST 4 : VÉRIFICATION OOS — Train/Test")
print("=" * 70)

# Simuler manuellement une évaluation OOS
r_idx_test = r_index[split:]
r_stk_test = r_stocks[split:]
T_test = len(r_idx_test)

print(f"  Train: {split} jours, Test: {T_test} jours")
print(f"  split={split}, T={len(r_index)}")

# Vérifier que train et test ne se chevauchent pas
if split + T_test != len(r_index):
    ERROR(f"Train ({split}) + Test ({T_test}) ≠ Total ({len(r_index)})")
else:
    OK(f"Train + Test = Total ({len(r_index)})")

# Tester avec k=5, min_te, rebal=21
rebal = 21
r_sub_test = r_stk_test[:, selected_idx]

# Simulation manuelle
all_diffs_manual = []
t = 0
while t < T_test:
    # Données disponibles
    if t > 0:
        r_idx_avail = np.concatenate([r_idx_train, r_idx_test[:t]])
        r_stk_avail = np.concatenate([r_stk_train, r_stk_test[:t]])
    else:
        r_idx_avail = r_idx_train
        r_stk_avail = r_stk_train
    
    r_sub_avail = r_stk_avail[:, selected_idx]
    
    # Vérifier qu'on n'utilise PAS de données futures
    avail_len = len(r_idx_avail)
    supposed_max = split + t  # train + test passé
    if avail_len != supposed_max:
        ERROR(f"t={t}: avail_len={avail_len} ≠ train+t={supposed_max}")
    
    # Optimiser
    w0 = np.ones(k) / k
    def obj2(w): return np.std(r_sub_avail @ w - r_idx_avail)
    res2 = minimize(obj2, w0, method="SLSQP", bounds=[(0,1)]*k,
                    constraints=[{"type":"eq","fun":lambda w:w.sum()-1}],
                    options={"maxiter":300})
    w = np.maximum(res2.x, 0)
    w = w / w.sum() if w.sum() > 0 else w0
    
    # Évaluer OOS
    t_end = min(t + rebal, T_test)
    r_idx_oos = r_idx_test[t:t_end]
    r_sub_oos = r_sub_test[t:t_end]
    
    r_port = r_sub_oos @ w
    diffs = r_port - r_idx_oos
    all_diffs_manual.extend(diffs.tolist())
    
    # Vérifier que l'évaluation est bien sur des données FUTURES (après t)
    if t < t_end:
        OK(f"t={t}: optim sur [0, {avail_len}], eval sur [{split+t}, {split+t_end}] → OOS")
    
    t += rebal

te_manual = np.std(all_diffs_manual)
print(f"\n  TE calculé manuellement: {te_manual:.6f}")
print(f"  Nombre de diffs: {len(all_diffs_manual)}, devrait être {T_test}")
if len(all_diffs_manual) != T_test:
    WARN(f"Nombre de diffs ({len(all_diffs_manual)}) ≠ T_test ({T_test}) — normal si T_test pas multiple de rebal")

# ============================================================================
# TEST 5 : VÉRIFICATION OOS — WALK-FORWARD
# ============================================================================
print("\n" + "=" * 70)
print("TEST 5 : VÉRIFICATION OOS — Walk-Forward")
print("=" * 70)

lookback = max(60, rebal * 2)
print(f"  lookback={lookback}, rebal={rebal}")

all_diffs_wf = []
t = lookback

while t < len(r_index):
    t0_w = max(0, t - lookback)
    r_idx_past = r_index[t0_w:t]
    r_stk_past = r_stocks[t0_w:t]
    
    # Vérifier qu'on n'accède PAS au futur
    if t0_w >= t:
        ERROR(f"WF t={t}: t0_w={t0_w} >= t={t} → fenêtre vide")
    
    past_len = len(r_idx_past)
    expected_len = t - t0_w
    if past_len != expected_len:
        ERROR(f"WF t={t}: past_len={past_len} ≠ expected={expected_len}")
    
    # Sélection sur passé
    mean_mcap_past = mcap_matrix[t0_w:t].mean(axis=0) if t0_w < len(mcap_matrix) else mean_mcap
    sel_idx = np.argsort(mean_mcap_past)[-k:][::-1].tolist()  # top_mcap sur passé
    
    r_sub_past = r_stk_past[:, sel_idx]
    
    # Optimisation sur passé
    w0 = np.ones(k) / k
    def obj3(w): return np.std(r_sub_past @ w - r_idx_past)
    res3 = minimize(obj3, w0, method="SLSQP", bounds=[(0,1)]*k,
                    constraints=[{"type":"eq","fun":lambda w:w.sum()-1}],
                    options={"maxiter":300})
    w = np.maximum(res3.x, 0)
    w = w / w.sum() if w.sum() > 0 else w0
    
    # Évaluation OOS
    t_end = min(t + rebal, len(r_index))
    r_idx_oos = r_index[t:t_end]
    r_sub_oos = r_stocks[t:t_end, :][:, sel_idx]
    
    r_port = r_sub_oos @ w
    all_diffs_wf.extend((r_port - r_idx_oos).tolist())
    
    t += rebal

te_wf = np.std(all_diffs_wf)
print(f"  TE Walk-Forward (top_mcap, k=5, rebal=21): {te_wf:.6f}")
print(f"  Jours évalués: {len(all_diffs_wf)}")
OK("Walk-Forward cohérent")

# ============================================================================
# TEST 6 : VÉRIFIER LES RÉSULTATS EXCEL
# ============================================================================
print("\n" + "=" * 70)
print("TEST 6 : VÉRIFICATION DU FICHIER RÉSULTATS")
print("=" * 70)

df_res = pd.read_excel(os.path.join(BASE_PATH, "resultats_replication_v2.xlsx"), sheet_name="Brut")

# Vérifier le nombre de lignes
expected_total = 2 * 6 * 9 * 8 * 5 * 6  # 2 approches × 6 ans × 9 k × 8 sel × 5 opt × 6 freq
print(f"  Lignes trouvées: {len(df_res)}, attendu: {expected_total}")
if len(df_res) != expected_total:
    ERROR(f"Nombre de lignes incorrect: {len(df_res)} vs {expected_total}")
else:
    OK(f"Nombre de lignes correct: {len(df_res)}")

# Vérifier les valeurs manquantes
nan_te = df_res["TE"].isna().sum()
if nan_te > 0:
    WARN(f"{nan_te} TE NaN ({nan_te/len(df_res)*100:.1f}%)")
else:
    OK("Aucun TE NaN")

# Vérifier que les TE sont dans une plage raisonnable
te_min = df_res["TE"].min()
te_max_val = df_res["TE"].max()
te_mean = df_res["TE"].mean()
print(f"  TE: min={te_min:.6f}, max={te_max_val:.6f}, mean={te_mean:.6f}")

if te_min < 0:
    ERROR(f"TE négatif trouvé: {te_min}")
else:
    OK("Tous les TE ≥ 0")

if te_max_val > 0.1:
    WARN(f"TE max très élevé: {te_max_val:.6f} — vérifier les cas extrêmes")
else:
    OK(f"TE max raisonnable: {te_max_val:.6f}")

# Vérifier que toutes les années sont présentes
years = sorted(df_res["Année"].unique())
expected_years = ["2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
if years == expected_years:
    OK(f"Toutes les 6 années présentes")
else:
    ERROR(f"Années manquantes: attendu {expected_years}, trouvé {years}")

# Vérifier que toutes les approches sont présentes
approches = sorted(df_res["Approche"].unique())
if "Train/Test" in approches and "Walk-Forward" in approches:
    OK("Les 2 approches présentes")
else:
    ERROR(f"Approches trouvées: {approches}")

# Par approche et année
for app in approches:
    for y in years:
        sub = df_res[(df_res["Approche"] == app) & (df_res["Année"] == y)]
        expected_per_year = 9 * 8 * 5 * 6  # 2160
        if len(sub) != expected_per_year:
            ERROR(f"{app} / {y}: {len(sub)} lignes (attendu {expected_per_year})")

OK("Toutes les combinaisons année × approche complètes")

# ============================================================================
# TEST 7 : VÉRIFIER QUE TE WALK-FORWARD >= TE TRAIN/TEST (en moyenne)
# ============================================================================
print("\n" + "=" * 70)
print("TEST 7 : COHÉRENCE ENTRE APPROCHES")
print("=" * 70)

te_tt_mean = df_res[df_res["Approche"] == "Train/Test"]["TE"].mean()
te_wf_mean = df_res[df_res["Approche"] == "Walk-Forward"]["TE"].mean()
print(f"  TE moyen Train/Test:  {te_tt_mean:.6f}")
print(f"  TE moyen Walk-Forward: {te_wf_mean:.6f}")

if te_wf_mean > te_tt_mean:
    OK(f"Walk-Forward TE > Train/Test TE → cohérent (WF plus strict)")
else:
    WARN(f"Walk-Forward TE < Train/Test TE — inhabituel mais pas nécessairement faux")

# Vérifier stabilité du classement
for dim in ["Sélection", "Optimisation"]:
    rank_tt = df_res[df_res["Approche"]=="Train/Test"].groupby(dim)["TE"].mean().sort_values()
    rank_wf = df_res[df_res["Approche"]=="Walk-Forward"].groupby(dim)["TE"].mean().sort_values()
    
    top3_tt = list(rank_tt.index[:3])
    top3_wf = list(rank_wf.index[:3])
    
    common = set(top3_tt) & set(top3_wf)
    print(f"  Top 3 {dim} TT: {top3_tt}")
    print(f"  Top 3 {dim} WF: {top3_wf}")
    if len(common) >= 2:
        OK(f"Au moins 2 des 3 meilleures {dim} sont communes → classement stable")
    else:
        WARN(f"Classement {dim} instable entre approches")

# ============================================================================
# TEST 8 : VÉRIFIER LA MONOTONICITÉ DU TE vs K
# ============================================================================
print("\n" + "=" * 70)
print("TEST 8 : MONOTONICITÉ TE vs k")
print("=" * 70)

for app in ["Train/Test", "Walk-Forward"]:
    te_by_k = df_res[df_res["Approche"]==app].groupby("k")["TE"].mean()
    is_monotone = all(te_by_k.iloc[i] >= te_by_k.iloc[i+1] for i in range(len(te_by_k)-1))
    if is_monotone:
        OK(f"{app}: TE décroissant avec k (monotone) ✓")
    else:
        WARN(f"{app}: TE NON monotone avec k — vérifier")
    for k_val in range(2, 11):
        print(f"    k={k_val}: TE={te_by_k[k_val]:.6f}")

# ============================================================================
# TEST 9 : VÉRIFIER QUE LE HEADER DES FICHIERS EST CORRECT
# ============================================================================
print("\n" + "=" * 70)
print("TEST 9 : VÉRIFICATION HEADER=10")
print("=" * 70)

for year_label, filename in FILES.items():
    path = os.path.join(BASE_PATH, filename)
    # Lire les premières lignes brutes
    df_raw = pd.read_excel(path, header=None, nrows=15)
    # Chercher la ligne avec "Date"
    date_row = None
    for i in range(15):
        row_vals = df_raw.iloc[i].astype(str).tolist()
        if any("Date" in str(v) for v in row_vals):
            date_row = i
            break
    if date_row is not None:
        if date_row == 10:
            OK(f"{year_label}: 'Date' trouvé à la ligne {date_row} (header=10 correct)")
        else:
            ERROR(f"{year_label}: 'Date' trouvé à la ligne {date_row} mais header=10 est utilisé!")
    else:
        ERROR(f"{year_label}: 'Date' introuvable dans les 15 premières lignes")

# ============================================================================
# TEST 10 : VÉRIFICATION FLOTTANT
# ============================================================================
print("\n" + "=" * 70)
print("TEST 10 : VÉRIFICATION TABLE FLOTTANT")
print("=" * 70)

df_flot = pd.read_excel(os.path.join(BASE_PATH, "Flottant_plafonnements_masi20.xlsx"))
print(f"  Colonnes: {list(df_flot.columns)}")
print(f"  Années uniques: {sorted(df_flot['Année'].unique())}")

# Vérifier que chaque année a des données
for year_label in FILES.keys():
    sub = df_flot[df_flot["Année"] == year_label]
    if len(sub) == 0:
        ERROR(f"Flottant: Aucune donnée pour {year_label}")
    else:
        n_valeurs = sub["Valeur"].nunique()
        OK(f"Flottant {year_label}: {n_valeurs} valeurs")
        
        # Vérifier que les flottants sont dans [0, 1]
        flottants = sub["Flottant"].values
        if any(f < 0 or f > 1 for f in flottants if not np.isnan(f)):
            WARN(f"Flottant {year_label}: certains flottants hors [0, 1]")
        
        # Vérifier que les plafonnements sont dans [0, 1]
        plafs = sub["Plafonnement"].values
        if any(p < 0 or p > 1 for p in plafs if not np.isnan(p)):
            WARN(f"Plafonnement {year_label}: certains plafonnements hors [0, 1]")

# ============================================================================
# BILAN FINAL
# ============================================================================
print("\n" + "=" * 70)
print("BILAN FINAL DE L'AUDIT")
print("=" * 70)
print(f"\n  ❌ ERREURS: {len(errors_found)}")
for e in errors_found:
    print(f"    {e}")
print(f"\n  ⚠️ AVERTISSEMENTS: {len(warnings_found)}")
for w in warnings_found:
    print(f"    {w}")

if len(errors_found) == 0:
    print("\n  🟢 AUCUNE ERREUR CRITIQUE TROUVÉE — LES RÉSULTATS SONT FIABLES")
else:
    print(f"\n  🔴 {len(errors_found)} ERREUR(S) CRITIQUE(S) TROUVÉE(S)")
