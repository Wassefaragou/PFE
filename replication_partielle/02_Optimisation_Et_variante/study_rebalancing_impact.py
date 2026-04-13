import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE_PATH = r"c:\Users\PC\Downloads\Stage\répliaction_partielle"

FILES = {
    "2020-2021": "20-21.xlsx",
    "2021-2022": "21-22.xlsx",
    "2022-2023": "22-23.xlsx",
    "2023-2024": "23-24.xlsx",
    "2024-2025": "24-25.xlsx",
    "2025-2026": "25-26-SANSANOMALIE.xlsx",
}

OUTPUT_FILE = os.path.join(BASE_PATH, "06_Analyse_Stress_2024_2025", "Sensibilite_Periode_Rebalancement.xlsx")

# --- PARAMÈTRES DE L'ÉTUDE ---
K_VALUES = [7, 8, 9, 10]
# 1j, 5j (1 sem), 10j (2 sem), 15j (3 sem), 21j (~1 mois), 31j (~1.5 mois), 42j (~2 mois), 63j (~3 mois)
REBAL_FREQUENCIES = [1, 5, 10, 15, 21, 31, 42, 63]
LOOKBACK = 126

def load_all_data():
    yearly_data = {}
    for year_label, filename in FILES.items():
        path = os.path.join(BASE_PATH, "02_Donnees_Historiques", filename)
        if not os.path.exists(path):
             path = os.path.join(BASE_PATH, filename)
             
        df = pd.read_excel(path, header=10)
        df = df.dropna(axis=1, how="all")
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        index_col = [c for c in df.columns if "MSEMSI20" in str(c)][0]
        stock_cols = [c for c in df.columns if c != "Date" and c != index_col]

        for col in [index_col] + stock_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        rets_df = df[[index_col] + stock_cols]
        rets = np.log(rets_df / rets_df.shift(1)).iloc[1:]
        rets = rets.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-0.10, 0.10)

        yearly_data[year_label] = {
            "r_index": rets[index_col].values,
            "r_stocks": rets[stock_cols].values,
            "stock_cols": stock_cols,
            "mcap_matrix": df[stock_cols].values[1:],
        }
    return yearly_data

# On fixe la méthode de sélection à "Lasso" (ou n'importe quelle heuristique rapide stable)
# pour isoler uniquement l'effet de Rebal_jours, et on optimise en Shrinkage.
from sklearn.linear_model import Lasso

def select_lasso(r_index, r_stocks, k):
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
        corrs = np.nan_to_num(np.array([np.corrcoef(r_index, r_stocks[:, i])[0, 1] for i in range(r_stocks.shape[1])]), nan=-1)
        for i in np.argsort(corrs)[::-1]:
            if i not in best:
                best.append(i)
            if len(best) == k: break
    return best[:k]

def optim_min_te_shrinkage(r_index, r_sub):
    k = r_sub.shape[1]
    w0 = np.ones(k) / k
    T = r_sub.shape[0]

    if T < k + 2: return w0

    try:
        data = np.column_stack([r_sub, r_index])
        cov_shrunk = LedoitWolf().fit(data).covariance_
        Sigma_ss = cov_shrunk[:k, :k]
        sigma_si = cov_shrunk[:k, k]

        def obj(w): return w @ Sigma_ss @ w - 2 * w @ sigma_si
        def grad(w): return 2 * Sigma_ss @ w - 2 * sigma_si

        bounds = [(0, 1)] * k
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        res = minimize(obj, w0, jac=grad, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 500, "ftol": 1e-12})
        if res.success:
            w = np.maximum(res.x, 0)
            return w / w.sum() if w.sum() > 0 else w0
    except Exception:
        pass
    def obj_fallback(w): return np.std(r_sub @ w - r_index)
    res = minimize(obj_fallback, w0, method="SLSQP", bounds=bounds, constraints=cons)
    if res.success:
        w = np.maximum(res.x, 0)
        return w / w.sum() if w.sum() > 0 else w0
    return w0


def run_rebalance_study():
    print("=" * 60)
    print("ÉTUDE DE SENSIBILITÉ : IMPACT DE LA FRÉQUENCE DE REBALANCEMENT")
    print("=" * 60)
    
    yearly_data = load_all_data()
    all_results = []
    
    t0_global = time.time()
    
    for k in K_VALUES:
        print(f"\n" + "="*40 + f"\n=> Test pour k = {k}\n" + "="*40)
        for rebal in REBAL_FREQUENCIES:
            print(f"\n=> Fréquence : {rebal} jours")
            
            for year_label, ydata in yearly_data.items():
                r_index = ydata["r_index"]
                r_stocks = ydata["r_stocks"]
                T = len(r_index)
                
                t = LOOKBACK
                all_diffs = []
                
                while t < T:
                    t0_w = max(0, t - LOOKBACK)
                    r_idx_past = r_index[t0_w:t]
                    r_stk_past = r_stocks[t0_w:t]
                    
                    # 1. Sélection (Lasso simple)
                    selected_idx = select_lasso(r_idx_past, r_stk_past, k)
                    
                    # 2. Optimisation
                    r_sub_past = r_stk_past[:, selected_idx]
                    w = optim_min_te_shrinkage(r_idx_past, r_sub_past)
                    
                    # 3. OOS
                    t_end = min(t + rebal, T)
                    r_idx_oos = r_index[t:t_end]
                    r_sub_oos = r_stocks[t:t_end, :][:, selected_idx]
                    
                    if len(r_idx_oos) == 0: break
                    
                    all_diffs.extend((r_sub_oos @ w - r_idx_oos).tolist())
                    t += rebal
                    
                if len(all_diffs) > 3:
                    te_oos = np.std(all_diffs)
                else:
                    te_oos = np.nan
                    
                all_results.append({
                    "k": k,
                    "Année": year_label,
                    "Fréquence (Jours)": rebal,
                    "TE_bps": te_oos * 10000,
                    "TE_%": te_oos * 100
                })
                
                print(f"     {year_label} -> TE: {te_oos*100:.3f}%")
            
    df_res = pd.DataFrame(all_results)
    
    # Agrégation par Fréquence et par k
    grp = df_res.groupby(["k", "Fréquence (Jours)"]).agg(
        TE_Moyen_pct=("TE_%", "mean"),
        TE_Moyen_bps=("TE_bps", "mean"),
        TE_Max_pct=("TE_%", "max") 
    ).reset_index()
    
    # Mapping texte
    def freq_label(j):
        if j == 1: return "1 jour (Quotidien)"
        elif j == 5: return "5 jours (1 sem)"
        elif j == 10: return "10 jours (2 sem)"
        elif j == 15: return "15 jours (3 sem)"
        elif j == 21: return "21 jours (~1 mois)"
        elif j == 31: return "31 jours (~1.5 mois)"
        elif j == 42: return "42 jours (~2 mois)"
        elif j == 63: return "63 jours (~3 mois)"
        return f"{j} jours"
        
    grp["Période"] = grp["Fréquence (Jours)"].apply(freq_label)
    
    # Export du fichier
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter') as w:
        grp.to_excel(w, sheet_name="Synthèse", index=False, columns=["k", "Fréquence (Jours)", "Période", "TE_Moyen_pct", "TE_Moyen_bps", "TE_Max_pct"])
        df_res.to_excel(w, sheet_name="Détail_Années", index=False)
        
        # Generation Matrice Pivot spéciale Chart
        pivot_chart = grp.pivot(index="Période", columns="k", values="TE_Moyen_pct").reindex([freq_label(j) for j in REBAL_FREQUENCIES])
        pivot_chart.to_excel(w, sheet_name="Chart_Data")
        
        # Création du graphe Multi-Lignes
        wb = w.book
        ws_chart = w.sheets["Synthèse"]
        chart = wb.add_chart({'type': 'line'})
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for col_num, k_val in enumerate(K_VALUES, start=1):
            chart.add_series({
                'name': f'k={k_val}',
                'categories': ['Chart_Data', 1, 0, len(REBAL_FREQUENCIES), 0],
                'values':     ['Chart_Data', 1, col_num, len(REBAL_FREQUENCIES), col_num],
                'marker': {'type': 'circle'},
                'line': {'color': colors[col_num-1 % len(colors)], 'width': 2.5}
            })
            
        chart.set_title({'name': "Impact Fréquence Rebalancement sur le TE (k=7 à 10)"})
        chart.set_x_axis({'name': 'Période (Labels)'})
        chart.set_y_axis({'name': 'Tracking Error Moyen (%)'})
        chart.set_style(10)
        
        ws_chart.insert_chart('H2', chart, {'x_scale': 1.6, 'y_scale': 1.3})
        
    print(f"\n✅ EXPORT ET GRAPHIQUE GÉNÉRÉS : {OUTPUT_FILE}")
    print(pivot_chart)

if __name__ == "__main__":
    run_rebalance_study()
