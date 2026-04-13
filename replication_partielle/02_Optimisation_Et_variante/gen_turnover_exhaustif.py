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

BASE_PATH = r"c:\Users\PC\Downloads\Stage\répliaction_partielle"

FILES = {
    "2020-2021": "20-21.xlsx",
    "2021-2022": "21-22.xlsx",
    "2022-2023": "22-23.xlsx",
    "2023-2024": "23-24.xlsx",
    "2024-2025": "24-25.xlsx",
    "2025-2026": "25-26-SANSANOMALIE.xlsx",
}

OUTPUT_FILE = os.path.join(BASE_PATH, "06_Analyse_Stress_2024_2025", "Historique_Total_Turnover_k8.xlsx")

def optim_min_te_shrinkage(r_index, r_sub):
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

    def obj_fallback(w): return np.std(r_sub @ w - r_index)
    res = minimize(obj_fallback, w0, method="SLSQP", bounds=bounds, constraints=cons)
    if res.success:
        w = np.maximum(res.x, 0)
        return w / w.sum() if w.sum() > 0 else w0
    return w0


def generate_full_history(k=8, rebal=5, lookback=126):
    print(f"LANCEMENT EXHAUSTIVE DEEP SEARCH (Turnover Tracking) pour k={k}")
    
    all_rebal_records = []
    
    # Pre-compute combinations to save time
    idx_combinations = list(combinations(range(20), k))
    print(f"Combinaisons par pas de temps : {len(idx_combinations):,}")

    t0_global = time.time()

    for year_label, filename in FILES.items():
        print(f"\nTraitement {year_label}...")
        
        # Check in multiple places to ensure we find it
        path = os.path.join(BASE_PATH, "02_Donnees_Historiques", filename)
        if not os.path.exists(path):
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
        
        # Array complet des dates pour mapper chaque étape
        dates_array = df['Date'].iloc[1:].dt.strftime('%d/%m/%Y').values

        r_index = rets[index_col].values
        r_stocks = rets[stock_cols].values
        
        T = len(r_index)
        t = lookback
        
        while t < T:
            t0_w = max(0, t - lookback)
            r_idx_past = r_index[t0_w:t]
            r_stk_past = r_stocks[t0_w:t]
            
            rebal_date = dates_array[t-1] # Le jour où l'on prend la décision

            # 1. BRUTE FORCE PROXY OLS
            results_proxy = []
            for c in idx_combinations:
                r_sub = r_stk_past[:, c]
                try:
                    w, _, _, _ = np.linalg.lstsq(r_sub, r_idx_past, rcond=None)
                    w = np.maximum(w, 0)
                    s_w = w.sum()
                    w = w / s_w if s_w > 0 else np.ones(k) / k
                    err = r_sub @ w - r_idx_past
                    results_proxy.append((np.sum(err ** 2), c))
                except Exception:
                    continue
            
            # Top 5
            results_proxy.sort(key=lambda x: x[0])
            top5_combos = [res[1] for res in results_proxy[:5]]
            
            # 2. DEEP OPTIMIZATION LEDOIT-WOLF
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
                    
            # 3. STOCKAGE DE CE PAS DE TEMPS
            # Préparer le vecteur complet des 20 poids
            full_w = [0.0] * 20
            # On mappe les k poids sur les bonnes colonnes
            for idx_w, stk_idx in enumerate(best_final_c):
                full_w[stk_idx] = best_final_w[idx_w]
                
            # Dictionary de base
            record = {
                "Année": year_label,
                "Date de Décision": rebal_date,
                "TE Estimé à l'instant t": best_final_te
            }
            # Ajout des 20 actions
            for i, stk in enumerate(stock_cols):
                record[stk] = full_w[i]
                
            all_rebal_records.append(record)
            
            t += rebal

    # Export
    print(f"\nCalcul terminé en {time.time()-t0_global:.1f} secondes. Création DataFrame...")
    df_export = pd.DataFrame(all_rebal_records)
    
    # Formatage de l'export Excel (Tableau Stylisé)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter') as writer:
        df_export.to_excel(writer, index=False, sheet_name="Matrice de Turnover (k=8)")
        wb = writer.book
        ws = writer.sheets["Matrice de Turnover (k=8)"]
        
        # Formats
        fmt_pct = wb.add_format({'num_format': '0.00%', 'align': 'center'})
        fmt_txt = wb.add_format({'align': 'center'})
        
        ws.set_column(0, 1, 15, fmt_txt)
        ws.set_column(2, 2, 18, wb.add_format({'num_format': '0.000'}))
        ws.set_column(3, len(df_export.columns)-1, 14, fmt_pct)
        
        ws.add_table(0, 0, len(df_export), len(df_export.columns)-1, {
            'columns': [{'header': c} for c in df_export.columns],
            'style': 'Table Style Medium 6'
        })
        
    print(f"✅ EXPORT RÉUSSI : {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_full_history(k=8, rebal=5, lookback=126)
