import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from masi20_advanced_te import load_all_data, run_selection, OPT_FUNCS

BASE_PATH = r"c:\Users\PC\Downloads\Stage\répliaction_partielle"

# Configuration optimale identifiée
CFG_K = 10
CFG_SEL = "lasso"
CFG_OPT = "min_te"
CFG_REBAL = 5
CFG_LB = "rolling_126"

def run_best_config():
    print(f"--- Évaluation de la meilleure configuration ---")
    print(f"k={CFG_K}, Selection={CFG_SEL}, Optim={CFG_OPT}, Rebal={CFG_REBAL}j, Lookback={CFG_LB}")
    
    yearly_data, df_flottant = load_all_data()
    
    all_dates = []
    all_r_idx = []
    all_r_port = []

    for year_label, ydata in yearly_data.items():
        r_index = ydata["r_index"]
        r_stocks = ydata["r_stocks"]
        stock_cols = ydata["stock_cols"]
        mcap_matrix = ydata["mcap_matrix"]
        
        # Récupération des dates (on doit les re-charger depuis le fichier car non stockées dans yearly_data)
        path = os.path.join(BASE_PATH, f"{year_label[2:4]}-{year_label[7:9]}.xlsx")
        if year_label == "2025-2026":
            path = os.path.join(BASE_PATH, "25-26-SANSANOMALIE.xlsx")
            
        df_raw = pd.read_excel(path, header=10)
        df_raw = df_raw.dropna(axis=1, how="all")
        df_raw["Date"] = pd.to_datetime(df_raw["Date"])
        df_raw = df_raw.sort_values("Date").reset_index(drop=True)
        dates = df_raw["Date"].iloc[1:].values  # skip first row for returns

        T = len(r_index)
        
        # Flottant
        df_fl = df_flottant[df_flottant["Année"] == year_label]
        flot_dict = dict(zip(df_fl["Valeur"], df_fl["Flottant"]))
        plaf_dict = dict(zip(df_fl["Valeur"], df_fl["Plafonnement"]))

        lb_size = 126
        if lb_size >= T:
            lb_size = T // 2
            
        t = lb_size
        
        while t < T:
            t0_w = max(0, t - lb_size)
            r_idx_past = r_index[t0_w:t]
            r_stk_past = r_stocks[t0_w:t]
            mcap_past = mcap_matrix[t0_w:t]
            mean_mcap_past = mcap_past.mean(axis=0)

            # 1. Sélection
            selected_idx = run_selection(CFG_SEL, r_idx_past, r_stk_past, mean_mcap_past, CFG_K)

            sel_names = [stock_cols[i] for i in selected_idx]
            caps = np.array([plaf_dict.get(s, 1.0) for s in sel_names])
            flotts = np.array([flot_dict.get(s, 1.0) for s in sel_names])
            mcaps_sel = mean_mcap_past[selected_idx]
            mcaps_float = mcaps_sel * flotts

            # 2. Optimisation
            r_sub_past = r_stk_past[:, selected_idx]
            w = OPT_FUNCS[CFG_OPT](r_idx_past, r_sub_past, caps=caps, mcaps=mcaps_float)

            # 3. OOS
            t_end = min(t + CFG_REBAL, T)
            r_idx_oos = r_index[t:t_end]
            r_sub_oos = r_stocks[t:t_end, :][:, selected_idx]
            dates_oos = dates[t:t_end]

            if len(r_idx_oos) == 0:
                break

            r_port = r_sub_oos @ w
            
            all_r_idx.extend(r_idx_oos.tolist())
            all_r_port.extend(r_port.tolist())
            all_dates.extend(dates_oos.tolist())
            
            t += CFG_REBAL

    df_res = pd.DataFrame({
        "Date": all_dates,
        "R_Index": all_r_idx,
        "R_Port": all_r_port
    })
    
    # TE exact recalculé sur la série complète vs par an
    te_total = np.std(df_res["R_Port"] - df_res["R_Index"])
    print(f"\nTE Total recalculé (toutes années concaténées) : {te_total:.6f} ({te_total*100:.3f}%)")
    
    # Rendements cumulés (prix base 100)
    # log returns -> exp(cumsum)
    df_res["Cum_Index"] = 100 * np.exp(np.cumsum(df_res["R_Index"]))
    df_res["Cum_Port"]  = 100 * np.exp(np.cumsum(df_res["R_Port"]))
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df_res["Date"], df_res["Cum_Index"], label="Index MASI 20", color="blue", linewidth=2)
    plt.plot(df_res["Date"], df_res["Cum_Port"], label=f"Portefeuille Répliqué (k={CFG_K})", color="orange", linewidth=2, linestyle='--')
    
    plt.title(f"Performance Cumulée : Portefeuille Optimal vs MASI 20\n(TE = {te_total*100:.3f}%, {CFG_SEL}, {CFG_OPT}, Rebal: {CFG_REBAL}j, LB: {CFG_LB})", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Valeur (Base 100)", fontsize=12)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out_img = os.path.join(BASE_PATH, "performance_optimale_advanced_te.png")
    plt.savefig(out_img, dpi=300)
    print(f"Graphique sauvegardé : {out_img}")

if __name__ == "__main__":
    run_best_config()
