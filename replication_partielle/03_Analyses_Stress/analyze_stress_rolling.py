import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

BASE_PATH = r"c:\Users\PC\Downloads\Stage\répliaction_partielle"
DATA_DIR = os.path.join(BASE_PATH, "02_Donnees_Historiques")

FILES = {
    "2020-2021": "20-21.xlsx",
    "2021-2022": "21-22.xlsx",
    "2022-2023": "22-23.xlsx",
    "2023-2024": "23-24.xlsx",
    "2024-2025": "24-25.xlsx",
    "2025-2026": "25-26-SANSANOMALIE.xlsx",
}

# 1. Charger et concaténer toutes les données pour avoir une série temporelle continue
print("Chargement des données historiques continues...")
all_dfs = []

for year_label, filename in FILES.items():
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_excel(path, header=10)
    df = df.dropna(axis=1, how="all")
    
    index_col = [c for c in df.columns if "MSEMSI20" in str(c)]
    index_col = index_col[0] if index_col else df.columns[1]
    stock_cols = [c for c in df.columns if c != "Date" and c != index_col]
    
    for col in [index_col] + stock_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Ne garder que Date, Index et les actions
    df = df[["Date", index_col] + stock_cols]
    df = df.rename(columns={index_col: "MASI20"})
    
    # Calcul des rendements journaliers (log returns)
    rets_df = df.set_index("Date")
    rets = np.log(rets_df / rets_df.shift(1)).iloc[1:]
    rets = rets.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-0.10, 0.10)
    
    all_dfs.append(rets)

# Fusionner en une seule timeline
df_total = pd.concat(all_dfs)
df_total = df_total.sort_index()

# Supprimer les doublons de dates exactes s'il y a chevauchement entre fichiers
df_total = df_total[~df_total.index.duplicated(keep='last')]

stock_cols = [c for c in df_total.columns if c != "MASI20"]
r_stk = df_total[stock_cols]
r_idx = df_total["MASI20"]

# 2. Calcul des métriques glissantes (Rolling 126 jours = ~6 mois)
WINDOW = 126
print(f"Calcul des métriques dynamiques (Rolling = {WINDOW} jours)...")

# A. Rolling Average Correlation
rolling_corr = []
for i in range(WINDOW, len(r_stk)):
    window_data = r_stk.iloc[i-WINDOW:i]
    corr_matrix = window_data.corr().values
    # Extraire le triangle supérieur hors diagonale
    mean_corr = corr_matrix[np.triu_indices_from(corr_matrix, k=1)].mean()
    rolling_corr.append(mean_corr)

# Remplir le début avec des NaNs
rolling_corr = [np.nan] * WINDOW + rolling_corr
df_total["Rolling_Corr"] = rolling_corr

# B. Rolling Cross-Sectional Dispersion
# Ecart type des rendements des 20 actions chaque jour, lissé sur 126 jours
daily_dispersion = r_stk.std(axis=1) * np.sqrt(252) * 100 # Dispersion quotidienne annualisée
df_total["Rolling_Dispersion"] = daily_dispersion.rolling(window=WINDOW).mean()

# 3. Génération des graphiques Rich Analytics
print("Création des visualisations dynamiques...")
sns.set_theme(style="darkgrid", context="talk")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

# Zone de Stress 2024-2025 à surligner
stress_start = pd.to_datetime("2024-05-01")
stress_end = pd.to_datetime("2025-05-01")

# Graphe 1 : L'effondrement de la corrélation
ax1.plot(df_total.index, df_total["Rolling_Corr"], color="darkorange", linewidth=2.5)
ax1.set_title(f"L'Effondrement des Corrélations (Moyenne glissante {WINDOW} jours)", fontsize=18, fontweight="bold", pad=20)
ax1.set_ylabel("Corrélation Moyenne MASI 20", fontsize=14)
ax1.axvspan(stress_start, stress_end, color='red', alpha=0.15, label="Période de Stress (2024-2025)")
ax1.axhline(y=df_total["Rolling_Corr"].mean(), color='grey', linestyle='--', alpha=0.7, label="Moyenne Historique (6 ans)")
ax1.legend(loc="upper right")

# Graphe 2 : L'explosion de la dispersion
ax2.plot(df_total.index, df_total["Rolling_Dispersion"], color="darkred", linewidth=2.5)
ax2.set_title(f"L'Explosion de la Dispersion Transversale (Chaos Inter-Actions)", fontsize=18, fontweight="bold", pad=20)
ax2.set_ylabel("Dispersion Annualisée (%)", fontsize=14)
ax2.set_xlabel("Années", fontsize=14)
ax2.axvspan(stress_start, stress_end, color='red', alpha=0.15, label="Zone de Rupture ($k=6$ échoue)")
ax2.axhline(y=df_total["Rolling_Dispersion"].mean(), color='grey', linestyle='--', alpha=0.7)
ax2.legend(loc="upper left")

# Annotations percutantes
min_corr_date = df_total["Rolling_Corr"].idxmin()
min_corr_val = df_total["Rolling_Corr"].min()
ax1.annotate(f'Plancher Historique: {min_corr_val:.2f}\nPerte du "Facteur Marché"', 
             xy=(min_corr_date, min_corr_val), xytext=(min_corr_date - pd.DateOffset(months=8), min_corr_val + 0.05),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             fontsize=12, fontweight='bold', color='darkred',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkred", lw=2))

max_disp_date = df_total["Rolling_Dispersion"].idxmax()
max_disp_val = df_total["Rolling_Dispersion"].max()
ax2.annotate(f'Pic Absolu: {max_disp_val:.1f}%\nLes actions divergent', 
             xy=(max_disp_date, max_disp_val), xytext=(max_disp_date - pd.DateOffset(months=10), max_disp_val - 2),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             fontsize=12, fontweight='bold', color='darkred',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkred", lw=2))

plt.tight_layout()
out_dir = os.path.join(BASE_PATH, "06_Analyse_Stress_2024_2025")
plt.savefig(os.path.join(out_dir, "Timeline_Stress_Test_HD.png"), dpi=300)
print(f"✅ Analyse temporelle riche sauvegardée dans : {os.path.join(out_dir, 'Timeline_Stress_Test_HD.png')}")
