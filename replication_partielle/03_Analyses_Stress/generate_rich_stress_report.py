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
OUT_DIR = os.path.join(BASE_PATH, "06_Analyse_Stress_2024_2025")

FILES = {
    "2020-2021": "20-21.xlsx",
    "2021-2022": "21-22.xlsx",
    "2022-2023": "22-23.xlsx",
    "2023-2024": "23-24.xlsx",
    "2024-2025": "24-25.xlsx",
    "2025-2026": "25-26-SANSANOMALIE.xlsx",
}

print("1. Chargement et Nettoyage des Données...")
all_dfs = []
annual_data = {}

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
    df = df.dropna(subset=["Date"])
    
    df = df[["Date", index_col] + stock_cols]
    df = df.rename(columns={index_col: "MASI20"})
    
    rets_df = df.set_index("Date")
    # Forward fill prices to avoid empty returns crashing the standard deviation
    rets_df = rets_df.fillna(method="ffill").fillna(method="bfill")
    rets = np.log(rets_df / rets_df.shift(1)).iloc[1:]
    rets = rets.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-0.10, 0.10)
    
    # Store per year
    annual_data[year_label] = rets
    all_dfs.append(rets)

df_total = pd.concat(all_dfs)
df_total = df_total.sort_index()
df_total = df_total[~df_total.index.duplicated(keep='last')]
df_total = df_total.loc[df_total.index.notna()]

# Remove any days where literally everything is exactly 0.0 (market closed but data row exists)
is_all_zero = (df_total == 0).all(axis=1)
df_total = df_total[~is_all_zero]

stock_cols = [c for c in df_total.columns if c != "MASI20"]
r_stk = df_total[stock_cols]
r_idx = df_total["MASI20"]

print("2. Calcul des Métriques Avancées (Rolling et Annuelles)...")
WINDOW = 126

# Rolling Correlation
rolling_corr = []
for i in range(len(r_stk)):
    if i < WINDOW:
        rolling_corr.append(np.nan)
        continue
    window_data = r_stk.iloc[i-WINDOW:i]
    corr_matrix = window_data.corr().values
    # Check for NaNs in corr_matrix (happens if a stock variance is 0 over 126 days)
    corr_vals = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
    # Use nanmean to ignore NaNs gracefully! (This fixes the empty chart bug)
    mean_corr = np.nanmean(corr_vals)
    rolling_corr.append(mean_corr)

df_total["Rolling_Corr_126j"] = rolling_corr

# Rolling Dispersion (Daily cross-sectional std, then smoothed)
daily_dispersion = r_stk.std(axis=1) * np.sqrt(252) * 100 
df_total["Dispersion_Jour_Annualisee"] = daily_dispersion
df_total["Rolling_Dispersion_126j"] = daily_dispersion.rolling(window=WINDOW).mean()

# Rolling Volatility Index
df_total["Rolling_Vol_MASI20_126j"] = r_idx.rolling(window=WINDOW).std() * np.sqrt(252) * 100


# --- STATISTIQUES ANNUELLES RICHES ---
stats_list = []
for year, data in annual_data.items():
    r_s = data.drop(columns=["MASI20"])
    r_i = data["MASI20"]
    
    vol = np.std(r_i) * np.sqrt(252) * 100
    disp = np.mean(np.std(r_s, axis=1)) * np.sqrt(252) * 100
    
    corr_m = r_s.corr().values
    mean_c = np.nanmean(corr_m[np.triu_indices_from(corr_m, k=1)])
    
    max_drawdown = (data["MASI20"].cumsum().cummax() - data["MASI20"].cumsum()).max() * 100
    
    ext_moves = np.sum((r_s.values > 0.015) | (r_s.values < -0.015)) / (r_s.shape[0] * r_s.shape[1]) * 100
    
    stats_list.append({
        "Année": year,
        "Jours Bourse": len(data),
        "Volatilité Indice (%)": vol,
        "Dispersion Transversale Moyenne (%)": disp,
        "Corrélation Moyenne": mean_c,
        "Pire Drawdown Indice (%)": max_drawdown,
        "Fréquence Chocs Jours (>1.5%)": ext_moves
    })

df_stats = pd.DataFrame(stats_list)

print("3. Exportation vers le Fichier Excel Riche...")
excel_path = os.path.join(OUT_DIR, "Master_Stress_Test_Analytics.xlsx")
with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
    # Feuille 1 : Synthèse Annuelle
    df_stats.to_excel(writer, sheet_name="Synthese_Annuelle_Risque", index=False)
    
    # Feuille 2 : Données Journalières et Rolling
    df_total.reset_index().to_excel(writer, sheet_name="Historique_Rolling_Data", index=False)
    
    # Feuille 3 : Matrices de Corrélation comparées
    corr_2021 = annual_data["2021-2022"].drop(columns=["MASI20"]).corr()
    corr_2024 = annual_data["2024-2025"].drop(columns=["MASI20"]).corr()
    
    corr_2021.to_excel(writer, sheet_name="Matrice_Corr_Calme_2021")
    corr_2024.to_excel(writer, sheet_name="Matrice_Corr_Stress_2024")
    
    # Formatage de l'Excel
    workbook = writer.book
    pct_format = workbook.add_format({'num_format': '0.00%'})
    num_format = workbook.add_format({'num_format': '0.000'})
    
    ws1 = writer.sheets["Synthese_Annuelle_Risque"]
    ws1.set_column("B:B", 15)
    ws1.set_column("C:G", 25, num_format)
    ws1.conditional_format('G2:G10', {'type': '3_color_scale'})
    ws1.conditional_format('E2:E10', {'type': '3_color_scale'})

print("4. Génération des Graphiques Trés Haute Qualité...")

stress_start = pd.to_datetime("2024-05-01")
stress_end = pd.to_datetime("2025-05-01")

# FIGURE 1 : TIMELINE CORRIGÉE (Corr + Disp + Vol)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 16), sharex=True)
sns.set_theme(style="whitegrid", context="talk")

# Plot 1: Correlation (Fixing the NaN bug)
curr_df = df_total.dropna(subset=["Rolling_Corr_126j"])
ax1.plot(curr_df.index, curr_df["Rolling_Corr_126j"], color="royalblue", lw=2.5)
ax1.axhline(curr_df["Rolling_Corr_126j"].mean(), color="grey", ls="--")
ax1.axvspan(stress_start, stress_end, color="red", alpha=0.1)
ax1.set_title("1. L'Effondrement de la Corrélation (Le Bruit augmente)", fontweight="bold")
ax1.set_ylabel("Corrélation Moyenne")

# Plot 2: Dispersion
ax2.plot(curr_df.index, curr_df["Rolling_Dispersion_126j"], color="crimson", lw=2.5)
ax2.axvspan(stress_start, stress_end, color="red", alpha=0.1)
ax2.axhline(curr_df["Rolling_Dispersion_126j"].mean(), color="grey", ls="--")
ax2.set_title("2. L'Explosion de la Dispersion (Les Titres Divergent)", fontweight="bold")
ax2.set_ylabel("Dispersion Annualisée (%)")

# Plot 3: Volatilité
ax3.plot(curr_df.index, curr_df["Rolling_Vol_MASI20_126j"], color="darkorange", lw=2.5)
ax3.axvspan(stress_start, stress_end, color="red", alpha=0.1)
ax3.axhline(curr_df["Rolling_Vol_MASI20_126j"].mean(), color="grey", ls="--")
ax3.set_title("3. Volatilité du MASI 20 (Risque Global)", fontweight="bold")
ax3.set_ylabel("Volatilité (%)")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Fig1_Timeline_Triple_HD.png"), dpi=300)
plt.close()

# FIGURE 2 : BOITE A MOUSTACHES DE LA DISPERSION (Violin Plot)
# On prépare les données
df_total["Annee_Civile"] = df_total.index.year
plt.figure(figsize=(14, 8))
sns.violinplot(data=df_total, x="Annee_Civile", y="Dispersion_Jour_Annualisee", palette="coolwarm", inner="quartile")
plt.title("Distribution de la Dispersion Quotidienne par Année", fontsize=16, fontweight="bold")
plt.ylabel("Dispersion Transversale Journalière (%)")
plt.xlabel("Année")
plt.savefig(os.path.join(OUT_DIR, "Fig2_Violin_Dispersion_Annuelle.png"), dpi=300)
plt.close()

# FIGURE 3 : HEATMAPS CORRELATION (2021 vs 2024)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

sns.heatmap(corr_2021, cmap="RdYlGn", vmin=-0.2, vmax=0.8, ax=ax1, cbar=False, xticklabels=False, yticklabels=False)
ax1.set_title("Corrélation en 2021 (Régime Normal)\nLes grands blocs verts montrent la cohésion", fontsize=14, fontweight="bold")

sns.heatmap(corr_2024, cmap="RdYlGn", vmin=-0.2, vmax=0.8, ax=ax2, cbar=True, xticklabels=False, yticklabels=False)
ax2.set_title("Corrélation en 2024 (Stress Test)\nLes blocs disparaissent, tout devient jaune/rouge (bruit)", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Fig3_Heatmap_Correlation_Comparaison.png"), dpi=300)
plt.close()

print("✅ Terminé. Excel riche et graphiques Premium générés.")
