import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

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

stats = []

for year_label, filename in FILES.items():
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_excel(path, header=10)
    df = df.dropna(axis=1, how="all")
    
    index_col = [c for c in df.columns if "MSEMSI20" in str(c)]
    index_col = index_col[0] if index_col else df.columns[1]
    stock_cols = [c for c in df.columns if c != "Date" and c != index_col]
    
    for col in [index_col] + stock_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    rets_df = df[[index_col] + stock_cols]
    rets = np.log(rets_df / rets_df.shift(1)).iloc[1:]
    rets = rets.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-0.10, 0.10)
    
    r_idx = rets[index_col].values
    r_stk = rets[stock_cols].values
    
    # 1. Volatilité de l'Indice (Annualisée)
    vol_index = np.std(r_idx) * np.sqrt(252) * 100
    
    # 2. Dispersion Transversale (Cross-Sectional Volatility)
    # Mesure l'écart type des rendements des 20 actions chaque jour, puis on fait la moyenne sur l'année.
    # Plus c'est élevé, plus les actions partent dans tous les sens (c'est le chaos).
    cross_dispersion = np.mean(np.std(r_stk, axis=1)) * np.sqrt(252) * 100
    
    # 3. Corrélation Moyenne
    # Plus la corrélation est basse, moins l'indice a une direction "claire", rendant la réplication avec peu de titres difficile.
    corr_matrix = pd.DataFrame(r_stk).corr().values
    mean_corr = corr_matrix[np.triu_indices_from(corr_matrix, k=1)].mean()
    
    # 4. Pourcentage de jours avec des mouvements extrêmes ( > 1.5% ou < -1.5% sur une action)
    extreme_moves = np.sum((r_stk > 0.015) | (r_stk < -0.015)) / (r_stk.shape[0] * r_stk.shape[1]) * 100

    stats.append({
        "Année": year_label,
        "Volatilité Indice (%)": vol_index,
        "Dispersion Transversale (%)": cross_dispersion,
        "Corrélation Moyenne": mean_corr,
        "Mouvements Extrêmes (%)": extreme_moves
    })

df_stats = pd.DataFrame(stats)

# Sauvegarde des stats en Excel
out_dir = os.path.join(BASE_PATH, "06_Analyse_Stress_2024_2025")
df_stats.to_excel(os.path.join(out_dir, "Statistiques_Chaos_Marche.xlsx"), index=False)

# Création des graphiques
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Analyse du Régime de Marché (Pourquoi 2024-2025 est-il un Stress Test ?)", fontsize=16, fontweight='bold')

# 1. Volatilité
sns.barplot(data=df_stats, x="Année", y="Volatilité Indice (%)", ax=axes[0, 0], palette="Blues")
axes[0, 0].set_title("Volatilité de l'Indice MASI 20 (Risque Macro)")
axes[0, 0].tick_params(axis='x', rotation=45)
# Highlight 24-25
axes[0, 0].patches[4].set_facecolor('red')

# 2. Dispersion
sns.barplot(data=df_stats, x="Année", y="Dispersion Transversale (%)", ax=axes[0, 1], palette="Greens")
axes[0, 1].set_title("Dispersion Transversale (Chaos Inter-Actions)")
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].patches[4].set_facecolor('red')

# 3. Correlation
sns.barplot(data=df_stats, x="Année", y="Corrélation Moyenne", ax=axes[1, 0], palette="Oranges")
axes[1, 0].set_title("Corrélation Moyenne entre les 20 Actions (Cohésion)")
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].patches[4].set_facecolor('red')

# 4. Extreme
sns.barplot(data=df_stats, x="Année", y="Mouvements Extrêmes (%)", ax=axes[1, 1], palette="Purples")
axes[1, 1].set_title("Fréquence des mouvements extrêmes (>1.5% jour)")
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].patches[4].set_facecolor('red')

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.savefig(os.path.join(out_dir, "Graphique_Diagnostic_Stress.png"), dpi=300)
print(df_stats.to_markdown())
print("\nGraphiques et tableau sauvegardés avec succès.")
