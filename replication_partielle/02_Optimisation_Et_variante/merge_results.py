
import pandas as pd
import os
import sys

BASE_PATH = r"c:\Users\PC\Downloads\Stage\répliaction_partielle"
FILE_V2 = os.path.join(BASE_PATH, "resultats_replication_v2.xlsx")
FILE_REMEDY = os.path.join(BASE_PATH, "resultats_remedy.xlsx")
OUTPUT_MERGED = os.path.join(BASE_PATH, "resultats_replication_merged.xlsx")

def merge_results():
    print("Chargement des résultats V2...")
    try:
        df_v2 = pd.read_excel(FILE_V2, sheet_name="Brut") 
    except Exception as e:
        print(f"Erreur chargement V2: {e}")
        return

    print("Chargement des résultats Remedy...")
    try:
        df_remedy = pd.read_excel(FILE_REMEDY, sheet_name="Brut")
    except Exception as e:
        print(f"Erreur chargement Remedy: {e}")
        return

    print(f"V2: {df_v2.shape}, Remedy: {df_remedy.shape}")
    
    # Fusion
    df_final = pd.concat([df_v2, df_remedy], ignore_index=True)
    print(f"Total fusionné: {df_final.shape}")

    with pd.ExcelWriter(OUTPUT_MERGED) as writer:
        df_final.to_excel(writer, sheet_name="Brut", index=False)
    
    print(f"Fichier fusionné généré: {OUTPUT_MERGED}")

if __name__ == "__main__":
    merge_results()
