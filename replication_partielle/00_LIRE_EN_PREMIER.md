# Livrable code - Replication partielle du MASI 20

Ce livrable a ete organise pour permettre une lecture rapide et claire du projet.

## 1. Objet du dossier

Ce dossier contient les scripts utilises pour :

- la replication partielle du MASI 20 ;
- le backtesting out-of-sample ;
- les variantes d'optimisation du tracking error ;
- les analyses de robustesse et de stress ;
- la generation des rapports finaux.

## 2. Lecture conseillee

Pour comprendre le projet rapidement, il est recommande de lire les fichiers dans cet ordre :

1. `01_Moteur_Principal_Replication_Backtesting/masi20_replication_v2.py`
   Script principal du projet. Il contient le moteur de backtesting strict out-of-sample, la selection des titres, l'optimisation des poids et l'evaluation finale.
2. `02_Optimisation_Et_Robustesse/masi20_extreme_te.py`
   Version de recherche exhaustive pour les portefeuilles tres concentres.
3. `02_Optimisation_Et_Robustesse/masi20_advanced_te.py`
   Version avancee de reduction du tracking error.


## 3. Structure du livrable

### `01_Moteur_Principal_Replication_Backtesting`

Bloc central du projet : moteur de replication et de backtesting.

### `02_Optimisation_Et_Robustesse`

Scripts d'amelioration, d'audit, de fusion de resultats, de turnover et d'etude de la robustesse.

### `03_Analyses_Stress`

Scripts dedies a l'analyse de la periode de stress 2024-2025 et aux diagnostics associes.



## 4. Fichiers les plus importants

- `masi20_replication_v2.py` : script coeur du projet.
- `masi20_replication_remedy.py` : variante corrective du moteur principal.
- `masi20_replication.py` : version initiale du moteur.
- `masi20_extreme_te.py` : recherche exhaustive sur les combinaisons.
- `masi20_advanced_te.py` : optimisation avancee du tracking error.


## 5. Donnees et sorties

Les scripts s'appuient sur les donnees historiques du projet, stockees dans le dossier source :

- `02_Donnees_Historiques`

Les sorties intermediaires et rapports sont generalement ecrits dans :

- `05_Archives_Et_Intermediaires`

## 6. Remarque technique

Plusieurs scripts contiennent des chemins absolus historiques (`BASE_PATH`) correspondant a l'environnement de travail du projet. Si les scripts doivent etre relances sur une autre machine, ces chemins devront etre ajustes.
