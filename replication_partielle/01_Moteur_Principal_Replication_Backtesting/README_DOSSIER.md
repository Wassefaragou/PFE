# Dossier 01 - Moteur principal

Ce dossier contient le coeur quantitatif du projet.

## Fichiers

- `masi20_replication_v2.py` : version principale et la plus aboutie du moteur de replication/backtesting.
- `masi20_replication_remedy.py` : version corrective intermediaire.
- `masi20_replication.py` : version initiale du moteur.

## Role de ce bloc

Ces scripts realisent les etapes principales du projet :

- chargement des donnees historiques ;
- calcul des rendements ;
- selection des actions ;
- optimisation des poids ;
- backtesting out-of-sample ;
- calcul du tracking error.

## Fichier a lire en priorite

Lire d'abord `masi20_replication_v2.py`.
