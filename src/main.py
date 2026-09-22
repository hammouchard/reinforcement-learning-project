import numpy as np
import pandas as pd


# ============================================================
# 1. PARAMÈTRES DU LABYRINTHE
# ============================================================

# Numérotation des états :
#
# 12   13   14   15
#  8    9   10   11
#  4    5    6    7
#  0    1    2    3

nombre_etats = 16
nombre_actions = 4

etat_depart = 0
etat_objectif = 15

murs = [5, 14]
feux = [7, 9]


# ============================================================
# 2. ACTIONS
# ============================================================

HAUT = 0
BAS = 1
GAUCHE = 2
DROITE = 3


# ============================================================
# 3. TABLE Q
# ============================================================

# Une ligne par état et une colonne par action.
# Toutes les Q-values sont initialisées à 0.

table_Q = np.zeros((nombre_etats, nombre_actions))


# ============================================================
# 4. ACTIONS POSSIBLES
# ============================================================

def actions_possibles(etat):

    actions = []

    # L'état objectif est terminal : aucune action ensuite.
    if etat == etat_objectif:
        return actions

    # Monter : +4
    if etat <= 11 and etat + 4 not in murs:
        actions.append(HAUT)

    # Descendre : -4
    if etat >= 4 and etat - 4 not in murs:
        actions.append(BAS)

    # Aller à gauche : -1
    if ((1 <= etat <= 3) or (5 <= etat <= 7) or (9 <= etat <= 11) or (13 <= etat <= 15)) and etat - 1 not in murs:
        actions.append(GAUCHE)

    # Aller à droite : +1
    if ((0 <= etat <= 2) or (4 <= etat <= 6) or (8 <= etat <= 10) or (12 <= etat <= 14)) and etat + 1 not in murs:
        actions.append(DROITE)

    return actions


# ============================================================
# 5. DÉPLACEMENT DU ROBOT
# ============================================================

def deplacer(etat, action):

    # Si l'action demandée est impossible, le robot ne bouge pas.
    if action not in actions_possibles(etat):
        return etat

    if action == HAUT:
        return etat + 4

    if action == BAS:
        return etat - 4

    if action == GAUCHE:
        return etat - 1

    if action == DROITE:
        return etat + 1


# ============================================================
# 6. RÉCOMPENSES
# ============================================================

def recompense(etat):

    # Objectif atteint
    if etat == etat_objectif:
        return 10

    # Case de feu
    if etat in feux:
        return -10

    # Case normale
    return -0.1


# ============================================================
# 7. PARAMÈTRES DU Q-LEARNING
# ============================================================

alpha = 0.1
gamma = 0.9
nombre_episodes = 1000


# ============================================================
# 8. APPRENTISSAGE
# ============================================================

for episode in range(nombre_episodes):

    # À chaque épisode, le robot repart de l'état 0.
    etat = etat_depart

    # L'épisode se termine lorsque l'état 15 est atteint.
    while etat != etat_objectif:

        # On récupère uniquement les actions réellement possibles.
        actions = actions_possibles(etat)

        # Pour l'instant, choix aléatoire parmi les actions possibles.
        action = np.random.choice(actions)

        # On effectue le déplacement.
        nouvel_etat = deplacer(etat, action)

        # On récupère la récompense.
        r = recompense(nouvel_etat)

        # Si on atteint l'objectif, il n'y a plus de récompense future.
        if nouvel_etat == etat_objectif:
            meilleure_valeur_future = 0
        else:
            actions_futures = actions_possibles(nouvel_etat)
            meilleure_valeur_future = np.max(table_Q[nouvel_etat, actions_futures])

        # Formule de mise à jour du Q-learning.
        table_Q[etat, action] = table_Q[etat, action] + alpha * (r + gamma * meilleure_valeur_future - table_Q[etat, action])

        # On continue depuis le nouvel état.
        etat = nouvel_etat


# ============================================================
# 9. AFFICHAGE PROPRE DE LA TABLE Q
# ============================================================

noms_actions = ["Haut", "Bas", "Gauche", "Droite"]
noms_etats = [f"s{i}" for i in range(nombre_etats)]

types_etats = []

for etat in range(nombre_etats):

    if etat == etat_depart:
        types_etats.append("Départ")

    elif etat == etat_objectif:
        types_etats.append("Objectif")

    elif etat in murs:
        types_etats.append("Mur")

    elif etat in feux:
        types_etats.append("Feu")

    else:
        types_etats.append("Normal")


table_affichage = pd.DataFrame(np.round(table_Q, 2), index=noms_etats, columns=noms_actions)
table_affichage.insert(0, "Type", types_etats)

print("\nTable des Q-values :")
print(table_affichage.to_string())