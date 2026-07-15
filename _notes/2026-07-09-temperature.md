---
title: "Ce que la température ne change pas"
date: 2026-07-09
topics: [inférence, échantillonnage]
status: note
---

La température agit sur la distribution de sortie, pas sur le calcul interne déjà accompli. À température nulle, le modèle ne devient ni plus rationnel ni déterministe au sens fort : il sélectionne simplement le token de plus forte probabilité à chaque étape, sous réserve des détails d’implémentation.

Les activations, les logits et leurs écarts restent les meilleurs objets d’observation. La température révèle ou masque une hésitation ; elle ne la crée pas.
