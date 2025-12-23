---
layout: post
title: "II. Géométrie de l'alignement : variétés, directions et zones d'ombre"
categories: [théorie, introduction]
---

### **Prélude : de la dynamique du flux résiduel à la cartographie latente**

#### **Rappel : inertie du flux résiduel**

L’analyse précédente a décrit l’inférence comme l’évolution d’un état $x^{(l)} \in \mathbb{R}^{d_{\text{model}}}$ le long d’un **flux résiduel** persistant. La loi locale est une **accumulation additive** de mises à jour, sous contrainte de normalisation :

$$
x^{(l+1)} ;=; x^{(l)} + \Delta x^{(l)}*{\mathrm{MHA}} + \Delta x^{(l)}*{\mathrm{MLP}},
$$

avec, dans une instanciation *pre-norm* typique,

$$
\tilde{x}^{(l)} = x^{(l)} + \mathrm{MHA}(\mathrm{RMSNorm}(x^{(l)})), \qquad
x^{(l+1)} = \tilde{x}^{(l)} + \mathrm{MLP}(\mathrm{RMSNorm}(\tilde{x}^{(l)})).
$$

Trois vulnérabilités structurelles pour l’alignement et la robustesse en découlent :

* **Saturation directionnelle (invariance à l’échelle) :** la RMSNorm impose une quasi-invariance d’échelle et rend la dynamique effective principalement **angulaire** (projections). Un terme adversarial à forte magnitude peut dominer l’orientation post-normalisation et réduire la projection sur une direction d’alignement (p. ex. $v_{\text{refus}}$), sans modification des poids.

* **Monisme architectural (control/data confusion) :** absence de séparation matérielle entre *instructions* et *données* ; les mêmes opérateurs (attention, MLP) réécrivent l’ensemble du contexte. Les garde-fous d’instruction (p. ex. *System Prompt*) ne constituent pas une contrainte inviolable, mais un signal contextuel diluable par la composition et la longueur du contexte.

* **Interférence par superposition :** la contrainte $N \gg d_{\text{model}}$ force un encodage en features non orthogonales. Cette compression induit des corrélations résiduelles entre features disjointes ; des combinaisons de features bénignes peuvent, par addition vectorielle, produire une projection non négligeable sur des directions associées à des comportements interdits.

#### **Perspective : géométrie et topologie**

Le cadre précédent caractérise le “moteur” (mécanique des mises à jour $\Delta x$). Le présent article caractérise la “carte” : **structure globale de l’espace latent** et conditions de stabilité des signaux d’alignement. Trois axes organisent cette transition :

1. **Hypothèse linéaire et causalité :** extraction de directions latentes (features lisibles linéairement), puis validation causale par interventions sur activations (*ablation*, *clamping*, *steering*), afin de distinguer lecture corrélative et mécanisme effectif.

2. **Topologie hors-distribution (OOD) :** analyse de la dégradation des frontières de décision lorsque l’état latent est projeté hors de la variété induite par les données naturelles (manifold), vers des régions peu contraintes (suffixes adversariaux, *glitch tokens*), où la généralisation des signaux d’alignement n’est pas garantie.

3. **Résolution de la superposition :** introduction des autoencodeurs épars (SAE) comme outils de factorisation visant à accroître la séparabilité des features, condition nécessaire au passage d’un alignement “au niveau prompt” vers des interventions plus localisées dans l’espace des représentations.

***

### **Prélude : De la cinétique du flux résiduel à la topologie de l'espace latent**

#### **Rappel : Formalisation de l'inertie sémantique**

L’analyse précédente a modélisé l’inférence comme l’évolution d’un état $x^{(l)} \in \mathbb{R}^{d_{\text{model}}}$ le long d’un **flux résiduel** persistant. La dynamique locale est régie par une **accumulation additive** de mises à jour $\Delta x$, sous contrainte de normalisation. Dans une architecture *Pre-Norm*, cette propagation s'exprime par :

$$
x^{(l+1)} = x^{(l)} + \mathcal{F}_{\text{attn}}(\eta(x^{(l)})) + \mathcal{F}_{\text{mlp}}(\eta(x'^{(l)}))
$$

où $\eta$ désigne l'opérateur RMSNorm. Trois vulnérabilités structurelles, ou primitives d'exploitation, découlent de cette arithmétique :

* **Saturation vectorielle et invariance d'échelle :** L'opérateur $\eta$ projetant l'état sur une hypersphère de rayon fixe, l'information devient exclusivement directionnelle. Un vecteur adversarial de forte magnitude ($||v_{\text{adv}}|| \gg ||v_{\text{sécu}}||$) domine l'orientation post-normalisation, rendant la projection du vecteur de sécurité négligeable par écrasement angulaire.
* **Confusion Plan de Contrôle / Plan de Données :** L'architecture, de type Von Neumann, ne présente aucune ségrégation matérielle entre instructions (*System Prompts*) et données (*User Inputs*). Les mécanismes d'attention traitent ces signaux de manière indistincte ; les garde-fous ne sont pas des invariants logiques, mais des signaux contextuels diluables.
* **Interférences par superposition (Polysemanticité) :** La contrainte dimensionnelle ($N_{\text{features}} \gg d_{\text{model}}$) impose un encodage non-orthogonal (superposition). Des concepts disjoints partagent des sous-espaces communs, permettant à une combinaison linéaire de features bénignes de générer une interférence constructive alignée avec une direction interdite (*Crosstalk*).

#### **Perspective : Géométrie, Variétés et Ingénierie des Représentations**

Le cadre précédent ayant caractérisé la cinétique des mises à jour, la présente étude se concentre sur la **topologie globale de l’espace latent**. L'analyse transite du traitement du signal vers la structure géométrique des représentations sémantiques.

1.  **Hypothèse de la Représentation Linéaire (LRA) :** Validation de l'encodage des concepts (y compris moraux) sous forme de directions linéaires stables. Introduction des techniques de **sondage linéaire** (*Linear Probes*) et de **pilotage d'activation** (*Activation Steering/Clamping*) pour démontrer la causalité des directions identifiées sur le comportement du modèle.
2.  **Topologie Hors-Distribution (OOD) et Hypothèse des Variétés :** Analyse de l'effondrement des frontières de décision lorsque l'état latent est projeté hors de la variété des données naturelles (*Manifold*). Étude des zones de comportement indéfini exploitées par les attaques par suffixes adversariaux et *glitch tokens*.
3.  **Désenchevêtrement par Autoencodeurs Épars (SAE) :** Traitement du problème de la superposition via la décomposition des activations en composantes éparses (*Sparse Autoencoders*). Cette approche vise à isoler des features monosemantiques, condition nécessaire au passage d'un alignement probabiliste global à des interventions vectorielles localisées.


---

### **Prélude : De la cinétique du flux résiduel à la topologie de l’espace latent**

#### **Rappel : formalisation de l’inertie sémantique**

L’article précédent a modélisé l’inférence comme l’évolution d’un état $x^{(l)} \in \mathbb{R}^{d_{\text{model}}}$ le long d’un **flux résiduel** persistant. La dynamique locale s’écrit comme une accumulation additive de mises à jour, sous contrainte de normalisation. Dans une architecture *Pre-Norm* typique :

$$
\begin{aligned}
\tilde{x}^{(l)} &= x^{(l)} + \underbrace{\mathrm{MHA}(\eta(x^{(l)}))}_{\Delta x^{(l)}_{\text{attn}}} \\
x^{(l+1)} &= \tilde{x}^{(l)} + \underbrace{\mathrm{MLP}(\eta(\tilde{x}^{(l)}))}_{\Delta x^{(l)}_{\text{mlp}}}
\end{aligned}
$$

où $\eta$ désigne l’opérateur de normalisation (souvent RMSNorm). Sous une forme canonique :

$$
\eta(x) = g \odot \frac{x}{\mathrm{RMS}(x)+\varepsilon}, \quad \text{avec} \quad \mathrm{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2}
$$

Cette structure impose une quasi-invariance à l’échelle ($\eta(\alpha x) \approx \eta(x)$ pour $\alpha > 0$), ce qui rend la propagation *effectivement directionnelle* : la dynamique dépend principalement d’angles, de projections et de produits scalaires dans l’espace latent.

Trois vulnérabilités structurelles — interprétables comme des primitives de fragilité pour l’alignement — en découlent :

* **Saturation vectorielle et écrasement angulaire (invariance d’échelle).**
    Si l’état s’écrit $x = v_{\text{sécu}} + v_{\text{adv}} + \epsilon$ avec $\|v_{\text{adv}}\| \gg \|v_{\text{sécu}}\|$, alors $\eta(x)$ s’aligne asymptotiquement avec la direction de $v_{\text{adv}}$. La contribution relative de $v_{\text{sécu}}$ aux projections ultérieures devient négligeable. Le phénomène est un *écrasement angulaire* : la composante “sécurité” n’est pas annulée, mais cesse d’être géométriquement dominante.

* **Confusion plan de contrôle / plan de données.**
    Aucun mécanisme architectural ne sépare *instructions* et *données* : les mêmes opérateurs (Attention/MLP) réécrivent l’ensemble du contexte dans un canal unique. Les garde-fous sont donc des **signaux contextuels** (embeddings et traces dans $x^{(l)}$), et non des invariants logiques. La contrainte de sûreté dépend uniquement de sa persistance géométrique face aux mises à jour successives $\Delta x^{(l)}$.

* **Interférences par superposition (polysemanticité).**
    La contrainte dimensionnelle ($N_{\text{features}} \gg d_{\text{model}}$) impose un encodage non-orthogonal : plusieurs features partagent des sous-espaces communs. Si des features $f_i$ contribuent additivement au résiduel, une combinaison “bénigne” $\sum a_i f_i$ peut induire une projection non négligeable sur une direction interdite $u$ via :
    $$
    \langle u, \sum_i a_i f_i \rangle = \sum_i a_i \langle u, f_i \rangle
    $$
    Cela crée un *crosstalk* capable d'activer un concept malveillant sans composant explicitement toxique en entrée.

#### **Perspective : Géométrie, variétés et ingénierie des représentations**

Le cadre précédent a caractérisé la cinétique locale des mises à jour. La présente étude se concentre sur la **structure globale de l’espace latent** : stabilité des directions d’alignement, robustesse hors-distribution et factorisation des features. Trois axes structurent cette transition :

1.  **Hypothèse de représentation linéaire et test causal.**
    Les concepts sont traités comme des directions lisibles linéairement dans des sous-espaces d’activation. Les **sondes linéaires** (*Linear Probes*) fournissent une lecture ; les interventions sur activations (*ablation, clamping, steering*) fournissent le critère de causalité, en distinguant la simple corrélation de lecture du mécanisme effectif de génération.

2.  **Topologie hors-distribution (OOD) et hypothèse des variétés.**
    La sécurité est une contrainte apprise sur la variété induite par les distributions naturelles (*Manifold Hypothesis*). Lorsque l’état latent est projeté hors de cette variété, les frontières de décision (dont les frontières de refus) ne sont plus garanties. Nous analyserons ces zones de comportement indéfini, la fragilité des signaux d’alignement face aux entrées atypiques (suffixes adversariaux, *glitch tokens*) et l'effondrement des généralisations morales.

3.  **Désenchevêtrement par autoencodeurs épars (SAE).**
    La superposition limite toute “désactivation ciblée” : intervenir sur une direction polysémantique revient à altérer plusieurs fonctions simultanément. Les **Sparse Autoencoders** seront introduits comme outils de décomposition visant une représentation éparse et monosemantique des features, condition *sine qua non* pour passer d'un alignement probabiliste global à des interventions vectorielles chirurgicales.

---


#### **Perspective : Géométrie, variétés et ingénierie des représentations**

Le cadre précédent a décrit un mécanisme local : à chaque couche, un état résiduel reçoit une mise à jour additive, puis est renvoyé dans un régime essentiellement directionnel par la normalisation. La question centrale devient alors globale : **quelles structures stables** existent dans l’espace latent, et **quelles conditions** garantissent qu’un signal d’alignement (refus, honnêteté, conformité) reste lisible et dominant au fil des couches et des perturbations ?

Une première étape consiste à rendre l’hypothèse “les concepts sont des directions” opératoire. Il ne s’agit plus d’énoncer une correspondance intuitive entre “refus” et un vecteur, mais de **cartographier** des sous-espaces où un attribut est lisible par des fonctionnels simples. Les **sondes linéaires** fournissent alors un instrument de lecture : elles exhibent des directions $w$ telles que $\langle w, x^{(l)} \rangle$ sépare deux régimes (p. ex. refus vs conformité). Cette lecture est toutefois insuffisante en elle-même : la suite impose un critère plus fort, celui de la **causalité**. Une direction candidate n’est considérée mécanistiquement pertinente que si une intervention sur les activations (ablation, clamping, steering) induit un changement comportemental reproductible, c’est-à-dire si la direction ne se contente pas d’encoder un corrélat mais participe au circuit effectif de génération.

Cette exigence expérimentale se heurte immédiatement à une limite topologique : la stabilité d’un signal d’alignement n’est pas uniforme dans tout l’espace des états. Les modèles sont majoritairement entraînés sur des distributions linguistiques naturelles ; l’espace des activations correspondant peut être vu comme une variété (au sens large) de forte densité. Dès que l’état $x^{(l)}$ est poussé **hors de cette variété**, aucune raison n’impose que les frontières apprises — y compris les frontières de refus — conservent leur forme. Dans ces régions peu contraintes, l’extrapolation est arbitraire : des entrées atypiques (suffixes adversariaux, séquences non naturelles, *glitch tokens*) peuvent déplacer la trajectoire de $x^{(l)}$ vers des zones où les signaux d’alignement cessent d’être dominants, voire cessent d’être définis de manière cohérente. L’article analyse cette “zone grise” comme un objet géométrique : non pas une simple faiblesse de filtrage, mais un problème de généralisation hors-support.

Enfin, même dans le régime in-distribution, la cartographie et l’intervention restent limitées par un verrou structurel déjà identifié : la **superposition**. La contrainte $N_{\text{features}} \gg d_{\text{model}}$ implique un empaquetage non orthogonal des features ; une “direction” isolée par un probe peut agréger plusieurs facteurs latents. Dans ce contexte, une intervention naïve sur une direction polysémantique équivaut à modifier simultanément plusieurs fonctions. La dernière étape introduit les **Sparse Autoencoders (SAE)** comme tentative de factorisation : projeter les activations dans une base plus éparse, où les composantes sont davantage séparables, afin de transformer des manipulations grossières (au niveau prompt ou au niveau d’un petit nombre de directions entremêlées) en interventions plus localisées au niveau des représentations.
