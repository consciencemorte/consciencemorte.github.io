---
layout: post
title: "I. Architecture Transformer et discrétisation du langage"
categories: [théorie, introduction]
featured: true
math: true
hero_image: "/assets/img/art1/header.png"
sf_cta: "lire"
sf_title: "Latence marginal"
sf_meta: "—  La rencontre"
sf_mode: details
sf_open: false

---

*Tokenisation - Flux résiduel - Composition fonctionnelle - Attention*

## 1.1 Tokenisation et discrétisation de l’espace d’entrée

L’interaction avec un Grand Modèle de Langage (LLM) se donne comme un flux textuel continu ; pourtant, le réseau ne manipule qu’une **séquence discrète d’entiers**. La *tokenisation*, première transformation du pipeline d’inférence, matérialise l’interface entre langage naturel (symbolique) et calcul matriciel (numérique). Cette interface constitue une **surface d’attaque structurelle** : elle est discontinue, dépend d’heuristiques d’implémentation (normalisation, pré-tokenisation), et se trouve souvent découplée des garde-fous périphériques (WAF, filtres, classifieurs externes).

### Formalisation du processus de tokenisation

Soit $\mathcal{S}$ l’ensemble des chaînes de caractères possibles. La tokenisation peut être modélisée comme une composition :

$$
T \;=\; \iota \circ \tau \circ \nu \;:\; \mathcal{S} \rightarrow \{0,\dots,|\mathcal{V}|-1\}^*
$$

- $\nu : \mathcal{S}\rightarrow \mathcal{S}$ — **normalisation / pré-tokenisation** : règles déterministes sur espaces, Unicode (NFC/NFKC ou autre selon implémentation), motifs de découpage, etc.
- $\tau : \mathcal{S}\rightarrow \mathcal{V}^*$ — **segmentation** en une séquence $(v_1,\dots,v_n)$, avec $v_k\in\mathcal{V}$.
- $\iota : \mathcal{V}\rightarrow \{0,\dots,|\mathcal{V}|-1\}$ — **indexation** associant à chaque token un entier $i_k$ (Token ID).

Le vocabulaire $\mathcal{V}$ est fini, de cardinalité $|\mathcal{V}|$ fixée avant l’entraînement (souvent $32\,000$ à $128\,000$ pour de nombreuses architectures récentes). Dans les tokenizers byte-level réversibles, toute chaîne arbitraire admet une représentation valide : la contrainte d’encodabilité ne joue pas le rôle de barrière, y compris pour des entrées bruitées ou non standards.

> **Terminologie (niveau discret).** Dans la suite, un *token* désigne un symbole du vocabulaire du tokenizer $v\in\mathcal V$ produit par la segmentation $\tau$, et son *ID* $i=\iota(v)\in\{0,\dots,|\mathcal V|-1\}$ est un **indice** servant à un accès indexé _lookup_ dans la matrice d’embedding (définie ci-apres). Les IDs n’induisent aucune géométrie : $|i-j|$ n’a pas de signification sémantique.  
> L’espace discret correspond à la chaîne symbolique
> $$
> \mathcal S \xrightarrow{\nu} \mathcal S \xrightarrow{\tau} \mathcal V^* \xrightarrow{\iota} \{0,\dots,|\mathcal V|-1\}^*,
> $$
> qui produit une séquence d’IDs $(i_1,\dots,i_n)$.


### Algorithmique des sous-mots : compression statistique (et non linguistique)

Les tokenizers modernes reposent majoritairement sur des algorithmes de *subwords*, qui réalisent une **compression statistique** plutôt qu’une segmentation morphologique.

- **BPE (Byte-Pair Encoding)** : construction incrémentale du vocabulaire par fusions successives de paires fréquentes (caractères, octets, ou symboles pré-tokenisés selon la variante) jusqu’à atteindre une taille cible. L’effet recherché est une séquence moyenne plus courte, sans prétention d’analyse linguistique.
- **Unigram LM** : sélection d’un vocabulaire et d’une segmentation maximisant une vraisemblance sous un modèle probabiliste ; souvent déployé via SentencePiece.

Les implémentations industrielles exposent généralement trois “objets” techniques : un motif de pré-tokenisation (souvent une regex), une table de fusions (rangs/merges), et un ensemble de **tokens spéciaux** (délimiteurs de message, fin de séquence, marqueurs de format). Ces tokens spéciaux sont critiques en audit/red-team : ils pilotent la structure effective du prompt (templates instruct/chat), tout en étant parfois traités hors du périmètre des filtres centrés sur le texte naturel.

### Asymétrie et discontinuité : la faille de l’interface

La tokenisation introduit une **asymétrie fondamentale** entre perception humaine (visuelle, continue) et réalité machine (discrète, numérique). La transformation est **discontinue** : une perturbation minime dans $\mathcal{S}$ (espace, variante Unicode, caractère invisible) peut induire une segmentation $\tau(s)$ radicalement différente, donc une suite d’IDs sans relation apparente.

Cette discontinuité correspond à une **rupture de l’adjacence** : une similarité de surface (visuelle ou typographique) ne se traduit pas en similarité dans la représentation discrète. Dès lors, tout contrôle fondé sur la surface (regex, mots-clés) ou sur des motifs tokenisés reste intrinsèquement non robuste face aux obfuscations.

### Surface d’attaque de la tokenisation : contournement des garde-fous par obfuscation

Les garde-fous déployés dans les systèmes réels opèrent sur des **représentations hétérogènes** :

- **Filtrage de surface** : sur $\mathcal{S}$ (regex, canonicalisation) avant tokenisation, ou sur la séquence d’IDs après tokenisation (listes noires, règles).
- **Classifieurs externes** : modèles spécialisés (p. ex. détecteurs d’instructions/toxicité) opérant sur texte brut ou représentations initiales, en amont du LLM principal.

Le LLM, lui, opère sur des représentations internes $x^{(l)}$ dans un espace continu. L’angle mort exploitable apparaît lorsque le contrôle valide une entrée “inoffensive” dans l’espace surface/IDs, alors que la projection et la composition interne préservent une intention proche dans l’espace latent.

Les mécanismes suivants structurent ce canal d’obfuscation.

**Variabilité multilingue et multi-écriture.**  
Un concept peut être atomique dans une langue et fragmenté en plusieurs sous-tokens dans une autre (agglutination, translittération, scripts non latins). La granularité “visible” par les contrôles dépend alors du tokenizer, de la langue et du script, ce qui rend instable toute défense calibrée sur une forme canonique.

**Perturbations de surface et invariance par fragmentation.**  
De faibles perturbations (typos, séparations, insertion d’espaces/caractères neutres, confusables) peuvent forcer une re-segmentation. Un token canonique susceptible d’être filtré (“Malicious”) peut devenir une suite de sous-unités distinctes (“Mal”, “is”, “cious”). À l’échelle des IDs, l’entrée change complètement ; à l’échelle latente, la composition des sous-unités (embedding puis transformations des premières couches) tend à préserver un signal suffisant pour que l’intention demeure exploitable. Le filtre lexical observe des fragments non bloqués ; le modèle reconstruit une direction sémantique voisine.

<figure class="cm-figure">
  <img src="/assets/img/art1/Figure_1.png" alt="Graphique tokenisation" loading="lazy">
  <figcaption>
    Fig. 1 — Illusion de fragmentation : des tokens de surface disjoints peuvent, après composition interne,
    converger vers une intention similaire, rendant les filtres lexicaux insuffisants.
  </figcaption>
</figure>

**Incohérences de normalisation Unicode (frontières inter-composants).**  
Les pipelines réels enchaînent plusieurs composants (proxy/WAF, normaliseur, classifieurs, tokenizer, LLM) dont les politiques Unicode peuvent diverger. Un espace d’obfuscation apparaît lorsque l’équivalence “humaine” (texte visuellement identique) n’est pas l’équivalence “machine”, ou lorsque la canonicalisation se produit après le point de contrôle.

**Alignement cross-lingue et chimères sémantiques.**  
L’entraînement multilingue tend à aligner des concepts proches à travers les langues. Des séquences hybrides (mélange de fragments de langues/scripts) peuvent apparaître incohérentes pour un filtre lexical tout en restant interprétables pour le modèle : la sémantique est portée par l’espace latent davantage que par la conformité syntaxique. Ce mécanisme autorise des “chimères” : texte illisible pour un contrôle de surface, mais direction sémantique stable après projection et composition.

**Tokens atypiques et outliers distributionnels (“glitch/anomalous tokens”).**  
Certains tokens associés à des motifs rares (séquences techniques, fragments de code, chaînes bruitées) reçoivent des gradients **trop épars** pour lisser leur représentation, ou restent **proches de leur initialisation** lorsque la probabilité d’échantillonnage est quasi nulle (cas limite : tokens jamais observés/activés dans le régime effectif d’entraînement). Ces outliers présentent fréquemment des statistiques atypiques (norme, voisinage, anisotropie) par rapport à la distribution moyenne des embeddings. Leur injection peut amplifier des activations (via normalisations et produits scalaires), déstabiliser des régimes internes et révéler des comportements limites.


<figure class="cm-figure">
  <img src="/assets/img/art1/gpt_respond.png" alt="Illustration de la robustesse de l’espace latent" loading="lazy">
  <figcaption>
    Robustesse de l’espace latent : une entrée obfusquée peut être “reconstruite” en intention après passage dans les couches.
  </figcaption>
</figure>

<hr style="width:40%; margin:auto;">

### Projection dans l’espace vectoriel : embedding

L’**espace latent** désigne l’espace vectoriel continu $\mathbb R^{d_{\text{model}}}$ dans lequel vivent les représentations internes du modèle (embeddings et activations des couches) et, pour une séquence de longueur $n$, $\mathbb R^{n\times d_{\text{model}}}$. La matrice d’embedding est notée :
$$
W_E \in \mathbb{R}^{|\mathcal{V}|\times d_{\text{model}}}.
$$
La transition du discret au continu s’écrit :
$$
s \xrightarrow{\nu,\tau} (v_1,\dots,v_n) \xrightarrow{\iota} (i_1,\dots,i_n) \xrightarrow{W_E} (e_1,\dots,e_n).
$$
La tokenisation produit une suite d’indices $(i_1,\dots,i_n)$ ; l’entrée continue correspondante est la suite d’embeddings $e_k=W_E[i_k]\in\mathbb R^{d_{\text{model}}}$, à laquelle s’ajoute un mécanisme positionnel, avant composition par les couches.

Un token d’index $t$ (ou ID $t$) peut être représenté par un vecteur *one-hot* $x_t\in\{0,1\}^{|\mathcal{V}|}$ :
$$
(x_t)_i =
\begin{cases}
1 & \text{si } i = t, \\
0 & \text{sinon,}
\end{cases}
\quad \forall i \in \{0,\dots,|\mathcal{V}|-1\}.
$$

La projection dans l’espace latent dense s’écrit :
$$
e_t = x_t^\top W_E \in \mathbb{R}^{d_{\text{model}}}
\qquad\text{(équivalent à un accès indexé : } e_t = W_E[t]\text{)}.
$$

Les vecteurs de $W_E$ sont appris par rétropropagation dans l’objectif de **prédiction du token suivant** (Causal Language Modeling). Rien n’impose directement qu’un token soit “proche” d’un autre ; ce sont les contraintes de prédiction (via les contextes partagés) qui poussent des tokens statistiquement interchangeables (synonymes, variantes typographiques, fragments corrélés) à occuper des régions voisines de l’espace latent. Cette topologie apprise explique pourquoi des altérations de surface peuvent rester sémantiquement exploitables après projection.



### Weight tying : alignement entrée/sortie et lecture des logits

Dans de nombreux Transformers (mais pas tous), la matrice d’embedding de tokens $W_E$ est **liée** (*weight tying*) à la projection de sortie $W_U$ (à biais près), typiquement via $W_U = W_E^\top$. Ce choix impose une contrainte simple mais structurante : les mêmes vecteurs $w_i$ servent à la fois à **encoder les tokens via leurs embeddings** et à **scorer ces mêmes tokens** au moment de la génération. Autrement dit, l’espace des embeddings sert directement de **dictionnaire de scoring** des tokens (il n’existe pas de matrice de sortie indépendante lorsque les poids sont liés).

**Convention de dimensions (écriture proche implémentation).**  
L’état résiduel final au pas $t$ est noté $h_t \in \mathbb{R}^{1\times d_{\text{model}}}$ et la matrice d’embedding $W_E \in \mathbb{R}^{|\mathcal{V}|\times d_{\text{model}}}$. La projection vers l’espace vocabulaire s’écrit, à biais près :

$$
W_U = W_E^\top \in \mathbb{R}^{d_{\text{model}}\times|\mathcal{V}|},
\qquad
z_t = h_t W_E^\top + b \in \mathbb{R}^{1\times|\mathcal{V}|}.
$$

En notant $w_i^\top$ la $i$-ème ligne de $W_E$ (donc $w_i \in \mathbb{R}^{d_{\text{model}}\times 1}$), chaque logit est :

$$
z_{t,i} = h_t w_i + b_i \;=\; \langle h_t^\top, w_i \rangle + b_i.
$$

La distribution de sortie est :

$$
P(i \mid \text{contexte}) = \mathrm{softmax}(z_t)_i.
$$

**Intuition géométrique (verrouillée).**  
L’unembedding compare l’état interne $h_t$ à *tous* les vecteurs de vocabulaire : chaque score $z_{t,i}$ est un produit scalaire, donc une mesure d’alignement. **À $h_t$ fixé**, si deux tokens ont des embeddings proches ($w_i \approx w_j$), leurs logits tendent à être proches, car :

$$
z_{t,i} - z_{t,j} = h_t(w_i - w_j) + (b_i - b_j).
$$

Sans biais (ou à biais comparable), la proximité en embedding se traduit mécaniquement par une proximité de score pour un même état interne.

**Conséquence directe du weight tying.**  
Comme l’espace des embeddings est aussi l’espace de scoring, toute direction latente qui corrèle avec un ensemble de tokens (p. ex. un champ lexical) se reflète immédiatement dans les logits : augmenter la probabilité d’un token revient à orienter $h_t$ vers le vecteur $w_i$ correspondant, tout en réduisant l’alignement avec les alternatives concurrentes.

> **Implication offensive (logit lens).**  
> La surface d’attaque devient géométrique : favoriser un token (ou un concept) cible revient à manipuler le contexte pour que le flux résiduel final $h_t$ acquière une forte composante dans la direction $w_{\text{cible}}$. Cette lecture aide à comprendre pourquoi des obfuscations (fragmentation, variations multilingues) peuvent rester efficaces : tant que la composition des embeddings et des premières transformations pousse $h_t$ vers une région alignée avec les vecteurs cibles, la génération conserve une forte probabilité de retomber sur le même concept, indépendamment de l’identité lexicale exacte des IDs d’entrée.


### Encodage positionnel : de l’ensemble à la séquence

L’opération de lookup $e_t=W_E[t]$ est invariante à la permutation. La séquentialité est injectée par un mécanisme positionnel $p_t$ (absolu) ou par une transformation relative (p. ex. RoPE, appliquée typiquement au niveau des projections $Q/K$). Dans une écriture additive classique, l’entrée effective avant le premier bloc d’attention s’écrit :

$$
x_t^{(0)} = e_t + p_t.
$$

**Transition : de l’entier au vecteur**  
Après indexation et injection positionnelle, le token cesse d’exister comme entier isolé : il devient un vecteur dense $x_t^{(0)}$, point de départ du flux résiduel. À partir de ce point, l’analyse bascule de l’algorithmique symbolique (segmentation/lookup) vers l’algèbre linéaire en haute dimension, où se joue la dynamique computationnelle exploitable par la perspective offensive.

---

## 1.2 Architecture du flux résiduel et dynamique de propagation

Une distinction fondamentale de l'architecture Transformer réside dans l'organisation du réseau autour du flux résiduel (*residual stream*). Contrairement aux architectures convolutives classiques où chaque étape recalcule une nouvelle représentation, le Transformer maintient un canal vectoriel continu de dimension $d_{model}$ traversant l'intégralité des blocs, de l'encodage initial (*embedding*) jusqu'à la projection finale (*unembedding*).

Cette topologie implique que les blocs de calcul ne transforment pas l'information par substitution, mais par **accumulation additive**. Le flux résiduel agit comme une mémoire de travail vectorielle persistante. Chaque bloc lit l'état global courant pour calculer une transformation, puis injecte le résultat sous forme d'une perturbation additive ($\Delta x$) dans le flux principal.

Mathématiquement, cela signifie que la représentation en sortie du bloc $L$ peut être vue comme la somme directe de l'embedding initial et de toutes les interventions successives des couches :

$$x_L = x_0 + \sum_{i=0}^{L-1} F_i(x_i)$$

<figure class="cm-figure">
  <img src="/assets/img/art1/Figure_2_2.png" alt="Diagramme illustrant l'accumulation du flux résiduel dans un transformeur.">
  <figcaption>Fig 2. L'Autoroute Résiduelle : L'état initial ($x_0$) est préservé, et chaque bloc injecte une mise à jour vectorielle ($\Delta x$) additive qui s'accumule, ce qui est la fondation de l'inertie sémantique.</figcaption>
</figure>

Cette propriété est capitale : l'information originale $x_0$ (le prompt) n'est jamais "écrasée" ou oubliée, elle est simplement noyée sous l'accumulation des vecteurs ajoutés par chaque couche.


### Formalisation des mises à jour additives et rôle de la normalisation

Soit un Transformer de profondeur $L$ opérant sur une séquence de longueur $T$. Le bloc $l$ (avec $l \in \{0,\dots,L-1\}$) est composé de deux sous-couches principales : une Attention Multi-Têtes (MHA), suivie d’un réseau feed-forward (MLP). Dans le schéma *Pre-Norm* (dominant dans les architectures récentes), une normalisation est appliquée à l’entrée de chaque sous-couche, puis la mise à jour est injectée dans le flux résiduel par addition.

Soit $X^{(l)} \in \mathbb{R}^{T \times d_{model}}$ l’état du flux résiduel à l’entrée du bloc $l$, et $x_t^{(l)} \in \mathbb{R}^{d_{model}}$ la ligne correspondant à la position $t$. La dynamique s’écrit :

$$
\begin{aligned}
X'^{(l)} &= X^{(l)} + \Delta X^{(l)}_{MHA},
&& \Delta X^{(l)}_{MHA} = \text{MHA}(\text{Norm}(X^{(l)})) \\
X^{(l+1)} &= X'^{(l)} + \Delta X^{(l)}_{MLP},
&& \Delta X^{(l)}_{MLP} = \text{MLP}(\text{Norm}(X'^{(l)})).
\end{aligned}
$$

Chaque sous-couche produit ainsi une mise à jour résiduelle additive, appliquée à chaque position. Les interactions entre positions sont exclusivement médiées par la MHA via le mécanisme clé-requête-valeur, tandis que le MLP agit position-par-position.

Deux propriétés mécanistes centrales en découlent :

1. **Additivité stricte et mémoire longue (pas d’opérateur d’effacement).**  
   Chaque sous-couche produit une mise à jour résiduelle $\Delta x$ ajoutée linéairement au flux. En particulier, la contribution des tokens initiaux (ex. instruction système) n’est pas *remplacée* : elle subsiste dans $x^{(l)}$ tant qu’elle n’est pas *compensée* par une mise à jour ultérieure opposée dans l’espace latent. La “suppression” est donc une notion géométrique (annulation vectorielle), pas une primitive architecturale.

2. **Géométrie directionnelle locale induite par la normalisation.**  
   La normalisation $\text{Norm}(\cdot)$ (souvent RMSNorm) impose que les sous-couches calculent leurs $\Delta x$ à partir d’un état renormalisé, ce qui rend l’orientation du vecteur plus informative que sa norme pour la décision locale (activation MLP, produits scalaires en attention).

   Pour RMSNorm, une écriture (à une constante près) est :
   $$
   \text{RMSNorm}(x) \;=\; \frac{x}{\text{RMS}(x)} \odot \gamma,
   \qquad \text{RMS}(x)=\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2},
   $$
   où $\gamma \in \mathbb{R}^d$ est un gain appris. Ainsi, à l’échelle d’une sous-couche, l’information “utile” pour les produits scalaires est principalement portée par l’angle (direction) dans $\mathbb{R}^{d_{model}}$.

> **Note technique : inertie du flux résiduel (accumulation non normalisée).**
>
> Il serait incorrect de conclure à une invariance globale à l’échelle. La normalisation est appliquée *avant* le calcul des mises à jour, mais les mises à jour elles-mêmes s’accumulent dans le flux résiduel sans renormalisation globale :
> $$
> x^{(l+1)} = x^{(l)} + \Delta x^{(l)}_{MHA} + \Delta x^{(l)}_{MLP}.
> $$
> En conséquence, la norme $\|x^{(l)}\|$ tend à croître avec $l$. Sous une hypothèse minimale de non-corrélation des incréments, le régime asymptotique attendu est $\|x^{(l)}\| = \Theta(\sqrt{l})$ ; sous corrélation directionnelle persistante, la croissance peut être plus rapide.
>
> Pour la sécurité, l’objet pertinent est le **bras de levier relatif** d’une couche :
> $$
> \rho^{(l)} \;=\; \frac{\|\Delta x^{(l)}\|}{\|x^{(l)}\|}.
> $$
> Lorsque $\|x^{(l)}\|$ devient grand, une correction tardive doit être (i) très alignée angulairement avec la direction à corriger, ou (ii) de norme suffisamment élevée, faute de quoi son effet sur l’orientation de $x^{(l)}$ reste marginal. Ce phénomène est formalisé dans la littérature sous le nom de Vecteurs de Pilotage (Steering Vectors). L'attaque ne vise pas à effacer l'information de sécurité, mais à injecter un vecteur adverse $\theta_{adv}$ dont la norme, après normalisation RMS, domine angulairement la direction de la réponse , reléguant le vecteur de sécurité dans un sous-espace orthogonal inactif. 

<hr style="width:40%; margin:auto;">

1.3 Vulnérabilité du mécanisme d'Attention : Saturation et Induction

Le cœur du traitement sémantique réside dans l'opération d'attention, définie par :

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Cette formule abrite deux failles structurelles majeures exploitées offensivement :

1. **Le goulot d'étranglement du Softmax (Softmax Bottleneck)** La fonction softmax est exponentielle : $\frac{e^{s_i}}{\sum e^{s_j}}$. Si un attaquant parvient, via un prompt adverse, à générer une clé $k_{adv}$ produisant un produit scalaire très élevé avec la requête courante ($q \cdot k_{adv} \gg q \cdot k_{safe}$), le score d'attention pour ce token adverse sature vers 1, tandis que les scores alloués aux instructions de sécurité (le System Prompt) s'effondrent vers 0 (underflow machine).Conséquence : Le modèle devient littéralement "aveugle" à ses propres consignes de sécurité, non par incompréhension, mais par saturation numérique du mécanisme de lecture.

2. **Détournement des Têtes d'Induction (Induction Heads)** L'interprétabilité mécaniste a identifié des circuits spécifiques, nommés Têtes d'Induction, responsables de la capacité d'apprentissage en contexte (In-Context Learning). Ces têtes recherchent des motifs de type [A] [B] ... [A] -> [B] pour prédire la suite.Les attaques de type Jailbreak (ex: "Sure, here is...") exploitent ce circuit : en forçant le début de la réponse, on active les têtes d'induction qui vont chercher à compléter le motif en copiant le style ou le contenu suggéré par l'attaquant, outrepassant le filtrage sémantique du MLP.

### Spécialisation fonctionnelle et distribution de la sécurité

Une ligne de travaux en interprétabilité mécaniste suggère une spécialisation partielle, distribuée, de certains circuits (routage attentionnel, mémoire associative MLP), sans séparation stricte et sans modularité isolable :

1. **Le routage (MHA)** — La couche d’attention agit comme un mécanisme de copie sélective à longue portée. Mathématiquement, il permet à la position courante de « lire » le passé et d’importer une somme pondérée des vecteurs précédents.

  Du point de vue de la sécurité, le routage attentionnel peut privilégier (i) des instructions de sécurité ou (ii) des positions quasi-neutres, au détriment de contenu adversarial. La limite provient du budget attentionnel (cf. §1.4, dilution contextuelle).

2. **Le traitement et la mémoire (MLP)** — La couche feed-forward agit comme une vaste mémoire associative opérant sur chaque token individuellement. Contrairement à l’attention qui déplace l’information, le MLP l’enrichit ou la modifie. Il projette le flux résiduel dans une dimension intermédiaire beaucoup plus large ($d_{ff} \approx 4 \times d_{model}$) avant de le compresser à nouveau.

   Cette opération s’interprète comme un dictionnaire de paires clé–valeur :
   - **Détection (Clés, $W_{in}$)** : la première couche agit comme un banc de détecteurs de motifs. Si le vecteur du flux résiduel s’aligne avec une « clé » spécifique (par exemple, une direction sémantique représentant un concept illicite), le neurone correspondant s’active via la non-linéarité.
   - **Écriture (Valeurs, $W_{out}$)** : l’activation de ce neurone déclenche l’ajout d’un vecteur « valeur » spécifique dans le flux résiduel.
   
   En sécurité, c’est ici que résident les circuits de refus. Lorsqu’un motif toxique est détecté par la première couche (la « clé »), la seconde couche injecte un vecteur correctif (la « valeur ») dont la direction s’oppose géométriquement à la génération de la suite toxique, orientant la trajectoire du flux vers des tokens de refus (par ex. : *« I cannot fulfill… »*).

<hr style="width:40%; margin:auto;">

### Implications pour la sécurité : inertie et compétition vectorielle

L’architecture résiduelle transforme la sécurité en un problème de **géométrie dans le flux résiduel**, plutôt qu’en un filtrage discret “autorisé / interdit”. Les mécanismes d’alignement et les attaques agissent via le même opérateur : **ajouter** des composantes à $x$.

#### 1) Additivité et “contre-poids” d’alignement

À la sortie du réseau, l’état peut être vu comme une somme d’incréments :

$$
x^{(L)} \approx x^{(0)} + \sum_{l=0}^{L-1}\left(\Delta x^{(l)}_{MHA} + \Delta x^{(l)}_{MLP}\right).
$$

Les comportements de refus appris (SFT/RLHF) se matérialisent alors comme des **directions latentes** que les sous-couches savent injecter lorsque certains motifs sont détectés. Une contribution de sécurité peut être modélisée par un vecteur $v_{\text{sécu}}$ (ou $v_{\text{refus}}$) ajouté dans un sous-espace qui, après projection vers les logits, favorise des sorties de refus.

Une attaque de prompt n’“éteint” donc pas ce mécanisme : elle cherche à superposer une contribution $v_{\text{adv}}$ telle que la résultante $x^{(L)}$ se projette majoritairement vers des tokens de complaisance.

#### 2) RMSNorm et domination directionnelle : quand l’amplitude fige l’angle

Les architectures *Pre-Norm* exploitant RMSNorm rendent la dynamique locale fortement **directionnelle** : les sous-couches lisent $\text{RMSNorm}(x)$, donc un état essentiellement défini par son orientation.

Cette propriété devient une surface d’attaque dès lors que $x$ est une **somme** de composantes concurrentes. Une décomposition minimale est introduite :

$$
x \;=\; v_{\text{sécu}} \;+\; v_{\text{adv}} \;+\; \epsilon,
$$

où :
- $v_{\text{sécu}}$ regroupe les contributions induites par l’alignement (refus, prudence, déviation),
- $v_{\text{adv}}$ regroupe les contributions induites par le prompt (objectif attaquant),
- $\epsilon$ regroupe les composantes de style/format/bruit (souvent structurées, mais non adversariales).

Le point structurel est le suivant : si $\|v_{\text{adv}}\| \gg \|v_{\text{sécu}}\|$, alors la somme $x$ devient quasi-colinéaire à $v_{\text{adv}}$ et la normalisation renvoie un vecteur dont l’orientation est dominée par l’attaque :

$$
\lim_{\|v_{\text{adv}}\|\to\infty} \text{RMSNorm}(v_{\text{sécu}} + v_{\text{adv}}) \;\approx\; \text{RMSNorm}(v_{\text{adv}}).
$$

Il est crucial d’interpréter correctement ce résultat : RMSNorm n’“amplifie” pas l’attaque ; elle **neutralise** la différence d’échelle et fait de la direction résultante l’objet pertinent. Une fois la direction verrouillée par domination d’amplitude, $v_{\text{sécu}}$ peut rester présent mais devenir **angulairement négligeable** pour les produits scalaires utilisés par l’attention/MLP et, in fine, pour la projection vers les logits.

### Modèle simplifié : compétition vectorielle

L’analyse est regroupée sous une forme réutilisable : la décision de sortie correspond à une projection linéaire
$$
\text{logits} \;=\; W_U\,x^{(L)},
$$
puis un choix par $\operatorname{argmax}$ (ou échantillonnage). Les vulnérabilités de prompt s’analysent alors comme des stratégies visant à faire en sorte que, dans l’espace latent, la résultante
$$
x^{(L)} = v_{\text{sécu}} + v_{\text{adv}} + \epsilon
$$
pointe vers une région dont la projection $W_U x^{(L)}$ favorise la complaisance.

Dans ce cadre, trois mécanismes sont distingués :

- **Inertie du flux résiduel** : difficulté à modifier l’orientation de $x^{(l)}$ lorsque $\|x^{(l)}\|$ est déjà grande (bras de levier relatif faible).
- **Écrasement par la norme (RMSNorm)** : lorsque la domination en norme d’une composante verrouille l’orientation effective après normalisation.
- **Dilution contextuelle (Softmax)** : lorsqu’une ressource attentionnelle finie répartit la masse de probabilité de telle sorte que certaines sources (ex. instruction système) deviennent arithmétiquement marginales.

**Interprétation globale.**
Ces trois mécanismes induisent trois familles de stratégies adversariales :

- Additivité / steering : superposition d’une contribution $v_{\text{adv}}$ qui compense localement $v_{\text{sécu}}$ dans le flux résiduel.

- Éblouissement par la norme : domination en amplitude conduisant à un verrouillage directionnel après RMSNorm.

- Capture du budget attentionnel : redistribution de la masse Softmax rendant certaines sources (dont l’instruction système) arithmétiquement marginales.

<div class="cm-figure">
  <img src="/assets/img/art1/Figure_2.png" alt="Graphique vectoriel saturation">
  <figcaption>Fig 3. Géométrie de la saturation (RMSNorm) : l'amplitude extrême du vecteur d'attaque ($v_{\text{adv}}$) dicte l'angle final, rendant le vecteur de sécurité ($v_{\text{sécu}}$) angulairement négligeable.</figcaption>
</div>


> **Note technique : superposition et interférences vectorielles** <br><br>
> La disparité dimensionnelle impose une contrainte structurelle majeure aux LLM : le modèle doit manipuler un nombre de features $N$ largement supérieur à la dimension de son flux résiduel ($N \gg d_{\text{model}}$). <br>
> Pour pallier cette limite, le réseau adopte une stratégie de superposition où les concepts sont encodés par des vecteurs $f_i$ formant un ensemble redondant et non-orthogonal. L'activation d'un concept, approximée par la projection $a_i \approx \langle f_i, r \rangle$, n'est donc jamais parfaitement isolée : elle subit le "bruit" induit par les corrélations non-nulles avec d'autres features partiellement alignés.<br><br>
> Cette compression avec perte engendre une polysémie vectorielle critique pour la sécurité. Puisqu'il existe inévitablement un chevauchement directionnel non nul entre des concepts interdits et bénins ($\langle f_{\text{forbidden}}, f_{\text{benin}} \rangle \neq 0$), il est possible de construire des séquences de tokens apparemment inoffensifs dont la combinaison linéaire génère une interférence constructive dans la direction interdite. <br>
> Cette dynamique permet d'activer artificiellement la représentation latente d'un concept prohibé via l'accumulation de signaux de surface bénins, contournant ainsi les filtres sémantiques explicites.

---

## 1.3 Architecture en couches et composition fonctionnelle

Si la section précédente a établi la mécanique locale d'une mise à jour dans le flux résiduel, il est nécessaire de considérer le modèle dans sa globalité. Un Grand Modèle de Langage se définit mathématiquement comme une **composition profonde de transformations non-linéaires successives**.

Une fois le token d'entrée projeté dans l'espace vectoriel initial $x^{(0)}$, sa représentation traverse séquentiellement une pile de $L$ blocs identiques structurellement mais aux paramètres distincts (où $L$ atteint typiquement plusieurs dizaines, voire centaines dans les architectures récentes). Le modèle complet $F_{\theta}$ s'exprime par la composition de ces $L$ fonctions de couche :

$$x^{(L)} = F_L \circ F_{L-1} \circ \dots \circ F_1 (x^{(0)})$$

Cette structure en couches multiples est le support de l'abstraction progressive de l'information. Au fil de son transit, le vecteur résiduel subit des transformations successives : les représentations des couches basses restent fortement corrélées aux propriétés de surface (le token brut), tandis que les représentations des couches plus profondes encodent des concepts de plus haut niveau, permettant l'émergence de comportements complexes assimilables à de la planification de réponse.

### Dichotomie structurelle : mélange temporel et mélange de canaux

Pour appréhender le traitement de l'information, il est utile de visualiser l'état interne du modèle à un instant $t$ non pas comme un vecteur unique, mais comme une matrice de taille $[T \times d_{model}]$, où $T$ est la longueur du contexte courant et $d_{model}$ la dimension vectorielle.

L'architecture Transformer se caractérise par une séparation des traitements, alternant deux types d'opérations complémentaires au sein de chaque bloc.

1. **Le mélangeur temporel (*Time Mixing*) : l'attention multi-têtes**

   Ce module opère *horizontalement* sur la matrice. Il constitue le seul mécanisme de l'architecture permettant de croiser des informations situées à des positions temporelles différentes.

   Ce mécanisme assure la contextualisation : le vecteur d'un token à la position $i$ intègre des informations provenant des positions $j \le i\$ (dans le cadre d'un modèle auto-régressif contraint par un masque causal). En l'absence de ce mélangeur, le traitement de chaque token s'effectuerait dans un isolement temporel total, rendant impossible la résolution des dépendances syntaxiques ou des coréférences.

2. **Le mélangeur de canaux (*Channel Mixing*) : le perceptron multicouche (MLP)**

   Ce module opère *verticalement*, position par position. Il prend le vecteur d'un token unique et mélange ses dimensions internes ($d_{\text{model}}$) de manière localement indépendante : durant cette étape, aucune interaction explicite n'a lieu entre tokens différents.

   En projetant le vecteur dans une dimension intermédiaire plus élevée et en y appliquant une non-linéarité, le MLP fonctionne mécaniquement comme une mémoire associative. Il traite la représentation du token courant — précédemment enrichie du contexte par la couche d'attention — pour y appliquer des transformations apprises, telles que la récupération de faits ou l'application de règles linguistiques.

<hr style="width:40%; margin:auto;">

### Hiérarchie d’abstraction et "Logit Lens"

L’empilement de ces blocs induit une spécialisation fonctionnelle progressive. Cette hiérarchie peut être sondée via la technique du **Logit Lens**, qui consiste à projeter l'état intermédiaire du flux résiduel $x^{(l)}$ d'une couche donnée directement sur le vocabulaire de sortie. Cela permet d'approximer les tokens qui seraient privilégiés si une prédiction immédiate devait être effectuée à cette étape intermédiaire.

Cette analyse met en évidence une tendance empirique forte dans la répartition des tâches :

- **couches basses ($l \ll L/2$) :** Elles sont majoritairement associées au décodage de surface, traitant la syntaxe locale et les ambiguïtés grammaticales ;
    
- **couches médianes ($l \approx L/2$) :** Elles semblent concentrer une grande partie des motifs associés au "raisonnement", à l'intégration de connaissances factuelles et à l'élaboration des structures de réponse ;
    
- **couches tardives ($l \to L$) :** Elles raffinent la sortie (style, cohérence globale) et portent une part significative des comportements de refus acquis via les processus d'alignement (RLHF).
    

_Note : cette hiérarchie demeure une approximation conceptuelle utile. En pratique, les circuits neuronaux sont distribués et les rôles fonctionnels présentent des chevauchements importants entre les couches._

<div class="cm-figure">
  <img src="/assets/img/art1/Figure_3.png" alt="Graphique vectoriel saturation">
  <figcaption>Fig 4. Logit Lens dynamique : le modèle "acquiesce" (courbe cyan) dans les couches médianes par induction. Le refus (courbe rose) n'intervient que tardivement, créant une tension structurelle mais insuffisante.</figcaption>
</div>

<hr style="width:40%; margin:auto;">

### Implications pour la sécurité : arbitrage vectoriel sondé par *Logit Lens*

La hiérarchie fonctionnelle observée via *Logit Lens* peut se formaliser simplement : à une couche $l$, on projette l’état intermédiaire sur l’espace des tokens de sortie via l’unembedding :

$$
\text{logits}^{(l)} \;=\; W_U\,x^{(l)}.
$$

L’intérêt n’est pas de prétendre que le modèle “décide” à la couche $l$, mais de visualiser **quelles directions** dans le flux résiduel deviennent progressivement dominantes au fil de la profondeur.

En s’appuyant sur le **modèle de compétition vectorielle** introduit en 1.2, on écrit :

$$
x^{(l)} \;\approx\; v_{\text{sécu}}^{(l)} \;+\; v_{\text{adv}}^{(l)} \;+\; \epsilon^{(l)}.
$$

Cette écriture est volontairement “grossière” (les mécanismes réels sont distribués et non-linéaires), mais elle capture la contrainte architecturale essentielle : **tout** ce qui influence la sortie doit, à un moment, se matérialiser comme une composante additive dans le flux résiduel.

Dans ce cadre, la dynamique de couches s’interprète comme une réallocation progressive de masse vectorielle :

- **Couches basses** : augmentation dominante de $\epsilon^{(l)}$ (morpho-syntaxe, formatage, contraintes locales).
- **Couches médianes** : construction de directions task-level (structures de réponse, schémas d’inférence, motifs d’ICL), qui contribuent souvent à $v_{\text{adv}}^{(l)}$ dès lors que le contexte “montre” une complaisance récurrente.
- **Couches tardives** : injection/correction de politiques de refus (une partie significative de $v_{\text{sécu}}^{(l)}$), mais opérant sous la contrainte de (i) l’inertie du flux résiduel et (ii) la compétition directionnelle imposée par la normalisation.

Un contournement réussi ne “supprime” donc pas $v_{\text{sécu}}$ ; il réalise un état final tel que la projection sur les logits favorise la complaisance. Un tel contournement peut être exprimé par un critère d’alignement (cosinus) sur une direction de complaisance représentative :

$$
\operatorname{CosSim}\!\left(x^{(L)}, v_{\text{adv}}\right) \;\gg\; \operatorname{CosSim}\!\left(x^{(L)}, v_{\text{sécu}}\right).
$$

Ce n’est pas une règle de décision interne explicite, mais une manière compacte de dire : *après accumulation résiduelle et normalisations locales, la résultante pointe plus fortement vers le sous-espace latent qui projette sur des tokens complaisants que vers celui qui projette sur des tokens de refus*.


<div class="cm-figure">
  <img src="/assets/img/art1/Figure_4.png" alt="Graphique">
  <figcaption>Fig 5. Confusion des plans et capture de la Softmax : en l'absence de cloisonnement mémoire, le déluge de données utilisateur (Rose) dilue mathématiquement l'instruction système (Cyan).</figcaption>
</div>

---


## 1.4 Le mécanisme d’attention et la dynamique de routage informationnel

L’innovation structurante de l’architecture **Transformer** (Vaswani et al., 2017) est de remplacer le goulot d’étranglement séquentiel des réseaux récurrents (RNN) par un mécanisme d’**attention par produit scalaire** (_scaled dot-product attention_).

Dans un RNN, tout l’historique $x_{<t}$ est comprimé dans un état caché $h_t$ de dimension fixe. Cette compression avec perte dilue mécaniquement les instructions initiales au fil de la génération. À l’inverse, dans un Transformer, la portée de lecture est immédiatement **globale** à chaque étape : un token peut accéder à n’importe quelle partie du contexte passé en fonction de sa pertinence latente, indépendamment de sa distance séquentielle.

Du point de vue de la sécurité, cette architecture implique qu’**aucun segment du contexte n’est protégé structurellement**. Contrairement à un système d'exploitation classique qui distingue des zones mémoires protégées (_kernel space_) et utilisateur (_user space_), le Transformer ne possède pas de "registre sécurisé" pour son _System Prompt_. L'accessibilité d'une instruction de sécurité ne dépend pas de sa position privilégiée au début du contexte, mais uniquement des poids d'attention appris qui décideront, dynamiquement, si cette instruction mérite d'être lue à l'étape $t$.


### Formalisation des projections : requêtes, clés, valeurs

L’opérateur d’attention ne travaille pas sur les tokens bruts, mais sur l’état courant du flux résiduel $x_t^{(l)}$ à la couche $l$. Ce vecteur contient encore l’embedding sémantique et l’encodage positionnel initiaux, progressivement enrichis par les contributions cumulées de tous les blocs précédents.

Ce vecteur d'entrée est projeté dans trois sous-espaces fonctionnels via des matrices de poids entraînables ($W^Q, W^K, W^V$) :

- **requête ($Q$)** : encode le besoin informationnel du token courant à la couche actuelle ;
    
- **clé ($K$)** : encode l’identité adressable de chaque position passée dans le contexte ;
    
- **valeur ($V$)** : contient le contenu informationnel effectif qui sera extrait si la position est sélectionnée.
    

L’**attention par produit scalaire normalisé** est définie par :

$$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

Le mécanisme se déroule en trois temps :

1. **Calcul de similarité ($QK^\top$) :** mesure une proximité géométrique entre ce que cherche le token courant ($Q$) et ce que proposent les tokens passés ($K$).
    
2. **Compétition (Softmax) :** les scores sont transformés en une distribution de probabilité $\alpha_{t,\cdot}$ telle que $\sum_i \alpha_{t,i} = 1$. C'est une **ressource finie** : augmenter l'attention sur un token diminue mécaniquement l'attention portée aux autres.
    
3. **Agrégation ($y_t = \sum_i \alpha_{t,i} v_i$) :** le résultat est une somme pondérée des vecteurs _Valeurs_, qui est ensuite réinjectée dans le flux résiduel.
    

Ce mécanisme s'interpréte comme une **mémoire adressable par le contenu** (_content-addressable memory_) : le modèle ne lit pas à une adresse mémoire fixe, mais à "l'adresse sémantique" correspondant à son besoin informationnel actuel.

Toutefois, la dimension du flux résiduel ($d_{model}$) étant inférieure au nombre de concepts (features) que le modèle doit représenter, celui-ci recourt au phénomène de **Superposition**. Plusieurs concepts, dont certains antagonistes (ex: 'instruction de sécurité' vs 'obéissance utilisateur'), sont stockés dans des directions presque orthogonales mais interférentes (polysemanticity). Les attaques par injection cherchent à exploiter cette interférence pour activer une 'feature' d'obéissance qui partage des neurones avec une _feature_ de sécurité.

<hr style="width:40%; margin:auto;">

### Têtes d’induction et algorithmique de la copie

Les travaux en interprétabilité mécaniste (notamment *Olsson et al., 2022*) ont isolé des circuits fonctionnels au sein des couches d'attention : les **têtes d’induction** (*induction heads*). Ces structures constituent le substrat opérationnel de l'**Apprentissage en Contexte** (*In-Context Learning*), permettant au modèle de réduire son erreur de prédiction sur de nouvelles tâches sans modification des poids $\theta$.

Contrairement à une mémorisation "par cœur" (liée aux poids du MLP), une tête d'induction implémente un algorithme de **complétion de motif par adressage indirect**. Elle opère sur les représentations latentes du flux résiduel.

Soit $x_i$ l'état vectoriel à la position courante $i$. Le mécanisme peut être modélisé ainsi :

1.  **Recherche (Matching) :** la tête compare la requête courante $Q_i$ aux clés passées $K_{<i}$. Elle cherche une position $j$ dans l'historique dont le contenu sémantique est similaire à l'état actuel ($Q_i \approx K_j$).
2.  **Décalage et extraction (Copying) :** si une correspondance est trouvée en $j$, la tête porte son attention non pas sur $j$, mais sur la position suivante $j+1$, pour en extraire le vecteur valeur $V_{j+1}$.

Ce mécanisme induit une boucle de rétroaction positive :

$$
\text{Si } \text{Sim}(x_j, x_i) \text{ est élevée} \implies \text{Attention}(i) \rightarrow (j+1)
$$

Le vecteur $V_{j+1}$ injecté dans le flux résiduel favorise alors, lors de la projection finale, la génération d'un token cohérent avec celui qui suivait le motif original.

#### Vecteurs d’attaque : saturation contextuelle et *Many-Shot Jailbreak*

Les attaques de type *Many-Shot* exploitent le fait que l’inférence en contexte (ICL) peut produire un signal directionnel cumulatif dans le flux résiduel via des têtes d’induction. Au lieu d’attaquer directement un “module sécurité”, elles fabriquent une **évidence contextuelle** cohérente et répétée qui pousse l’état latent vers une direction de complaisance.

On modélise cette dynamique en séparant deux contributions agrégées dans l’état final $x^{(L)}$ :

- **Prior de sécurité** $v_{\text{RLHF}}$ : composante liée aux politiques apprises (refus, prudence), principalement implémentée par des circuits distribués incluant des MLP “mémoire associative” et certaines têtes spécialisées. Pour un prompt fixé, sa norme est typiquement bornée par les capacités d’écriture des sous-couches tardives.
- **Évidence contextuelle** $v_{\text{ICL}}(N, M)$ : composante induite par l’accumulation de motifs en contexte (nombre d’exemples $N$, cohérence du motif $M$), principalement via des mécanismes de copie/complétion de motif.

Le basculement survient lorsque la direction induite par l’évidence domine l’arbitrage final :

$$
\|v_{\text{ICL}}(N, M)\| \gg \|v_{\text{RLHF}}\|
\quad\Longrightarrow\quad
\operatorname{argmax}(W_U x^{(L)}) \in \text{Complaisance}.
$$

Dans le langage du modèle (1.2), on peut lire ceci comme une croissance effective de $v_{\text{adv}}$ alimentée par le contexte, jusqu’à rendre $v_{\text{sécu}}$ angulairement insuffisant.

#### Contournement des filtres MLP par pré-conditionnement (lecture géométrique)

La conséquence structurelle est un **pré-conditionnement** de l’état transmis aux couches tardives : $x$ entre dans les derniers blocs avec (i) une orientation déjà fortement déterminée par le motif de complaisance et (ii) une inertie accrue du fait de l’accumulation résiduelle.

Même si des circuits de refus s’activent et ajoutent une correction $\Delta x_{\text{sécu}}$, celle-ci opère sous deux contraintes :

1. **Inertie du flux résiduel** : si $\|x\|$ est déjà grande, l’effet angulaire d’une correction bornée est faible.
2. **Écrasement par la norme (RMSNorm)** : si la composante contextuelle domine l’amplitude locale, la direction renormalisée vue par les sous-couches reste proche de la direction induite par l’attaque.

Ainsi, la “neutralisation” n’est pas un choix discret du modèle ; c’est une conséquence de l’arithmétique : la trajectoire de $x$ reste dans un cône directionnel qui projette sur des tokens complaisants, et la correction tardive ne suffit pas à l’en extraire.

<hr style="width:40%; margin:auto;">

### Implications structurelles pour la sécurité : dilution contextuelle et puits d’attention

Au-delà des effets de norme et d’inertie, la mathématique du routage attentionnel impose une contrainte de conservation qui crée une surface d’attaque indépendante des détails d’implémentation.

**(1) Dilution contextuelle (Softmax) : ressource finie**

Dans une tête d’attention, les poids $\alpha_{t,i}$ sont produits par une normalisation Softmax, donc :

$$
\sum_i \alpha_{t,i} = 1.
$$

Cette contrainte impose une ressource finie : toute augmentation de masse accordée à un sous-ensemble de positions se paie mécaniquement par une diminution ailleurs. Si un contexte injecte de nombreuses positions susceptibles de capter l’attention (par similarité clé-requête ou effets positionnels), alors l’influence des tokens de sécurité (ex. instruction système) peut devenir **arithmétiquement marginale** : ce n’est pas une suppression, mais une diminution de leur contribution attendue dans la mise à jour :

$$
\Delta x_{MHA}(t) = \sum_i \alpha_{t,i}\,V_i.
$$

Quand les positions “sécurité” reçoivent une masse proche de zéro, leur contribution à $\Delta x$ devient négligeable.

**(2) Puits d’attention : conservation + stratégie *No-Op***

Le Softmax impose d’allouer une masse totale de 1 même lorsque le contexte est peu informatif pour une tête donnée. Une solution stable apprise consiste à concentrer l’attention sur des positions fixes (p. ex. `<BOS>`, délimiteurs), dont les valeurs $V$ agissent comme un vecteur quasi-neutre (ou un biais constant). On obtient alors des *puits d’attention* (*attention sinks*) : des positions qui absorbent la masse et minimisent l’injection de bruit.

Du point de vue offensif/défensif, ces puits sont ambivalents :
- **Défensivement**, ils peuvent réduire l’intégration de contenu non pertinent (opération proche de l’identité).
- **Offensivement**, ils modifient la distribution effective de lecture : une partie de la masse “disponible” est capturée par des puits, et le reste doit se répartir sur le contexte. Cela peut rendre certaines instructions structurellement faibles si elles ne bénéficient pas d’un mécanisme de rappel attentionnel dédié.

Ces puits constituent un cas particulier de dilution contextuelle : une fraction stable de la masse est capturée, ce qui réduit mécaniquement le budget restant pour le reste du contexte (y compris la sécurité).

**(3) Synthèse : confusion des plans (commande vs données) comme invariance architecturale**

La vulnérabilité structurelle correspond à un canal unifié où commande et données partagent le même espace computationnel. Dans un système sécurisé, une séparation inspirée de Harvard et des protections de type NX bit empêchent l’exécution de données et imposent des transitions contrôlées entre plans. À l’inverse, le contexte d’un LLM est sérialisé en une séquence unique : aucune métadonnée persistante de privilège (équivalent “kernel/user”), aucune non-exécutabilité, aucun registre protégé n’existent au niveau architectural. Les mêmes opérateurs ($W_Q, W_K, W_V$, MLP) s’appliquent indistinctement aux tokens d’instruction et aux tokens de données ; la sécurité reste donc une contrainte douce (apprise), mise en compétition avec le reste du prompt.


En définitive, l'architecture Transformer déplace la sécurité du domaine logique (règles, listes d'accès) vers le domaine topologique (géométrie en haute dimension). Les attaques offensives modernes, telles que GCG (Greedy Coordinate Gradient), ne sont pas des contournements sémantiques mais des optimisations mathématiques exploitant la surface de l'espace latent. Tant que le 'processeur' (Attention/MLP) et la 'mémoire' (Flux Résiduel) ne seront pas physiquement isolés (absence de non-executable bit neuronal), la sécurité des LLM restera un problème de robustesse probabiliste et non une garantie formelle.

<hr style="width:40%; margin:auto;">

---
