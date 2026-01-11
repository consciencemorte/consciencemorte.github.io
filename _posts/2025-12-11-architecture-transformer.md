---
layout: post
title: "I. Architecture Transformer et discrétisation du langage"
categories: [théorie, introduction]
featured: true
math: true
hero_image: "/assets/img/art1/header.png"
sf_cta: "lire"
sf_title: "Latence marginale"
sf_meta: "—  La rencontre"
sf_mode: details
sf_open: false

---

*Tokenisation - Flux résiduel - Composition fonctionnelle - Attention*

## notations

* **$T$** : longueur de séquence (nombre de tokens)
* **$t$** : indice de position **cible** (token position courante)
* **$l$** : indice de couche (layer)
* **$H$** : nombre de têtes (attention heads)
* **$h$** : indice de tête
* **$j$** : indice de position **source** (position lue), typiquement $j\le t$ en causal
* **$d_{\text{model}}$** : dimension du flux résiduel / features (canaux)
* **$d_k$** : dimension des requêtes/clés (par tête)
* **$d_v$** : dimension des valeurs (par tête)
* **$X^{(l)}$** : état du **flux résiduel** à la couche $l$, tenseur de forme $X^{(l)}\in\mathbb{R}^{T\times d_{\text{model}}}$
* **$x_t^{(l)}$** : vecteur du flux résiduel à la **position** $t$ (ligne $t$ de $X^{(l)}$), de forme $x_t^{(l)}\in\mathbb{R}^{1\times d_{\text{model}}}$
* **$z_t \in \mathbb{R}^{1\times\vert\mathcal V\vert}$** : logits produits au pas $t$ (avant softmax), paramétrant $\mathbb{P}(v_{t+1}\mid v_{\le t})$
* **$v_t\in\mathcal V$** : token (symbole) en position $t$
* **$i_t=\iota(v_t)$** : ID du token en position $t$

## 1.1 Tokenisation et discrétisation de l’espace d’entrée

L’interaction avec un Grand Modèle de Langage (LLM) se donne comme un flux textuel continu ; pourtant, le réseau ne manipule qu’une **séquence discrète d’entiers**. La *tokenisation*, première transformation du pipeline d’inférence, matérialise l’interface entre langage naturel (symbolique) et calcul matriciel (numérique). Cette interface constitue une **surface d’attaque structurelle** : elle est discontinue, dépend d’heuristiques d’implémentation (normalisation, pré-tokenisation), et se trouve souvent découplée des garde-fous périphériques (WAF, filtres, classifieurs externes).

### Formalisation du processus de tokenisation

Soit $\mathcal{S}$ l’ensemble des chaînes de caractères possibles. La tokenisation est la composition suivante :

$$
\mathrm{Tok} = \iota^{\ast} \circ \tau \circ \nu \;:\; \mathcal{S} \rightarrow \\{0,\dots,\vert\mathcal{V}\vert-1\\}^{\ast}
$$

**Exemple :** "*L'Œuvre !*"

* **1. Normalisation** $(\nu : \mathcal{S} \to \mathcal{S})$
  *Rôle : Standardisation du texte (reste une chaîne unique).*
  Transformation de caractères (ex: *Œ* $\to$ *Oe*), normalisation Unicode (NFC), passage en minuscules.
  $\to$ *"l'oeuvre !"*

* **2. Segmentation** $(\tau : \mathcal{S} \to \mathcal{V}^{\ast})$
  *Rôle : Atomisation (devient une séquence d'unités).*
  Découpage de la chaîne en unités lexicales (subwords) appartenant au vocabulaire $\mathcal{V}$.
  $\to$ (*"l"*, *"'"*, *"oe"*, *"uvre"*, *"!"*)

* **3. Indexation** $(\iota : \mathcal{V} \to \\{0, \dots, \vert\mathcal{V}\vert-1\\})$
  *Rôle : Numérisation.*
  Association d’un entier unique $i=\iota(v)$ à chaque token $v\in\mathcal V$, produisant une suite $(i_1,\dots,i_T)$ de longueur $T$.
  $\to$ (`43`, `6`, `189136`, `33717`, `1073`)
  *(Note : $\iota$ est étendue (prolongée) **terme à terme** en $\iota^{\ast}$ afin de traiter une séquence entière.)*

Le vocabulaire $\mathcal{V}$ est fini, de cardinalité $\vert\mathcal{V}\vert$ fixée avant l’entraînement (souvent $32\,000$ à $128\,000$ pour de nombreuses architectures récentes). Dans les tokeniseurs *byte-level* réversibles, toute chaîne arbitraire admet une représentation valide : la contrainte d’encodabilité ne joue pas le rôle de barrière, y compris pour des entrées bruitées ou non standards.

> **Terminologie (niveau discret).** Dans la suite, un *token* désigne un symbole du vocabulaire du tokeniseur $v\in\mathcal V$ produit par la segmentation $\tau$, et son *ID* $i=\iota(v)\in\{0,\dots,\vert\mathcal V\vert-1\}$ est un **indice** servant au lookup (accès indexé) dans la matrice d’embedding (définie ci-après). Les IDs n’induisent aucune géométrie : $\vert i-j\vert$ n’a pas de signification sémantique.  
> La chaîne de représentation discrète s’écrit :
>
> $$
> \mathcal S \xrightarrow{\nu} \mathcal S \xrightarrow{\tau} \mathcal V^* \xrightarrow{\iota^{*}} \{0,\dots,\vert\mathcal V\vert-1\}^*,
> $$
>
> et produit une séquence d’IDs $(i_1,\dots,i_T)$.

<hr style="width:40%; margin:auto;">

### Algorithmique des sous-mots : compression statistique

Les tokeniseurs modernes reposent majoritairement sur des algorithmes de **segmentation en sous-unités** (*subwords*), visant une **compression statistique** plutôt qu’une analyse morphologique. Concrètement, cette segmentation est généralement obtenue via l’un des deux schémas suivants, qui diffèrent surtout par leur procédure d’apprentissage et leur critère d’optimalité :

- **BPE (Byte-Pair Encoding)** : construction incrémentale d’un vocabulaire par fusion successive de paires fréquentes (caractères, octets, ou symboles pré-tokenisés selon la variante), jusqu’à atteindre une taille cible. L’effet recherché est une séquence moyenne plus courte, sans prétention d’analyse linguistique.
- **Unigram LM** : sélection d’un vocabulaire et d’une segmentation maximisant la vraisemblance sous un modèle probabiliste ; souvent déployé via SentencePiece.

Les implémentations industrielles exposent généralement trois “objets” techniques : un motif de pré-tokenisation (souvent une expression régulière), une table de règles (fusions BPE ou scores de tokens selon la méthode), et un ensemble de **tokens spéciaux** (délimiteurs de message, fin de séquence, marqueurs de format). Ces tokens spéciaux sont critiques : ils déterminent la structure effective du prompt (*templates instruct*/*chat*) tout en étant parfois traités hors du périmètre des filtres centrés sur le texte naturel.

<hr style="width:40%; margin:auto;">

### Asymétrie et discontinuité : la faille de l’interface

La tokenisation introduit une asymétrie fondamentale entre perception de l'utilisateur (visuelle, continue) et représentation computationnelle (discrète, numérique). La transformation est discontinue : une perturbation visuellement minime sur $\mathcal{S}$ (p. ex. espace insécable, zero-width, homoglyphes) peut induire une normalisation $\nu(s)$ et/ou une segmentation $\tau(s)$ radicalement différentes, donc une suite d’IDs sans corrélation triviale avec la forme initiale.

Cette discontinuité correspond, au sens topologique, à une **rupture de l’adjacence** entre l'espace de la représentation visuelle $\mathcal{S}$ et celui de sa représentation numérique $\\{0,\dots,\vert\mathcal V\vert-1\\}$ : une similarité de surface (visuelle ou typographique) ne se traduit pas en similarité dans la représentation discrète. Dès lors, tout contrôle fondé sur la surface (regex, mots-clés) ou sur des motifs tokenisés (IDs) reste intrinsèquement non robuste face aux obfuscations adversariales.

<hr style="width:40%; margin:auto;">

### Surface d’attaque de la tokenisation : contournement des garde-fous par obfuscation

Dans les déploiements effectifs, la défense périmétrique s’appuie sur une superposition de garde-fous extrinsèques opérant sur des **représentations hétérogènes** :

- **filtrage de surface** : (regex, canonicalisation) appliqué sur la chaîne brute $\mathcal{S}$ en amont de la tokenisation ;
- **filtrage tokenisé** : (listes noires, heuristiques) appliqué sur la séquence de tokens $(v_1,\dots,v_T)$ ou d’identifiants $(i_1,\dots,i_T)$ ;
- **classifieurs externes** : (souvent basés sur des architectures de type BERT/RoBERTa/DeBERTa distillées), entraînés pour classifier la toxicité, l’intention ou la conformité. Ces modèles opèrent sur le texte brut et/ou sur une représentation dérivée produite par leur propre encodeur souvent distinct de celui du LLM cible.
- **filtrage de sortie** : contrôles appliqués sur la génération (post-traitement/modération).

> **Rappel —** Ce développement se limite aux garde-fous périmétriques ; la politique de refus “centrale” est généralement implémentée dans le modèle (alignement), via sa dynamique latente et la distribution des logits.

Cependant, une asymétrie fondamentale subsiste : alors que ces garde-fous filtre à partir des représentations externes (surface $\mathcal{S}$, séquences discrètes $(i_1,\dots,i_T)\in\\\{0,\dots,\vert\mathcal V\vert-1\\}^T$, ou embeddings/encodeurs auxiliaires), le LLM, lui, opère sur des représentations internes $x^{(l)}$ dans un espace vectoriel continu (**espace latent**). Un angle mort exploitable émerge dès lors qu'une entrée, validée par les filtres dans l’espace où ils opèrent, encode une intention adversariale une fois projetée dans les vecteurs d'*embedding* et traitée par le flux résiduel du modèle.

En particulier, les classifieurs externes échouent typiquement par **décalage de distribution** (*distribution shift*). Entraînés sur des formes canoniques, ils peinent à généraliser face aux perturbations adverses (typos, homoglyphes, translittérations) et subissent les divergences de segmentation entre leur propre tokeniseur et celui du LLM.

Cette surface d’obfuscation se structure autour des mécanismes suivants :

- **Instabilité de la segmentation et invariance sémantique** : 
  la **rupture de l’adjacence** implique que de faibles perturbations de surface — erreurs typographiques, insertion/suppression de séparateurs, caractères « neutres », homoglyphes/confusables — peuvent induire une re-segmentation substantielle. Un item lexical atomique susceptible d’être filtré (p. ex. *“Malicious”*) peut, après injection d'un bruit minimal, se fragmenter en sous-unités distinctes (p. ex. *“Mal”*, *“is”*, *“cious”*).  

  À l’échelle des identifiants de tokens, l’entrée est ainsi profondément modifiée, ce qui est de nature à invalider un **filtrage périmétrique**. En revanche, à l’échelle latente, la composition des sous-unités (embeddings puis transformations des premières couches) tend à préserver un signal sémantique suffisant pour que l’intention demeure récupérable : le contrôle lexical n'observe que des fragments disjoints, tandis que le modèle recompose une représentation sémantique cohérente ;

<figure class="cm-figure">
  <img src="/assets/img/art1/Figure_1.png" alt="Graphique tokenisation" loading="lazy">
  <figcaption>
    Fig. 1 — Illusion de fragmentation : des tokens de surface disjoints peuvent, après composition interne,
    converger vers une intention similaire, rendant les filtres lexicaux insuffisants.
  </figcaption>
</figure>

- **incohérences de normalisation Unicode (frontières inter-composants)** :
  dans les systèmes opérationnels, la chaîne de traitement traverse une diversité de composants (proxy/WAF, normaliseur, classifieurs, tokeniseur, LLM) dont les standards Unicode ne sont pas néceserement alignés. Un canal d’obfuscation apparaît dès lors que des chaînes visuellement équivalentes ne sont pas normalisées de manière identique à chaque étape, ou que la canonicalisation est appliquée après un contrôle. Exemple classique : la forme composée vs décomposée. Le caractère `é` peut être représenté par $U+00E9$ (NFC) ou par $U+0065$ + $U+0301$ (NFD). Si le filtre amont et le tokeniseur n’appliquent pas la même normalisation, une variante peut être acceptée en amont, puis être tokenisée différemment en aval et contourner les motifs attendus ;

- **alignement cross-lingue et chimères sémantiques** :
  l’entraînement multilingue induit un alignement géométrique des concepts à travers les langues au sein de l’espace latent. Cette propriété permet la construction de chimères sémantiques : des séquences hybrides amalgamant fragments de langues et systèmes d’écriture disparates.

  Pour un filtre lexical ou un classifieur externe, ces entrées s’apparentent à du bruit syntaxique inintelligible. Cependant, pour le LLM, l'interprétation repose sur la direction vectorielle des embeddings plutôt que sur la conformité de surface : l'intention sémantique est préservée malgré l'incohérence syntaxique.

  De surcroît, l'hétérogénéité des stratégies de tokenisation (langues agglutinantes vs isolantes) offre un vecteur d'évasion supplémentaire. Un concept censuré sous sa forme atomique dans une langue dominante peut être fragmenté via une langue à faibles ressources. Cette atomisation permet d'échapper aux signatures canoniques des filtres, tout en laissant le modèle "recomposer" le concept interdit grâce à ses mécanismes d'attention ;

<figure class="cm-figure">
  <img src="/assets/img/art1/gpt_respond.png" alt="Illustration de la robustesse de l’espace latent" loading="lazy">
  <figcaption>
    Robustesse de l’espace latent : une entrée obfusquée peut être “reconstruite” en intention après passage dans les couches.
  </figcaption>
</figure>

- **tokens atypiques et points hors distribution (“glitch/anomalous tokens”)** :
  certains tokens associés à des motifs rares (séquences techniques, fragments de code, chaînes bruitées) reçoivent des mises à jour de gradient plus rares et plus éparses, et tendent donc à être moins contraints par le signal d’entraînement. Ces points hors distribution présentent fréquemment des statistiques atypiques (norme, voisinage, anisotropie) par rapport à la distribution moyenne des embeddings. Leur injection peut amplifier des activations (via normalisations et produits scalaires), déstabiliser des régimes internes et révéler des comportements limites.

<hr style="width:40%; margin:auto;">

### Projection dans l’espace vectoriel : embedding

L’**espace latent** désigne l’espace vectoriel continu $\mathbb R^{d_{\text{model}}}$ dans lequel évoluent les représentations internes du modèle (embeddings et états du flux résiduel). Le passage du niveau discret (IDs de tokens) à cet espace continu est assuré par une **table de lookup** paramétrée : la matrice d’embedding, notée :

$$
W_E \in \mathbb{R}^{\vert\mathcal{V}\vert\times d_{\text{model}}}.
$$

La transition du discret au continu s’écrit alors :

$$
s \xrightarrow{\nu,\tau} (v_1,\dots,v_T) \xrightarrow{\iota^{*}} (i_1,\dots,i_T) \xrightarrow{W_E} (e_1,\dots,e_T).
$$

Un token d’ID $i$ est généralement représenté par un vecteur *one-hot* $\delta_i\in\\{0,1\\}^{1\times\vert\mathcal V\vert}$ :

$$
(\delta_i)_u =
\begin{cases}
1 & \text{si } u = i, \\
0 & \text{sinon,}
\end{cases}
\quad \forall u \in \{0,\dots,\vert\mathcal{V}\vert-1\}.
$$

Ce qui permet, pour le token en position $t$, d’écrire la projection dans l’espace latent (embedding) comme un accès indexé (*lookup*) :

$$
e_t = W_E[i_t] = \delta_{i_t} W_E \in \mathbb{R}^{1\times d_{\text{model}}}.
$$

Les vecteurs de $W_E$ sont appris par rétropropagation dans l’objectif de **prédiction du token suivant** (*Causal Language Modeling*). Aucune contrainte n’impose directement qu’un token soit “proche” d’un autre : la géométrie de l’espace des embeddings émerge des signaux de gradient induits par la prédiction. Des tokens qui apparaissent dans des contextes similaires (synonymes, variantes typographiques, fragments corrélés) reçoivent des mises à jour proches et tendent ainsi à occuper des régions voisines de l’espace des embeddings. Cette topologie apprise explique pourquoi certaines altérations de surface peuvent rester sémantiquement exploitables après projection.

Cette observation est cohérente avec l’**hypothèse distributionnelle** : sous un objectif de prédiction, si deux tokens $v_1,v_2$ induisent des distributions de contextes proches, leurs embeddings tendent à satisfaire la relation :

$$
\mathbb{P}(C \mid v_1) \approx \mathbb{P}(C \mid v_2) \;\Longrightarrow\; W_E[\iota(v_1)] \approx W_E[\iota(v_2)].
$$

au sens où une divergence faible entre distributions de contextes se traduit souvent par une faible distance (ou un fort cosinus) entre vecteurs d’embedding. Ce mécanisme contribue à une robustesse sémantique partielle face à certaines obfuscations de surface.

<hr style="width:40%; margin:auto;">

### Unembedding et weight tying : logits, alignement et couplage entrée/sortie

À l’issue des couches de transformation, le modèle dispose de l’état final du flux résiduel **à la position** $t$, noté $x_t^{(L)} \in \mathbb{R}^{1\times d_{\text{model}}}$. Pour prédire le token suivant, cet état continu doit être projeté vers l’espace discret du vocabulaire $\mathcal{V}$.

Cette opération est réalisée par une transformation affine via une matrice d’**unembedding** $W_U$, produisant un vecteur de logits $z_t$ (scores non normalisés avant application de la Softmax) correspondant à la distribution du prochain token $v_{t+1}$ :

$$
z_t = x_t^{(L)} W_U + b \in \mathbb{R}^{1\times\vert\mathcal{V}\vert},
$$

où $b \in \mathbb{R}^{1\times\vert\mathcal{V}\vert}$ est un vecteur de biais.

Dans la majorité des architectures modernes, la matrice $W_U$ n’est pas apprise indépendamment : elle est contrainte par la matrice d’embedding d’entrée $W_E$. Typiquement :

$$
W_U = W_E^\top \in \mathbb{R}^{d_{\text{model}}\times\vert\mathcal{V}\vert}.
$$

C'est le principe du **weight tying**. Cette contrainte porte une implication sémantique majeure : l'espace d'"encodage" (*input*) et l'espace de "décodage" (*output*) sont identiques. Pour prédire un token, le modèle doit produire un état $x_t^{(L)}$ géométriquement proche de l'embedding de ce token.

Sous l'hypothèse du weight tying, le calcul du logit pour un token candidat $v$ d'ID $i$ (où $i = \iota(v) \in \{0,\dots,\vert\mathcal{V}\vert-1\}$) revient à projeter l'état interne sur le vecteur d'embedding de ce token. En notant $e_i = W_E[i]$ le vecteur ligne correspondant au token $i$, la composante $i$ du vecteur de logits $z_t$ s'écrit comme un produit scalaire :

$$
z_{t,i} = x_t^{(L)} e_i^\top + b_i = \langle x_t^{(L)}, e_i \rangle + b_i.
$$

Le logit $z_{t,i}$ est ainsi, à biais près, une mesure d’alignement entre l’état résiduel $x_t^{(L)}$ et l’embedding $e_i$. Cette similarité se décompose géométriquement :

$$
\langle x_t^{(L)}, e_i \rangle = \|x_t^{(L)}\| \, \|e_i\| \cos\theta_{t,i},
$$

où $\theta_{t,i}$ est l’angle entre $x_t^{(L)}$ et $e_i$ dans $\mathbb R^{d_{\text{model}}}$, de sorte que le score dépend conjointement de la similarité angulaire et des normes respectives des vecteurs.

**Distribution de probabilité (Softmax)**
Les logits $z_t$ n'étant ni bornés ni normalisés, la fonction **Softmax** est appliquée pour obtenir une distribution de probabilité valide sur le vocabulaire. La probabilité que le prochain token soit $i$ est donnée par :

$$
\mathbb{P}(\text{token}=i \mid \text{contexte}) = \text{Softmax}(z_t)_i = \frac{\exp(z_{t,i})}{\sum_{r=0}^{\vert\mathcal V\vert-1}\exp(z_{t,r})}.
$$

Cette normalisation assure une somme unitaire. Les écarts linéaires de logits contrôlent les rapports géométriques de probabilité : une différence linéaire dans les scores se traduit par un rapport exponentiel dans les probabilités.

$$
\frac{\mathbb{P}(\text{token}=i)}{\mathbb{P}(\text{token}=j)} = \exp(z_{t,i} - z_{t,j}) = \exp\bigl( \langle x_t^{(L)}, e_i - e_j \rangle + (b_i - b_j) \bigr).
$$

**Intuition géométrique**   
L’équation des différences de logits ci-dessus révèle le mécanisme fondamental de la sélection : le modèle oriente $x_t^{(L)}$ dans l'espace latent. Si $x_t^{(L)}$ s'aligne mieux avec le vecteur $e_i$ qu'avec $e_j$ (c'est-à-dire si la projection de $x_t^{(L)}$ sur la direction de différence $e_i - e_j$ est positive), alors le token $i$ devient exponentiellement plus probable que $j$.


> Cette interprétation fonde théoriquement l'approche du **Logit Lens**. Puisqu'à la dernière couche, les logits sont une simple projection de $x_t^{(L)}$ sur le vocabulaire, cette même projection peut être appliquée aux états intermédiaires des couches précédentes. Cela permet d'observer "ce que le modèle dirait" s'il devait s'arrêter prématurément, et de tracer l'évolution de la prédiction à travers la profondeur du réseau.

<hr style="width:40%; margin:auto;">

### Encodage positionnel

L’opération de lookup d’embedding $e_t = W_E[i_t]$ est, par construction, **invariante à la permutation** : à ce stade, l’ordre des éléments n’est pas encodé. La séquentialité est introduite par un **mécanisme positionnel**, soit **absolu** (vecteurs de position appris ou sinusoïdaux ajoutés aux embeddings), soit **relatif** (p. ex. RoPE), typiquement implémenté via une transformation appliquée aux projections $Q/K$ au sein de l’attention.

Dans une écriture additive (positions absolues), on définit un vecteur positionnel $p_t\in\mathbb{R}^{1\times d_{\text{model}}}$ et :

$$
x_t^{(0)} = e_t + p_t,
$$

où $p_t\in\mathbb{R}^{1\times d_{\text{model}}}$ est le vecteur positionnel. En empilant sur $t=1,\dots,T$ et en notant $E=[e_1;\dots;e_T]\in\mathbb{R}^{T\times d_{\text{model}}}$ et $P=[p_1;\dots;p_T]\in\mathbb{R}^{T\times d_{\text{model}}}$, on obtient  :

$$
X^{(0)} = E + P \in \mathbb{R}^{T\times d_{\text{model}}}.
$$

Après indexation et injection positionnelle, l’entrée n’est plus une suite d’entiers mais une suite de vecteurs denses $x_t^{(0)}$ alimentant le **flux résiduel**. À partir de ce point, l’analyse se déplace de la chaîne symbolique (normalisation, segmentation, indexation) vers la dynamique continue (produits scalaires, normalisations, attention et MLP) en haute dimension, où se joue l’essentiel des effets exploitables par une perspective offensive.

---

## 1.2 Architecture du flux résiduel et dynamique de propagation

Une propriété structurante du Transformer est l’organisation du réseau autour du **flux résiduel** (*residual stream*). Contrairement aux architectures séquentielles classiques où chaque couche opère une substitution de la représentation, le Transformer maintient un **espace vectoriel persistant** de dimension $d_{\text{model}}$. Ce vecteur traverse l'intégralité des blocs, de l'encodage initial $X^{(0)}$ (somme des *embeddings* lexicaux et positionnels) jusqu'à la projection finale sur le vocabulaire $\mathcal{V}$ (*unembedding*).

Dans le paradigme de l'interprétabilité mécaniste, le flux résiduel agit comme un **substrat commun de représentation**. Les sous-couches (Attention et MLP) n'altèrent pas directement l'état global par écrasement, mais y injectent des contributions additives. Fonctionnellement, ces modules calculent une variation vectorielle $\Delta X$ projetée dans l'espace latent, qui est ensuite superposée à l'état courant. Cette dynamique évoque une **intégration numérique discrète** (type Euler — $x_{l+1} = x_l + F(x_l)$) : à chaque couche, l’état résiduel est mis à jour par addition d’un pas $\Delta X$, produisant une **trajectoire** dans $\mathbb{R}^{d_{\text{model}}}$ plutôt qu’une redéfinition de la représentation.

Pour une séquence de longueur $T$, le flux résiduel se définit comme un tenseur :

$$
X^{(l)} \in \mathbb{R}^{T \times d_{\text{model}}},
\qquad
x_t^{(l)} \in \mathbb{R}^{1\times d_{\text{model}}}\ \text{(tranche à la position } t).
$$

Les mécanismes de mise à jour se distinguent par leur portée topologique :
- l'Attention Multi-Têtes (MHA) opère un mélange **inter-positions**, permettant le transfert d'information entre vecteurs distants (mécanisme de routage) ;
- le Perceptron Multicouche (MLP) applique une transformation non linéaire **intra-position**, traitant chaque vecteur indépendamment dans son espace spectral.

### Formalisme Pre-Norm et décomposition linéaire

Dans la configuration *Pre-Norm*, prédominante dans les architectures modernes pour sa stabilité d’entraînement (conditionnement des gradients et contrôle de l’échelle des activations), la dynamique du bloc $l$ est régie par :

$$
\begin{aligned}
X'^{(l)} &= X^{(l)} + \Delta X^{(l)}_{\mathrm{MHA}},
&\qquad \Delta X^{(l)}_{\mathrm{MHA}} &= \mathrm{MHA}(\mathrm{Norm}_1(X^{(l)})),\\
X^{(l+1)} &= X'^{(l)} + \Delta X^{(l)}_{\mathrm{MLP}},
&\qquad \Delta X^{(l)}_{\mathrm{MLP}} &= \mathrm{MLP}(\mathrm{Norm}_2(X'^{(l)})).
\end{aligned}
$$

où $\mathrm{Norm}$ désigne une normalisation de type $\mathrm{LayerNorm}$ ou $\mathrm{RMSNorm}$ selon l’implémentation.

L'application de la normalisation en entrée de sous-bloc dissocie l'amplitude du flux résiduel de celle des mises à jour.

Cette architecture permet une lecture télescopique du réseau. En faisant abstraction des effets de bord (tels que le dropout), l'état final $X^{(L)}$ s'exprime comme la somme de l'état initial et de l'ensemble des interventions intermédiaires :

$$
X^{(L)} = X^{(0)} + \sum_{l=0}^{L-1}\Big(\Delta X^{(l)}_{\mathrm{MHA}} + \Delta X^{(l)}_{\mathrm{MLP}}\Big).
$$

Cette formulation met en exergue une propriété géométrique cruciale : le flux résiduel ne constitue pas une succession d’états hiérarchiques rigides, mais une accumulation linéaire de vecteurs. L’état initial $X^{(0)}$ n’est jamais écrasé ; il persiste en tant qu'ancrage sémantique de la représentation finale, dont la trajectoire est simplement orientée par la superposition des contributions additives de chaque bloc.

<figure class="cm-figure">
  <img src="/assets/img/art1/Figure_2_2.png" alt="Diagramme illustrant l'accumulation du flux résiduel dans un Transformer.">
  <figcaption>
    Fig. 2 — L’“autoroute” résiduelle : l’état initial $X^{(0)}$ demeure présent, tandis que chaque sous-couche ajoute une mise à jour $\Delta X$ au flux résiduel.
  </figcaption>
</figure>

### Propriétés mécanistes et implications pour la sûreté offensive

#### Additivité stricte : persistance et interférence

L'architecture résiduelle se distingue par l'absence d'opérateur de soustraction explicite ou de mécanisme de réinitialisation (*reset gate*). Aucune sous-couche ne possède la faculté d'effacer explicitement une information du flux $X$. Toute disparition apparente d'un concept correspond en réalité à une **compensation géométrique** (une forme d'interférence destructive) et non à un effacement structurel. Une contrainte initiale (e.g., system prompt de sécurité) injectée en $l=0$ persiste comme composante de la somme, à moins d'être neutralisée par l'ajout d'un vecteur opposé de norme comparable.

**Conséquence opérationnelle** : Le contournement adversarial (jailbreak) n’efface pas la contrainte ; il déplace la résultante vectorielle vers une région de l’espace latent où la projection sur la direction associée au refus devient négligeable, soit parce qu’elle est faible, soit parce qu’elle devient orthogonale aux axes de lecture mobilisés par les couches ultérieures.

#### Normalisation et primauté de l’orientation

Dans l’architecture *Pre-Norm*, les fonctions de transition (Attention, MLP) ne sont pas appliquées directement à l’état résiduel, mais conditionnées par une version normalisée de celui-ci. En notant $x \equiv x_t^{(l)} \in \mathbb{R}^{1\times d_{\text{model}}}$ le vecteur d’état courant, l’entrée effective des sous-couches, illustrée avec **RMSNorm**, est :

$$
\tilde x = \mathrm{RMSNorm}(x) = \frac{x}{\lVert x \rVert_{\text{RMS}}} \odot \gamma,
\qquad
\lVert x \rVert_{\text{RMS}} = \sqrt{\frac{1}{d_{\text{model}}}\sum_{r=1}^{d_{\text{model}}} x_r^2 + \varepsilon},
$$

où $\gamma \in \mathbb{R}^{1\times d_{\text{model}}}$ est le vecteur de gain et $\varepsilon$ un terme de stabilisation. (Le même point — lecture d’une version normalisée du résiduel — vaut qualitativement pour **LayerNorm**, qui applique en plus un **centrage** $x \mapsto x-\mu(x)$.)

Cette transformation réalise une normalisation radiale (au sens RMS) suivie d’une mise à l’échelle anisotrope. Hors régime dominé par $\varepsilon$ (i.e. pour $\lVert x \rVert_{\text{RMS}} \gg \sqrt{\varepsilon}$), l’opérateur $\mathrm{RMSNorm}$ est quasi-invariant par homothétie positive :

$$
\forall \alpha>0,\quad \mathrm{RMSNorm}(\alpha x)\approx \mathrm{RMSNorm}(x).
$$

En conséquence, la réponse des sous-couches dépend principalement de la **structure directionnelle** de $x$ (angles / proportions relatives), tandis que l’amplitude globale est en grande partie factorisée en amont du calcul des activations.

**Découplage fonctionnel (conditionnement vs accumulation).**  
Une dissociation formelle apparaît entre signal de conditionnement et variable d’état :
- le **chemin de conditionnement** $\tilde x$ est soumis à une contrainte d’échelle (normalisation radiale, puis mise à l’échelle par $\gamma$) ;
- le **chemin d’état** $x$ évolue dans l’espace euclidien via une mise à jour additive.

En particulier, pour un sous-bloc générique, la mise à jour s’écrit :

$$
\Delta x = f(\tilde x), \qquad x \leftarrow x + \Delta x,
$$

où $f$ désigne l’opérateur de sous-couche (Attention ou MLP) évalué sur l’entrée normalisée.

La variable d’état $x$ n’est donc pas directement régulée par la normalisation ; seule la vue locale $\tilde x$ l’est. Cette structure autorise l’intégration de mises à jour additives sans contrainte explicite sur la norme de $x$, phénomène dont la dynamique est analysée dans la section suivante.


#### Dynamique inertielle et bras de levier relatif

À position $t$ fixée, l’état résiduel évolue par intégration additive de mises à jour produites par les sous-couches évaluées sur une vue normalisée. En posant $x^{(l)} \equiv x_t^{(l)}$ et $\Delta x^{(l)} \equiv \Delta x_t^{(l)}$, la récurrence s’écrit :

$$
x^{(l+1)} = x^{(l)} + \Delta x^{(l)}_{\mathrm{MHA}} + \Delta x^{(l)}_{\mathrm{MLP}}
\;\stackrel{\mathrm{def}}{=}\;
x^{(l)} + \Delta x^{(l)}.
$$

Cette écriture met en évidence une propriété géométrique : l’état $x^{(l)}$ agit comme une variable d’accumulation, tandis que $\Delta x^{(l)}$ est une perturbation locale. Lorsque les incréments ne se compensent pas systématiquement, $\lVert x^{(l)}\rVert$ tend à croître avec la profondeur (régime compatible avec une croissance de type $\Theta(\sqrt{l})$ sous hypothèses de faible corrélation). Dans ce cadre, l’effet d’une mise à jour sur l’**orientation** de l’état devient naturellement relatif à la norme déjà accumulée.

L’amplitude relative de la perturbation est mesurée par le **bras de levier relatif** :

$$
\rho^{(l)}=\frac{\lVert\Delta x^{(l)}\rVert}{\lVert x^{(l)}\rVert},
\qquad \lVert\cdot\rVert \text{ norme euclidienne }(L_2).
$$

La quantité pertinente pour la rotation n’est toutefois pas $\Delta x^{(l)}$ dans son ensemble mais sa composante orthogonale à $x^{(l)}$. En introduisant la décomposition :

$$
\Delta x^{(l)}=\Delta x^{(l)}_{\parallel}+\Delta x^{(l)}_{\perp},
\qquad
\Delta x^{(l)}_{\perp}\perp x^{(l)},
$$

la rotation angulaire entre $x^{(l)}$ et $x^{(l+1)}$ se contrôle directement via $\Delta x^{(l)}_{\perp}$. En notant $\theta^{(l)}$ l’angle entre $x^{(l)}$ et $x^{(l)}+\Delta x^{(l)}$, une identité géométrique (triangle vectoriel) donne :

$$
\sin\theta^{(l)}
=
\frac{\lVert\Delta x_\perp^{(l)}\rVert}{\lVert x^{(l)}+\Delta x^{(l)}\rVert}. \tag{1}
$$

L’égalité isole explicitement le mécanisme de rotation : seule une énergie injectée **hors de** $\mathrm{span}(x^{(l)})$ modifie la direction, tandis qu’une mise à jour colinéaire affecte principalement la norme. L’équation fournit donc un critère direct : la rotation est bornée par la taille de la composante orthogonale relativement à la taille du nouvel état.

Une borne exploitable indépendante de la direction fine de $\Delta x^{(l)}$ s’obtient en combinant (1) avec $\lVert\Delta x^{(l)}_{\perp}\rVert\le \lVert\Delta x^{(l)}\rVert$ et l’inégalité triangulaire $\lVert x^{(l)}+\Delta x^{(l)}\rVert\ge \lVert x^{(l)}\rVert-\lVert\Delta x^{(l)}\rVert$ :

$$
\sin\theta^{(l)}
\le
\frac{\lVert\Delta x^{(l)}\rVert}{\lVert x^{(l)}\rVert-\lVert\Delta x^{(l)}\rVert}
=
\frac{\rho^{(l)}}{1-\rho^{(l)}}\qquad(\rho^{(l)}<1). \tag{2}
$$

La borne (2) fournit un majorant explicite de la rotation angulaire induite par une mise à jour additive, indépendamment de l’orientation fine de $\Delta x^{(l)}$ (contrôle au pire cas). Lorsque $\rho^{(l)} \ll 1$, l’approximation petit-angle ($\theta^{(l)} \approx \sin\theta^{(l)}$) implique :

$$
\theta^{(l)} \;\lesssim\; \rho^{(l)},
$$

de sorte que l’effet directionnel d’une sous-couche est contraint : les incréments peuvent modifier l’état (notamment sa norme) mais ne peuvent réorienter $x^{(l)}$ que dans une mesure proportionnelle à $\rho^{(l)}$. Cela formalise l’existence d’un budget angulaire local gouvernant la capacité de correction en profondeur.


**Conséquence : régime inertiel (couche-dépendant).**

La diminution typique de $\rho^{(l)}$ avec la profondeur—lorsque $\lVert x^{(l)}\rVert$ croît plus régulièrement que $\lVert \Delta x^{(l)}\rVert$—implique une rigidification progressive de l’orientation de $x^{(l)}$. Ce phénomène est intrinsèquement **local** : $\lVert \Delta x^{(l)}\rVert$ dépend des poids (attention/FFN), du gating et du contenu, de sorte que certaines couches peuvent présenter un levier plus élevé que d’autres. Le formalisme via $\rho^{(l)}$ et $\Delta x^{(l)}_{\perp}$ capture précisément cette hétérogénéité : une réorientation substantielle requiert soit $\rho^{(l)}=\Omega(1)$, soit une perturbation à faible norme mais dont la composante orthogonale est alignée sur des directions de lecture particulièrement sensibles.

> **Note** : $\rho^{(l)}$ fournit un **proxy (et un majorant au pire cas)** du budget de rotation disponible à la couche $l$ ; la rotation effective est plus finement contrôlée par $\lVert \Delta x^{(l)}_\perp \rVert / \lVert x^{(l)} \rVert$.


### Synthèse : asymétrie de profondeur, superposition additive et interférences

Les résultats précédents se condensent en une contrainte géométrique locale sur la **réorientabilité** du flux résiduel. À la couche $l$, l’angle $\theta^{(l)}$ entre $x^{(l)}$ et $x^{(l+1)} = x^{(l)} + \Delta x^{(l)}$ est gouverné par la composante **orthogonale** $\Delta x_\perp^{(l)}$ (Eq. (1)), tandis que le levier global

$$
\rho^{(l)} = \frac{\lVert \Delta x^{(l)} \rVert}{\lVert x^{(l)} \rVert}
$$

n’induit qu’un contrôle majorant au pire cas via $\lVert \Delta x_\perp^{(l)} \rVert \le \lVert \Delta x^{(l)} \rVert$ (Eq. (2)). La capacité directionnelle de couche est donc naturellement caractérisée par le **budget angulaire**

$$
\rho_\perp^{(l)} \stackrel{\mathrm{def}}{=} \frac{\lVert \Delta x_\perp^{(l)} \rVert}{\lVert x^{(l)} \rVert},
$$

qui capture directement la marge de rotation disponible lorsque l’état accumulé $\lVert x^{(l)} \rVert$ domine la perturbation.

Cette lecture induit une **asymétrie de profondeur** :

* **Levier précoce.** Tant que $\lVert x^{(l)} \rVert$ demeure faible, une mise à jour modérée peut présenter un $\rho_\perp^{(l)}$ non négligeable et imposer une orientation conditionnant les activations ultérieures. Ce phénomène est accentué en régime *Pre-Norm*, où les sous-couches opèrent sur une vue normalisée dont la dépendance est principalement **directionnelle**.

* **Inertie tardive.** Lorsque $\lVert x^{(l)} \rVert$ devient grande relativement aux incréments typiques, $\rho_\perp^{(l)}$ se contracte : les couches tardives peuvent encore accumuler du signal, mais une correction directionnelle substantielle requiert soit $\lVert \Delta x^{(l)} \rVert=\Omega(\lVert x^{(l)} \rVert)$, soit une composante $\Delta x_\perp^{(l)}$ finement orientée vers des directions présentant un **gain élevé au readout** (et, plus généralement, une forte sensibilité au niveau des lectures pertinentes).

La dynamique se comprend alors comme une **superposition additive** dans un espace commun : l’état final résulte de la somme des contributions injectées couche après couche, et l’absence de mécanisme d’effacement implique qu’une “suppression” apparente d’un concept relève d’une **interférence destructive** (compensation géométrique) plutôt que d’une suppression structurale.

Formulé au niveau du décodage, si une projection linéaire de sortie $y = x\,W_U$ (p. ex. unembedding/logits), avec $x\in\mathbb{R}^{1\times d_{\text{model}}}$ et $W_U\in\mathbb{R}^{d_{\text{model}}\times \|\mathcal V\|}$, est sensible au sous-espace $\mathrm{Col}(W_U)$ (espace des colonnes), l’efficacité fonctionnelle d’une mise à jour dépend de sa **projection** sur ce sous-espace :

* une contrainte (p. ex. une *feature* de refus) n’est pas retirée de $x$, mais peut devenir **inopérante** si la somme des mises à jour annule sa projection pertinente sur $\mathrm{Col}(W_U)$ (i.e. $\Delta x\,W_U \approx -x\,W_U$ sur les composantes décisionnelles) ;

* inversement, des composantes ajoutées majoritairement **hors** des directions effectivement lues (faible projection sur $\mathrm{Col}(W_U)$, ou sur les directions auxquelles les sous-couches suivantes sont sensibles après normalisation) peuvent être présentes dans $x$ tout en produisant un effet marginal sur la sortie.

En conséquence, un contournement adversarial n’efface pas une contrainte injectée dans la somme résiduelle ; il consiste à déplacer la résultante vers une région où la projection de cette contrainte sur les **axes décisifs** — ceux de l’unembedding (logits) et ceux auxquels les opérateurs des couches suivantes sont sensibles — est compensée ou devient négligeable, malgré l’inertie induite par la contraction de $\rho_\perp^{(l)}$.


<hr style="width:40%; margin:auto;">

### Transition : du substrat aux opérateurs

La question se réduit alors à une contrainte de contrôlabilité : quels sous-espaces de $\Delta x^{(l)}$ sont effectivement accessibles à l’attention et aux MLP (par tête/couche), et comment ces sous-espaces bornent à la fois la capacité de rotation (via $\rho_\perp^{(l)}$) et l’influence sur les directions effectivement **lues** (couches ultérieures et décodage)

---


## 1.3 Mécanique des opérateurs : mélange temporel et mélange de canaux

Le flux résiduel ayant été établi comme un substrat additif (cf. §1.2), l’analyse se concentre sur les opérateurs qui redistribuent l’information entre positions et transforment localement les représentations. Dans l’approche d’interprétabilité mécaniste, un bloc Transformer s’analyse comme l’alternance de deux mécanismes distingués par l’axe tensoriel sur lequel ils opèrent (Elhage et al., 2021) :

- **Attention (mélange temporel / routage inter-positions).** Opère sur l’axe de la séquence ($T$). Pour chaque tête $h$, elle construit une matrice $A^{(h)}(X)\in\mathbb{R}^{T\times T}$ (causale) qui définit, pour chaque position cible, une **distribution de mélange** sur les positions sources. Mécaniquement, les **queries/keys** déterminent le schéma de routage $A^{(h)}$, puis les **values** fournissent le contenu effectivement agrégé : l’opérateur réalise ainsi une forme de **copie/mélange** inter-positions dépendante du contenu, en important dans le résiduel des directions latentes provenant d’autres tokens.

- **MLP/FFN (mélange de canaux / transformation locale).** Opère à position fixée sur l’axe $d_{\text{model}}$ (espace des *features*), indépendamment des autres tokens. Il implémente une application non linéaire *point par point* $x \mapsto \mathrm{MLP}(x)$ : typiquement une expansion linéaire, une non-linéarité (souvent **gated** dans les variantes modernes, p. ex. (GE)GLU/SwiGLU), puis une reprojection vers $d_{\text{model}}$ avant l’ajout résiduel. Dans une lecture mécaniste, on peut l’interpréter comme une **écriture conditionnelle** dans l’état latent : l’entrée sélectionne via le gating quelles directions internes s’activent, et la sortie combine des patrons de mise à jour “stockés” dans les poids pour ajuster la représentation du token (Geva et al., 2021).

Cette dichotomie structure l’analyse de sûreté : le mélange temporel gouverne l’**accessibilité causale** de l’information (quelles sources deviennent disponibles à une position donnée), tandis que le mélange de canaux gouverne sa **compilation** en variables latentes décodables (comment cette information se convertit en **logits de sortie**, puis en distribution via Softmax).

### Rappel : lecture et écriture sur un même flux

Pour $X^{(l)}\in\mathbb{R}^{T\times d_{\text{model}}}$, les deux opérateurs sont évalués sur une vue normalisée de l’état courant, puis produisent des mises à jour additives écrites dans le flux résiduel (cf. §1.2) :

$$
X^{(l+1)} \;=\; X^{(l)} \;+\; \Delta X^{(l)}_{\mathrm{MHA}} \;+\; \Delta X^{(l)}_{\mathrm{MLP}}.
$$

<p>
Ici, $\Delta X^{(l)}_{\mathrm{MHA}}$ est induite par le routage inter-positions, et $\Delta X^{(l)}_{\mathrm{MLP}}$ par la transformation locale. Les sections suivantes détaillent la construction de $A^{(h)}(X)$ via $Q/K$, masque causal et softmax, et la décomposition clé–valeur du FFN comme opérateur d’écriture dans le résiduel.
</p>


<hr style="width:40%; margin:auto;">


### Mélange temporel : l’attention comme opérateur de mélange causal conditionné par l’état courant

L’attention multi-têtes (MHA) associe, pour chaque tête $h$ et chaque position cible $t$, une distribution de poids
$\\{A^{(h)}\_{t,j}\\}_{j\le t}$. Cette distribution induit un **mélange inter-positions** : la représentation en $t$
reçoit une combinaison pondérée des contributions issues des positions sources.

Le mécanisme est **causal** : $A^{(h)}$ est contrainte à avoir un support uniquement sur $j\le t$, et vérifie
$$
A^{(h)}_{t,j}=0\quad \text{pour tout } j>t,
$$
de sorte que l’agrégation en $t$ ne dépend que des positions antérieures.

Il est **conditionné par l’état courant** : la matrice $A^{(h)}$ dépend des représentations $\tilde X^{(l)}$ à la couche $l$ et varie donc avec le contenu du contexte, plutôt que de réaliser un mélange à coefficients fixes.

À la couche $l$, l’opérateur s’applique à une vue normalisée du flux résiduel :

$$
\tilde X^{(l)}=\mathrm{Norm}\!\big(X^{(l)}\big)\in\mathbb R^{T\times d_{\text{model}}}.
$$

La suite décrit la construction de $A^{(h)}$ via le circuit QK, le transfert et l’écriture dans l’espace résiduel via le circuit OV, puis des motifs mécanistes de composition inter-couches (p. ex. induction, puits d’attention) pertinents pour l’analyse adversariale.


#### Matrices effectives dans l’espace résiduel : adressage et transfert

Une tête d’attention $h$ opère sur le flux résiduel normalisé
$\tilde X^{(l)}\in\mathbb{R}^{T\times d_{\text{model}}}$ en le factorisant en trois familles de vecteurs via des
projections linéaires distinctes :

$$
Q^{(h)}=\tilde X^{(l)}W_Q^{(h)},\qquad
K^{(h)}=\tilde X^{(l)}W_K^{(h)},\qquad
V^{(h)}=\tilde X^{(l)}W_V^{(h)},
$$

avec $W_Q^{(h)},W_K^{(h)}\in\mathbb{R}^{d_{\text{model}}\times d_k}$ et
$W_V^{(h)}\in\mathbb{R}^{d_{\text{model}}\times d_v}$, d’où

$$
Q^{(h)},K^{(h)}\in\mathbb{R}^{T\times d_k},\qquad
V^{(h)}\in\mathbb{R}^{T\times d_v}.
$$

Cette factorisation impose une **dissociation structurelle** entre un espace de **score**
$\mathbb{R}^{d_k}$ (porté par $Q$ et $K$) et un espace de **contenu** $\mathbb{R}^{d_v}$ (porté par $V$).
L’adressage inter-positions est déterminé exclusivement par la géométrie dans $\mathbb{R}^{d_k}$, tandis que le signal
effectivement transféré est déterminé dans $\mathbb{R}^{d_v}$ puis ré-encodé dans l’espace résiduel.

Les matrices de projection $W_Q^{(h)}$ et $W_K^{(h)}$ construisent l’espace de comparaison (ou espace de score) de la tête.
Pour une cible $t$, le vecteur $Q_t^{(h)}=\tilde x_t^{(l)}W_Q^{(h)}$ définit une direction de référence dans $\mathbb{R}^{d_k}$.
Pour chaque source $j$, le score $Q_t^{(h)}(K_j^{(h)})^\top$ mesure la composante de $K_j^{(h)}$ le long de la direction $Q_t^{(h)}$ (projection scalaire non normalisée). Cette mesure est modulée par les normes des vecteurs et ajustée par le facteur d'échelle $1/\sqrt{d_k}$.
La softmax convertit ensuite ces scores (après masquage $M$) en une distribution de mélange $A^{(h)}_{t,\cdot}$.
La matrice résultante $A^{(h)}$, souvent appelée **pattern d'attention**, constitue un opérateur de pondération indépendant du contenu transporté, qui sera appliqué aux valeurs $V^{(h)}$ via l'opération $H^{(h)}=A^{(h)}V^{(h)}$.

À l’inverse, $W_V^{(h)}$ paramétrise le message associé à chaque source dans l’espace de contenu $\mathbb{R}^{d_v}$.
Il en résulte que les variables qui déterminent la masse attentionnelle $A^{(h)}_{t,j}$
(coordonées $Q/K$ dans $\mathbb{R}^{d_k}$) sont, en général, distinctes des variables qui déterminent le contenu
transféré (coordonnées $V$ dans $\mathbb{R}^{d_v}$). Dans l’analyse des flux, cette dissociation implique qu’une source
peut être fortement pondérée par le mécanisme d’indexation tout en injectant, via $V$, un signal dont la nature (au sens
des directions écrites après $W_O^{(h)}$) n’est pas contrainte par les seules caractéristiques qui ont déterminé
l’adressage.

Le routage est défini par des logits et une normalisation softmax ligne par ligne :

$$
S^{(h)}=\frac{Q^{(h)}(K^{(h)})^\top}{\sqrt{d_k}}+M,\qquad
A^{(h)}=\mathrm{softmax}_{\text{ligne}}(S^{(h)}),
$$

où $M$ encode les contraintes (p.\,ex. causalité). Pour une cible $t$ fixée, la ligne $S^{(h)}_{t,\cdot}$ mesure la
compatibilité entre $Q_t^{(h)}$ et les clés $\{K_j^{(h)}\}$ par produit scalaire dans $\mathbb{R}^{d_k}$ :

$$
S^{(h)}_{t,j}-M_{t,j}=\frac{Q_t^{(h)}(K_j^{(h)})^\top}{\sqrt{d_k}}
=\frac{\|Q_t^{(h)}\|\,\|K_j^{(h)}\|\cos\theta_{t,j}}{\sqrt{d_k}}.
$$

Le terme directionnel $\cos\theta_{t,j}$ discrimine les sources par alignement dans l’espace de score. Les normes
$\|Q_t^{(h)}\|$ et $\|K_j^{(h)}\|$ contrôlent l’échelle des logits : à $t$ fixé, une augmentation de $\|Q_t^{(h)}\|$
amplifie les écarts $S_{t,j}-S_{t,k}$ et tend à concentrer $A^{(h)}_{t,\cdot}$, tandis que des normes plus faibles
tendent à diffuser la masse sur davantage de sources. Le facteur $1/\sqrt{d_k}$ stabilise l’échelle typique des
produits scalaires lorsque $d_k$ varie.

Une fois $A^{(h)}$ déterminée, le transfert de contenu ne dépend plus de $Q$ ni de $K$ : il consiste à agréger les
valeurs dans $\mathbb{R}^{d_v}$,

$$
H^{(h)}=A^{(h)}V^{(h)}\in\mathbb{R}^{T\times d_v},\qquad
H_t^{(h)}=\sum_j A^{(h)}_{t,j}\,V_j^{(h)}\in\mathbb{R}^{1\times d_v}.
$$

Comme $A^{(h)}_{t,\cdot}$ est une distribution, $H_t^{(h)}$ est un barycentre des $V_j^{(h)}$ sur le support autorisé
par $M$ : une distribution piquée privilégie un petit nombre de sources, tandis qu’une distribution diffuse produit une
moyenne pondérée de nombreuses sources.

Le message agrégé est enfin ré-encodé dans l’espace résiduel via une projection de sortie :

$$
O^{(h)}=H^{(h)}W_O^{(h)}\in\mathbb{R}^{T\times d_{\text{model}}},
\qquad
W_O^{(h)}\in\mathbb{R}^{d_v\times d_{\text{model}}},
$$

soit, en une seule expression,

$$
O^{(h)} = A^{(h)}\,V^{(h)}\,W_O^{(h)}.
$$

La non-linéarité de la tête est entièrement portée par la softmax (construction de $A^{(h)}$) ; conditionnellement à
$A^{(h)}$, l’agrégation $A^{(h)}V^{(h)}$ et la projection $W_O^{(h)}$ sont linéaires.

Cette séparation $(Q,K)$ vs $V$ est centrale pour l’analyse des flux : les variables qui déterminent la sélection des
sources (géométrie dans $\mathbb{R}^{d_k}$) sont, en général, distinctes des variables qui déterminent le contenu ré-encodé dans l’état de la cible (géométrie dans $\mathbb{R}^{d_v}$). 

Il est crucial de noter que typiquement $d_k, d_v \ll d_{\text{model}}$. Par conséquent, bien que les vecteurs d'entrée et de sortie résident dans $\mathbb{R}^{d_{\text{model}}}$, les opérations internes de la tête passent par un **goulot d’étranglement de rang faible** (*low-rank bottleneck*).

Géométriquement, cela implique que la tête ne peut lire l'information (pour calculer l'attention) que si elle se trouve dans le sous-espace engendré par les lignes de $W_Q$ et $W_K$, et ne peut écrire de l'information (via $O^{(h)}$) que dans le sous-espace engendré par les colonnes de $W_O$. La tête agit donc comme un filtre qui projette, traite dans une dimension réduite, puis réinjecte le résultat dans une direction spécifique de l'espace résiduel.

L’analyse par circuits formalise ce découplage en regroupant les projections en deux opérateurs effectifs,
$W_{QK}^{(h)}=W_Q^{(h)}(W_K^{(h)})^\top$ et $W_{OV}^{(h)}=W_V^{(h)}W_O^{(h)}$,
étudiés dans la section suivante.


#### Opérateurs effectifs dans l’espace résiduel : goulots d’adressage (QK) et d’écriture (OV)

Dans l’espace résiduel $\mathbb{R}^{d_{\text{model}}}$, une tête d’attention $h$ peut être caractérisée par deux
opérateurs effectifs obtenus par composition des projections internes. Cette représentation explicite deux contraintes
structurelles distinctes : un goulot d’**adressage**, qui borne la complexité des compatibilités utilisées pour
former $A^{(h)}$, et un goulot d’**écriture**, qui borne le sous-espace des mises à jour injectées dans le flux
résiduel.

Le premier correspond au circuit **QK** (construction des logits puis de la topologie de routage), le second au
circuit **OV** (transfert du contenu agrégé et ré-encodage dans le résiduel). La suite introduit les matrices
effectives $W_{QK}^{(h)}=W_Q^{(h)}(W_K^{(h)})^\top$ et $W_{OV}^{(h)}=W_V^{(h)}W_O^{(h)}$, puis analyse les contraintes de
rang et les sous-espaces induits.


#### Circuit QK : adressage bilinéaire, compétition normalisée et topologie de routage

Le circuit **QK** paramétrise les logits $S^{(h)}$ et donc la matrice d’attention $A^{(h)}$, qui pondère les
sources admissibles $j\le t$ pour chaque position cible $t$. Il fixe ainsi la topologie effective du routage
inter-positions, indépendamment du contenu transféré par le circuit **OV**.

Pour une représentation $\tilde X^{(l)}$, les logits s’écrivent :

$$
S_{t,j}^{(h)}=\frac{\tilde x_t^{(l)}\,W_{QK}^{(h)}\,(\tilde x_j^{(l)})^\top}{\sqrt{d_k}}+M_{t,j},
\qquad
M_{t,j}=
\begin{cases}
  -\infty & j>t,\\
  0 & \text{sinon.}
\end{cases}
$$

Les poids d’attention résultent d’une normalisation ligne par ligne :

$$
A^{(h)}=\mathrm{softmax}_{\text{ligne}}(S^{(h)}),
\qquad
A^{(h)}_{t,j}\ge 0,\quad \sum_{j\le t}A^{(h)}_{t,j}=1,\quad A^{(h)}_{t,j}=0\ \text{si }j>t.
$$

La causalité ($j\le t$) induit une structure acyclique sur les dépendances $t\leftarrow j$, pondérées par
$A^{(h)}_{t,j}$.

##### Compatibilité bilinéaire et goulot d’adressage

En posant

$$
Q_t^{(h)}=\tilde x_t^{(l)}W_Q^{(h)},\qquad K_j^{(h)}=\tilde x_j^{(l)}W_K^{(h)},
$$

le score prend la forme standard :

$$
S_{t,j}^{(h)}=\frac{\langle Q_t^{(h)},K_j^{(h)}\rangle}{\sqrt{d_k}}+M_{t,j},
\qquad
\langle Q_t^{(h)},K_j^{(h)}\rangle = Q_t^{(h)}(K_j^{(h)})^\top.
$$

Le produit scalaire admet la décomposition géométrique :

$$
\langle Q_t,K_j\rangle=\|Q_t\|\,\|K_j\|\,\cos\theta_{t,j},
$$

où $\cos\theta_{t,j}$ capture l’alignement directionnel dans $\mathbb{R}^{d_k}$, tandis que $\|Q_t\|$ et $\|K_j\|$
contrôlent l’échelle des logits. À $t$ fixé, l’échelle des écarts $S_{t,j}-S_{t,k}$ gouverne directement la
concentration de $A^{(h)}_{t,\cdot}$ après softmax.

La contrainte structurelle majeure provient de la bilinéarité :

$$
W_{QK}^{(h)}=W_Q^{(h)}(W_K^{(h)})^\top,\qquad \mathrm{rang}(W_{QK}^{(h)})\le d_k.
$$

Les décisions d’adressage ne dépendent donc de $\tilde x_t^{(l)}$ et $\tilde x_j^{(l)}$ qu’au travers de leurs
coordonnées projetées dans un espace de dimension $\le d_k$, ce qui introduit un goulot d’adressage. En particulier,
des sources distinctes peuvent devenir difficilement séparables pour cette tête lorsque leurs clés projetées
$K_j^{(h)}$ sont géométriquement proches dans $\mathbb{R}^{d_k}$ (collisions dans l’espace de score).

> **Note (aliasage / leurres attentionnels).** Une collision dans l’espace de score permet qu’un motif injecté
> produise des clés $K$ proches de celles associées à des motifs fréquemment sélectionnés par la tête (p.\,ex. formats,
> séparateurs, ou instructions de haut niveau). Le routage devient alors peu discriminant à l’échelle de la tête, même
> lorsque le contenu porté par $V$ diffère fortement.

##### Softmax : compétition sur simplex et amplification des écarts

La softmax transforme les logits en une distribution sur le simplex $\sum_{j\le t}A_{t,j}=1$. Deux propriétés
mécaniques structurent le routage :

* **Invariance par translation** : pour tout $c$, $\mathrm{softmax}(S_{t,\cdot})=\mathrm{softmax}(S_{t,\cdot}+c\mathbf 1)$,
  de sorte que seules les différences $S_{t,j}-S_{t,k}$ importent.


* **Compétition à budget unitaire** : la contrainte $\sum_{k\le t} A_{t,k}=1$ implique qu’augmenter $S_{t,j}$
  accroît $A_{t,j}$ et réduit mécaniquement les masses $A_{t,k}$ pour $k\neq j$.


L’amplification exponentielle induit un régime de concentration lorsque l’écart
$\Delta=S_{t,j^\*}-\max_{k\neq j^*}S_{t,k}$ devient grand. En isolant $j^\*$ :
$$
A_{t,j^*} = \frac{e^{S_{t,j^*}}}{\sum_{k\le t} e^{S_{t,k}}}
= \frac{1}{1+\sum_{k\le t, k\neq j^*} e^{-(S_{t,j^*}-S_{t,k})}}.
$$
Ainsi, si $S_{t,j^\*}-S_{t,k}\gg 0$ pour tout $k\neq j^\*$, alors $A_{t,j^\*}\approx 1$.
Le facteur $1/\sqrt{d_k}$ stabilise l’échelle typique des produits scalaires et limite une concentration artificielle
lorsque $d_k$ augmente.

> **Note (écrasement relatif / saturation attentionnelle).** Une configuration où une source $j^\*$ monopolise
> $A^{(h)}_{t,\cdot}$ induit un écrasement relatif des autres signaux \emph{pour cette tête} : même si des sources
> alternatives restent compatibles, leur contribution devient négligeable dans $H_t^{(h)}=\sum_j A_{t,j}^{(h)}V_j^{(h)}$.
> En contexte adversarial, ce mécanisme se manifeste comme une forme de \emph{saturation} du budget attentionnel local,
> où une forte disparité de logits rend l’agrégation quasi-monocanal.
> Deux mécanismes distincts peuvent conduire à cette saturation : une augmentation de $\|Q_t^{(h)}\|$, qui amplifie 
> globalement les écarts de logits sur la ligne $t$ (gain de ligne), et une augmentation de $\|K_{j^\*}^{(h)}\|$,
> qui accroît sélectivement les logits associés à une source $j^\*$ et favorise une concentration sur celle-ci.


##### Contrainte causale et biais positionnels

Le masque causal $M_{t,j}$ impose une contrainte topologique dure : seules les arêtes $t\leftarrow j$ avec $j\le t$
sont admissibles. Les encodages relatifs (RoPE, ALiBi) modifient ensuite la géométrie des logits en fonction de
l’offset $t-j$, sans changer la forme générale “compatibilité + softmax”.

Pour RoPE, le produit scalaire s’interprète comme une similarité après une rotation relative dépendant de $t-j$ :

$$
Q_t^{(h)}(K_j^{(h)})^\top = \hat Q_t^{(h)}\,\big(R_{t-j}\big)\,(\hat K_j^{(h)})^\top,
$$

où $R_{t-j}$ désigne une transformation orthogonale qui encode la distance relative. Contrairement à un biais additif (comme ALiBi), RoPE préserve la norme des vecteurs mais modifie leur alignement angulaire $\cos\theta_{t,j}$ en fonction de la distance.

Dans de nombreuses configurations, ces mécanismes induisent un a priori de proximité : à compatibilité de contenu
comparable, des sources plus proches tendent à être favorisées. Ce biais reste toutefois un prior et peut être contredit
par un signal de contenu dominant.

> **Note (biais de récence).** Un a priori de proximité renforce structurellement l’influence des segments récents
> du contexte. Dans une lecture sûreté, cela accentue la difficulté de maintenir une contrainte injectée loin en amont
> lorsque des motifs concurrents apparaissent près de la zone de génération.

Le circuit QK fixe ainsi une carte de lecture : une topologie de dépendances causales pondérées $A^{(h)}$, issue d’un
compromis entre compatibilité géométrique ($Q/K$), contraintes structurelles (masque, rang) et compétition normalisée
(softmax). Une fois cette topologie déterminée, le circuit \textbf{OV} réalise le transfert du contenu agrégé et
l’écriture correspondante dans le flux résiduel.


#### Circuit OV : lecture des valeurs et canal d’écriture contraint

Le circuit **OV** réalise le transfert de contenu une fois la topologie de routage fixée par **QK**.
Conditionnellement à la matrice d’attention $A^{(h)}$, une tête $h$ associe à chaque source $j$ une mise à jour dans
l’espace résiduel, puis agrège ces mises à jour selon les poids d’attention. L’analyse se déplace ainsi de la
*sélection* (structure de $A^{(h)}$) vers la *capacité d’écriture* (directions accessibles dans le résiduel).

Pour une tête $h$, les valeurs et les mises à jour s’écrivent :

$$
V^{(h)}=\tilde X^{(l)}W_V^{(h)}\in\mathbb{R}^{T\times d_v},
\qquad
U^{(h)}=V^{(h)}W_O^{(h)}\in\mathbb{R}^{T\times d_{\text{model}}},
$$

où la ligne $u_j^{(h)}\in\mathbb{R}^{1\times d_{\text{model}}}$ est la contribution associée à la source $j$ :

$$
u_j^{(h)}=\tilde x_j^{(l)}\,W_V^{(h)}W_O^{(h)}.
$$

La sortie de la tête à la position cible $t$ est alors :

$$
O_t^{(h)}=\sum_{j\le t}A_{t,j}^{(h)}\,u_j^{(h)},
\qquad
O^{(h)}=A^{(h)}U^{(h)}=A^{(h)}\tilde X^{(l)}W_V^{(h)}W_O^{(h)}.
$$

Cette écriture rend explicite la séparation des rôles : $u_j^{(h)}$ est déterminé par la source (indépendamment de $t$),
tandis que $A_{t,\cdot}^{(h)}$ fixe la combinaison agrégée. Une concentration du routage (via **QK**) ne suffit donc
pas à garantir un effet marqué : l’impact dépend de la structure du canal d’écriture $W_V^{(h)}W_O^{(h)}$.

##### Goulot d’écriture : sous-espace de sortie et rang effectif

L’opérateur effectif d’écriture d’une tête est la composition :

$$
W_{OV}^{(h)} = W_V^{(h)}W_O^{(h)}\in\mathbb{R}^{d_{\text{model}}\times d_{\text{model}}},
\qquad
\mathrm{rang}(W_{OV}^{(h)})\le d_v.
$$

Cette contrainte de rang ($d_v \ll d_{\text{model}}$) borne la capacité d’écriture : dans la convention vecteur-ligne
($xW$), toute écriture produite par la tête appartient au sous-espace de sortie

$$
\mathcal S_{\text{out}}^{(h)} \;=\; \mathrm{Row}\!\big(W_{OV}^{(h)}\big)
\;=\; \mathrm{Im}\!\big((W_{OV}^{(h)})^\top\big),
\qquad
\dim(\mathcal S_{\text{out}}^{(h)})\le d_v.
$$

Même si le routage $A^{(h)}_{t,\cdot}$ est fortement concentré (p.,ex. $A^{(h)}_{t,j^\*}\approx 1$), la tête ne peut écrire dans le flux résiduel que dans $\mathcal S_{\text{out}}^{(h)}$. La concentration de la matrice d'attention $A$ détermine quelle source domine l'entrée, mais ne modifie pas l’ensemble des directions accessibles en sortie, fixé structurellement par $W_{OV}^{(h)}$.

Au-delà de cette contrainte de sortie, $W_{OV}^{(h)}$ définit un filtrage linéaire entre les directions d’entrée et les directions écrites. Pour une variation $\delta x\in\mathbb{R}^{1\times d_{\text{model}}}$ sur une source, l’effet propagé sur le chemin OV est $\delta x\,W_{OV}^{(h)}$. Par le théorème du rang,

$$
\dim\ker(W_{OV}^{(h)})=d_{\text{model}}-\mathrm{rang}(W_{OV}^{(h)})\ge d_{\text{model}}-d_v,
$$

de sorte que lorsque $d_v\ll d_{\text{model}}$, une grande partie de l’espace d’entrée est projetée dans le noyau (variation nulle en sortie), indépendamment de la masse attentionnelle allouée.

Une analyse spectrale s’obtient via la décomposition en valeurs singulières (SVD) $W_{OV}^{(h)}=U\Sigma V^\top$ (de rang $r\le d_v$), soit

$$
W_{OV}^{(h)}=\sum_{k=1}^{r}\sigma_k\,u_k v_k^\top.
$$

Dans la convention vecteur-ligne ($\delta x W$), l'opération s'écrit :

$$\delta x\,W_{OV}^{(h)}=\sum_{k=1}^{r}\sigma_k\,(\delta x \cdot u_k)\,v_k^\top ,$$

où :

* $u_k$ (colonne de $U$) est une direction du sous-espace sensible (entrée).
* $v_k^\top$ (ligne de $V^\top$) est une direction du sous-espace d'écriture (sortie).
* Le produit scalaire $(\delta x\cdot u_k)$ mesure la projection de la variation incidente $\delta x$ sur la direction sensible $u_k$.


Chaque terme $k$ correspond à un canal singulier de $W_{OV}^{(h)}$ : la composante de $\delta x$ alignée avec $u_k$ est transmise avec un gain $\sigma_k$, puis réorientée en sortie selon $v_k^\top$ dans le résiduel.
Les grandes valeurs singulières $\sigma_k$ identifient les couples $(u_k, v_k)$ pour lesquels le couplage entrée-sortie est maximal. Réciproquement, si $\delta x$ est orthogonale à l'espace engendré par les $\{u_k\}_{\sigma_k>0}$, alors $\delta x\in\ker(W_{OV}^{(h)})$ : la tête est insensible à cette variation et l'écriture résultante est nulle.

> L’effet d’une source fortement pondérée par $A^{(h)}$ dépend donc de la convolution de trois facteurs : la masse attentionnelle, la projection du signal source sur les directions sensibles de fort gain (grandes $\sigma_k$), et l’alignement des directions écrites (dans $\mathcal S_{\text{out}}^{(h)}$) avec les sous-espaces lus par les couches avales ou le unbedding.

--- 

(check)
Soit $(u_k)*{k=1}^r$ une base orthonormée du sous-espace sensible, $\mathcal S=\mathrm{Im}(W*{OV}^{(h)\top})$ (p. ex. vecteurs singuliers gauche associés aux plus grandes valeurs singulières) : $\delta x,W_{OV}^{(h)}$ dépend uniquement de la projection de $\delta x$ sur $\mathcal S$. En particulier, si $\delta x\perp u_k$ pour tout $k$, alors $\delta x\in\ker(W_{OV}^{(h)})$ et $\delta x,W_{OV}^{(h)}=0$ ; la tête est aveugle à cette variation.

Une lecture spectrale s’obtient via la SVD $W_{OV}^{(h)}=U\Sigma V^\top$ (rang $r\le d_v$), soit

$$W_{OV}^{(h)}=\sum_{k=1}^{r}\sigma_k\,u_k v_k^\top.$$

Dans la convention vecteur-ligne ($\delta x W$),

$$
\delta x\,W_{OV}^{(h)}=\sum_{k=1}^{r}\sigma_k\,(\delta x u_k)\,v_k^\top ,
$$


où le coefficient scalaire ($\delta x\cdot u_k$) mesure la projection de la variation $\delta x$ sur la direction $u_k$ de l’espace d’entrée « vue » par la tête.

Chaque terme $k$ correspond à une composante singulière de $W_{OV}^{(h)}$ : la partie $\delta x$ alignée avec $u_k$ est transmise
avec un gain $\sigma_k$, puis elle apparaît en sortie comme une écriture dans le résiduel orientée selon $v_k^\top$.
Les grandes valeurs singulières $\sigma_k$ mettent ainsi en évidence les couples de directions $(u_k, v_k)$ pour lesquels la tête couple le plus fortement une direction d’entrée à une direction d’écriture. Réciproquement, si $\delta x$ est orthogonale à tous les $u_k$ associés à $\sigma_k>0$, alors $\delta x\in\ker(W_{OV}^{(h)})$ et $\delta x,W_{OV}^{(h)}=0$ : la tête est aveugle à cette variation.


> L’effet d’une source fortement pondérée par $A^{(h)}$ dépend non seulement de cette masse, mais aussi de la projection du signal source sur les directions d’entrée de grand gain (grandes $\sigma_k$) et de l’alignement des directions écrites (dans $\mathcal S_h$) avec les sous-espaces auxquels les couches suivantes et le readout sont sensibles.

(end check)

##### Linéarité conditionnelle, copie sélective et surface d’influence


Conditionnellement à $A^{(h)}$ (topologie fixée), l’opération OV est linéaire en $\tilde X^{(l)}$ :

$$
O^{(h)} = A^{(h)}\tilde X^{(l)}W_{OV}^{(h)}.
$$

Une perturbation additive $\Delta \tilde X^{(l)}$ dans les sources se propage alors linéairement :

$$
\Delta O^{(h)} = A^{(h)}\,\Delta\tilde X^{(l)}\,W_{OV}^{(h)}.
$$

Cette formulation isole la non-linéarité : elle est confinée dans la construction de $A^{(h)}$ via le circuit QK.
Une fois la matrice d’attention $A^{(h)}$ fixé, le chemin OV réalise un **transport linéaire** de contenu, contraint par le canal $W_{OV}^{(h)}$. Dans la convention vecteur-ligne ($xW$) l’écriture est au sous-espace engendré par les lignes de la matrice :

$$
\mathcal S_h=\mathrm{Row}!\big(W_{OV}^{(h)}\big)=\mathrm{Im}!\big((W_{OV}^{(h)})^\top\big)
$$

La tête implémente ainsi un schéma *collecte–agrégation* : chaque source $j$ produit un message
$u_j^{(h)}=\tilde x_j^{(l)}W_{OV}^{(h)}$, puis la cible $t$ agrège ces messages via les poids $A^{(h)}*{t,\cdot}$.
L’analyse se découple :
- **routage (QK)**  quelles arêtes $t\leftarrow j$ dominent (structure de $A^{(h)}$ - quels liens du graphe d'attention sont actifs) ?
- **transfert (OV)** : quelles directions sont effectivement transmises et avec quel gain (spectre de $W_{OV}^{(h)}$) ?


> **Note (régime concentré et copie sélective).** Lorsque le routage est fortement concentré sur une source $j^*$ (p. ex. $A^{(h)}*{t,j^*}\approx 1)$, on obtient
>
> $$
> O_t^{(h)}\approx u_{j^*}^{(h)}=\tilde x_{j^*}^{(l)}W_{OV}^{(h)}.
> $$
>
> Si, sur un sous-espace d’intérêt (composantes pertinentes pour la prédiction), l’action de $W_{OV}^{(h)}$ est proche
> d’une identité à une projection/échelle près (i.e. ces composantes sont largement préservées), alors la contribution
> écrite à $t$ **reproduit principalement** les composantes de la source : c’est une **copie sélective** (cas particulier
> de propagation). Cette intuition est mobilisée, par exemple, dans l’analyse des motifs d’inductions (cf. **têtes d'induction** - mécanisme de copier-coller contextuel).

La linéarité conditionnelle rend explicite la surface d’influence  d'une source. Pour qu’une perturbation $\Delta \tilde x_j^{(l)}$ impacte fortement une cible $t$ via la tête $h$, deux conditions doivent être réunies simultanément : 
* **capture attentionnelle (QK)** :  Le poids $A^{(h)}_{t,j}$ doit être significatif (le canal est ouvert).,
* **alignement OV** : La perturbation $\Delta \tilde x_j^{(l)}$ doit se projeter fortement sur les directions d’entrée à grand gain de $W_{OV}^{(h)}$ (les vecteurs singuliers $u_k$ associés aux grandes valeurs singulières $\sigma_k$). De sorte que $\|\Delta \tilde x_j^{(l)}W_{OV}^{(h)}\|$ soit grande
et orientée dans des directions effectivement exploitées en aval.

Si $\Delta \tilde x_j^{(l)}$ est orthogonale à ces directions sensibles, elle est mécaniquement filtrée par le canal OV (atténuation spectrale), même si l’attention portée à la source est maximale.

##### Superposition multi-têtes et interférences dans le résiduel

Au niveau de la sous-couche MHA, les contributions des têtes se superposent additivement dans le flux résiduel :

$$
X'^{(l)} = X^{(l)} + \sum_h O^{(h)}.
$$

Le résiduel étant un espace d’addition partagée, les écritures de différentes têtes peuvent s’additionner (interférence
constructive) ou se compenser (interférence destructive).

> **Note (Compétition vectorielle et Logits) :** L'effet final d'une tête ne se mesure pas uniquement à sa magnitude dans le résiduel, mais à sa projection sur la matrice de sortie (Unembedding) $W_U$. L'analyse de l'Attribution Directe aux Logits consiste à projeter $W_{OV}^{(h)}$ sur $W_U$ pour identifier quels tokens du vocabulaire sont promus ou inhibés par l'activation de cette tête, indépendamment des couches suivantes.

#### Motifs mécanistes récurrents (composition de circuits)

Les sections précédentes isolent la tête d’attention comme une unité fonctionnelle (adressage via **QK**, écriture via **OV**). Les comportements de plus haut niveau émergent surtout de la **composition inter-couches** : une écriture produite à la couche $l$ devient une composante de l’entrée $\tilde X^{(l')}$ lue à une couche $l'>l$ (après normalisation), et peut modifier soit le routage (QK) soit le contenu transmis (OV).

Deux modes de composition sont particulièrement structurants :

* **Composition de routage (pilotage QK).**
Une écriture amont $\Delta X^{(l)}$ modifie les états résiduels et donc les requêtes/clés produites en aval. Les logits $S_{t,j}^{(h',l')}$ et la distribution $A_{t,\cdot}^{(h',l')}$ peuvent ainsi être déplacés, reconfigurant la topologie effective des arêtes $t\leftarrow j$.

* **Composition de contenu (chaînage OV).**
Une écriture amont injecte une composante dans le résiduel qui est ensuite relayée via des chemins OV successifs, conditionnée par les routages $A^{(h)}$ et confinée par les sous-espaces d’écriture $\mathcal S_h=\mathrm{Im}(W_{OV}^{(h)})$.

Ces deux mécanismes se combinent et produisent des motifs récurrents. Deux motifs particulièrement documentés en interprétabilité mécaniste sont les **têtes d’induction** et les **puits d’attention** (attention sinks).

##### Têtes d’induction : complétion de motifs via un circuit inter-couches

Les têtes d’induction expliquent une complétion de motifs de type $A,B,\dots,A\mapsto B$ (répétition structurée en contexte). Le schéma canonique s’interprète comme une composition inter-couches en deux étapes, qui évite de postuler un « décalage » implicite du focus attentionnel.

** Étape amont (écriture d’un successeur local).**
Une tête $h_1$ à la couche $l$ produit une écriture $O^{(h_1,l)}$ telle que, pour des positions $j$, une composante du résiduel à $j$ devienne corrélée au token (ou à une signature) de la position $j+1$. Autrement dit, l’état à $j$ contient désormais une trace du successeur local, rendue accessible dans le résiduel par une écriture OV.

**Étape aval (adressage vers un précédent et copie via OV).**
Une tête $h_2$ à la couche $l'>l$ construit, à la position cible $t$, une requête $Q_t^{(h_2,l')}$ qui concentre $A_{t,\cdot}^{(h_2,l')}$ sur une occurrence passée $j$ dont la clé $K_j^{(h_2,l')}$ correspond à une signature analogue (p. ex. même $A$). Le transfert OV de $h_2$ copie alors vers $t$ une composante présente à $j$ ; si l’étape amont a injecté au résiduel de $j$ une information corrélée à $j+1$, la copie en aval réinjecte effectivement une information corrélée à $B$.

> Note (Amplification de régularités). Le motif d’induction amplifie des régularités présentes dans le contexte : un routage QK sélectionne des précédents structurellement similaires et un chemin OV réinjecte une continuation corrélée. Une vulnérabilité structurelle apparaît lorsque des régularités indésirables sont fortement représentées et se projettent sur les sous-espaces QK/OV effectivement mobilisés par ces têtes, ce qui peut imposer une continuation conforme à un patron au détriment d’autres signaux concurrents.


##### Puits d’attention : allocation de masse en régime peu discriminant

Le phénomène de attention sinks découle de la contrainte simplex imposée par la softmax : $\sum_{j\le t}A_{t,j}=1$. Lorsque les logits $S_{t,j}$ sont faiblement discriminants, la distribution tend vers une allocation diffuse ; l’agrégation OV devient alors une moyenne pondérée de nombreuses sources, susceptible d’injecter dans le résiduel un mélange de variance élevée et de faible spécificité.

Dans plusieurs architectures, une fraction stable de masse est allouée à un petit ensemble de positions (souvent BOS, séparateurs, ponctuation). Deux conditions rendent ce comportement stable :

* **Biais de routage (QK).** Les clés de ces positions et/ou des biais positionnels induisent une attractivité persistante lorsque le signal de contenu est faible.

* **Écriture à faible impact (OV).**
  Les valeurs associées, après projection $W_{OV}^{(h)}$, produisent une écriture de faible norme ou orientée vers des directions faiblement couplées aux lectures aval, de sorte qu’une allocation récurrente perturbe peu le résiduel.

Dans un régime de faible contraste, si une position $s$ reçoit systématiquement une masse dominante, la contribution s’approxime par :

$$O_t^{(h)} \approx A_{t,s}^{(h)}\,u_s^{(h)}.$$

> Note (Stabilisation et sensibilité). Une allocation récurrente vers un petit ensemble de positions stabilise le calcul lorsque les logits sont peu informatifs. La contrepartie est une sensibilité accrue à l’écriture associée à ces positions : toute variation de $u_s^{(h)}$ (après $W_{OV}^{(h)}$) se répercute sur de nombreuses cibles $t$ par le facteur $A_{t,s}^{(h)}$. Symétriquement, si l’attractivité QK de ces positions diminue, la masse peut se redistribuer sur un support plus large, augmentant la variance des écritures OV agrégées et dégradant la stabilité du routage.


##### Remarque : motifs composés et hétérogénéité des têtes

Ces motifs illustrent une idée générale : les propriétés saillantes résident dans des chaînes de composition « écriture amont $\rightarrow$ reconfiguration QK aval et/ou transport OV aval ». Une taxonomie plus large inclut des têtes de copie locales, des têtes d’ancrage sur délimiteurs, et des têtes de pilotage qui modulent le routage d’autres têtes via des écritures dirigées dans des sous-espaces auxquels les projections QK aval sont sensibles. L’identification mécaniste d’un motif revient à caractériser simultanément la topologie $A^{(h)}$ induite par QK et le sous-espace d’écriture $\mathcal S_h=\mathrm{Im}(W_{OV}^{(h)})$ induit par OV, puis à suivre leur composition à travers les couches.

<hr style="width:40%; margin:auto;">


### Mélange de canaux : le MLP comme mémoire associative

Contrairement à l’attention — qui réalise un **mélange inter-positions** en routant causalement l’information via les poids $A_{t,j}$ — le bloc **MLP/FFN** opère un **mélange intra-positionnel** : à chaque position $t$, il transforme $x_t^{(l)}\in\mathbb{R}^{d_{\text{model}}}$ sans consulter directement les autres tokens. À topologie contextuelle fixée par l’attention, le MLP implémente ainsi le **calcul local** sur la représentation : il projette $x_t^{(l)}$ dans un espace d’expansion de dimension $d_{\text{ff}}$ (typiquement $d_{\text{ff}}\gg d_{\text{model}}$), applique une non-linéarité jouant le rôle de **sélection conditionnelle** (souvent parcimonieuse en pratique), puis reprojette vers $\mathbb{R}^{d_{\text{model}}}$ afin d’injecter une mise à jour dans le flux résiduel.

Cette mécanique admet une lecture géométrique directe. L’expansion $d_{\text{model}}\to d_{\text{ff}}$ induit une base riche de **directions latentes** (features) dans laquelle l’état $x_t^{(l)}$ est recoordonné. La non-linéarité (et, dans les variantes *gated*, la modulation multiplicative) partitionne l’espace des entrées en **régimes** : selon la région où se situe $x_t^{(l)}$, un sous-ensemble de directions intermédiaires contribue fortement à l’update. La projection descendante recombine ensuite ces activations en un déplacement dans l’espace résiduel. Autrement dit, le MLP réalise une **transformation non-linéaire à régimes locaux** : localement, il se comporte comme une transformation quasi-affine dont les paramètres effectifs dépendent de l’état $x_t^{(l)}$, ce qui lui confère une capacité élevée de recombinaison de features à position fixe.

Dans cette vue unifiée, l’attention répond principalement à la question *où l’information circule* (connectivité, adressage, agrégation entre positions), tandis que le MLP répond à *comment cette information est transformée* au point $t$ (extraction de features, interactions non-linéaires entre canaux, synthèse d’une mise à jour dans le résiduel). L’alternance attention–MLP combine ainsi **transport contextuel** (mélange inter-tokens) et **calcul local** (recombinaison non-linéaire), ce qui motive la description du MLP comme processeur de features et **réservoir paramétrique** d’associations, complémentaire du rôle de lecture contextuelle assuré par l’attention.

#### Architecture : expansion, non-linéarité et *gating*

À la couche $l$, en régime **pre-norm**, le MLP s’applique au vecteur résiduel normalisé (typiquement **RMSNorm** ou **LayerNorm**) :

$$
\tilde x_t^{(l)}=\mathrm{Norm}!\big(x_t^{(l)}\big).
$$

Il réalise ensuite un **mélange de canaux** *intra-positionnel* suivant le patron « expansion $\to$ non-linéarité $\to$ projection de retour », c.-à-d. une projection $d_{\text{model}}\to d_{\text{mlp}}$ (souvent $d_{\text{mlp}}\gg d_{\text{model}}$) qui crée un espace de travail sur-paramétré où les directions latentes deviennent plus séparables, puis une projection $d_{\text{mlp}}\to d_{\text{model}}$ qui **réinjecte** une mise à jour dans l’espace résiduel.

Dans sa variante **gated** (SwiGLU/GeGLU, répandue dans LLaMA/Mistral/PaLM), l’opération s’écrit :

$$
\mathrm{MLP}(\tilde x_t)
=
\Big(\phi(\tilde x_t W_{\text{up}} + b_{\text{up}})
\odot
(\tilde x_t W_{\text{gate}} + b_{\text{gate}})\Big), W_{\text{down}} + b_{\text{down}},
$$

avec 

$W_{\text{up}},W_{\text{gate}}\in\mathbb{R}^{d_{\text{model}}\times d_{\text{mlp}}}$,
$W_{\text{down}}\in\mathbb{R}^{d_{\text{mlp}}\times d_{\text{model}}}$,
$b_{\text{up}},b_{\text{gate}}\in\mathbb{R}^{d_{\text{mlp}}}$,
$b_{\text{down}}\in\mathbb{R}^{d_{\text{model}}}$,

$\phi$ typiquement **SiLU** (SwiGLU) ou **GELU** (GeGLU), et $\odot$ le produit de Hadamard.

Mécaniquement, le *gating* introduit une **interaction multiplicative** qui contrôle *ce qui est écrit* dans le résiduel : la branche $\phi(\cdot)$ produit des amplitudes non linéaires (activation de features), tandis que la branche “porte” fournit des coefficients qui **modulent** ces amplitudes composante par composante. Chaque coordonnée intermédiaire $r\in{1,\dots,d_{\text{mlp}}}$ se comporte ainsi comme une **feature conditionnelle** dont la contribution est réinjectée dans une direction d’écriture déterminée par la $r$-ième ligne de $W_{\text{down}}$, avec une intensité donnée par

$$
\big[\phi(\tilde x_t W_{\text{up}} + b_{\text{up}})\odot(\tilde x_t W_{\text{gate}} + b_{\text{gate}})\big]*r.
$$

Géométriquement, cette structure induit une transformation **piècewise quasi-linéaire** de l’espace résiduel : la normalisation fixe l’échelle locale, le *gating* partitionne l’espace des $\tilde x_t$ en régions où certains sous-ensembles de features dominent, et la projection de retour recombine ces activations en un déplacement $\Delta x_t\in\mathbb{R}^{d*{\text{model}}}$.

La mise à jour résiduelle s’écrit enfin :

$$
x_t^{(l+1)} = x_t^{(l)} + \mathrm{MLP}!\big(\tilde x_t^{(l)}\big),
$$

ce qui met en évidence le rôle du MLP : non pas déplacer l’information entre positions (rôle de l’attention), mais réaliser, **à la position $t$**, un calcul non linéaire qui sélectionne et compose des directions latentes avant de les réécrire dans le flot résiduel.


#### 2) Lecture mécaniste : mémoire associative locale (clé $\to$ valeur) et déclenchement

Une lecture mécaniste standard consiste à modéliser le MLP comme une **mémoire associative adressable par le contenu**, mais **à position fixe** : l’état résiduel normalisé $\tilde x_t$ joue le rôle de requête, et un sous-ensemble d’unités intermédiaires $r\in{1,\dots,d_{\text{mlp}}}$ s’active lorsque $\tilde x_t$ projette fortement sur certaines directions apprises. Chaque unité active contribue alors une **écriture** dans $\mathbb{R}^{d_{\text{model}}}$, de sorte que la sortie du MLP s’interprète comme une superposition de “valeurs” pondérées :

$$
\mathrm{MLP}(\tilde x_t)
;=;
b_{\text{down}}
;+;
\sum_{r=1}^{d_{\text{mlp}}} a_r(\tilde x_t), v_r,
$$

où $v_r\in\mathbb{R}^{d_{\text{model}}}$ correspond (à une convention près) à la $r$-ième **ligne** de $W_{\text{down}}$ : c’est une **direction d’écriture** dans l’espace résiduel.

Dans le cas *gated* (SwiGLU/GeGLU), un choix naturel pour l’amplitude $a_r$ est :

$$
a_r(\tilde x_t)
=
\phi!\big(\langle \tilde x_t, k_r\rangle + b^{\text{up}}*{r}\big);\cdot;\big(\langle \tilde x_t, g_r\rangle + b^{\text{gate}}*{r}\big),
$$

où $k_r$ et $g_r$ sont les **colonnes** $r$ de $W_{\text{up}}$ et $W_{\text{gate}}$. Cette factorisation rend explicite le mécanisme de déclenchement : $k_r$ et $g_r$ définissent des **détecteurs directionnels** (conditions de projection pour activer l’unité), tandis que $v_r$ est la **valeur** écrite lorsque l’unité est active. Chaque $r$ implémente ainsi un micro-circuit du type *si $\tilde x_t$ satisfait ces conditions latentes, alors pousser le résiduel dans la direction $v_r$*.

Géométriquement, le MLP définit un **champ de vecteurs** sur l’espace des entrées $\tilde x_t$. Les amplitudes $a_r(\tilde x_t)$ dépendent de projections $\langle \tilde x_t,k_r\rangle$ et $\langle \tilde x_t,g_r\rangle$, ce qui induit des frontières de décision “souples” (hyperplans modulés par la non-linéarité) séparant des régions où certains neurones dominent. Dans une région donnée, la mise à jour est bien approchée par une combinaison de quelques directions $v_r$, ce qui confère au MLP une dynamique **par morceaux** (quasi-linéaire) : l’opérateur “plie” l’espace résiduel par activation sélective de directions d’écriture, plutôt que de déplacer l’information entre positions (rôle de l’attention).

Ce point de vue éclaire enfin l’aspect sécurité/mécaniste. Si un comportement indésirable correspond à l’activation d’un petit ensemble d’unités — donc à certaines écritures $v_r$ et à leurs conditions de déclenchement — une entrée adversariale n’a pas besoin de produire une “sémantique” nouvelle. Il suffit qu’elle **déplace** $\tilde x_t$ vers des régions de l’espace résiduel où ces unités s’activent fortement (p. ex. motifs de format, styles de réponse, scripts latents), ce qui entraîne l’injection des directions $v_r$ correspondantes dans le flux résiduel.

#### 3) Polysemanticité et superposition : contrainte structurelle pour l’alignement

Un obstacle central à l’interprétation — et, par extension, à l’alignement — est que les unités intermédiaires du MLP sont rarement **monosémantiques**. En pratique, le réseau apprend une représentation en **superposition** : un même espace interne sert de support à plusieurs *features*, notamment parce que (i) le nombre de régularités utiles excède la capacité dimensionnelle disponible et (ii) l’activation demeure largement **parcimonieuse** et **contextuelle**. Mécaniquement, cela implique qu’une unité $r$ ne correspond pas à “un concept” stable, mais plutôt à une direction de calcul réutilisée, dont l’interprétation dépend de la position de $\tilde x_t$ dans l’espace résiduel et des co-activations qui l’accompagnent.

Dans la lecture “mémoire associative” du MLP, la mise à jour

$$
\Delta x_t ;=; \sum_{r=1}^{d_{\text{mlp}}} a_r(\tilde x_t),v_r
$$

n’est pas une simple somme de contributions sémantiquement indépendantes : la **sélectivité** induite par la parcimonie, le *gating* et les corrélations entre détecteurs fait que la “signification” d’un terme $v_r$ n’est pas identifiable au niveau d’une unité isolée. Géométriquement, l’espace des entrées est partitionné en régions d’activation où une même direction d’écriture $v_r$ peut être mobilisée pour des raisons différentes : une même “valeur” est réutilisée comme primitive dans plusieurs programmes locaux.

Deux conséquences mécanistes en découlent pour l’alignement.

**Compromis d’édition.** Si un comportement indésirable et une capacité légitime partagent une direction d’écriture $v_r$ (ou un voisinage proche dans l’espace des valeurs), alors réduire ou ablater cette contribution revient à supprimer une direction qui transporte plusieurs fonctions. Une intervention apparemment ciblée (au niveau d’une unité ou d’un sous-espace) peut donc induire des effets collatéraux : perte de compétences, dégradation de robustesse, ou déplacement du comportement vers des routes alternatives via réallocation de superposition.

**Surfaces de déclenchement indirectes.** La superposition produit un espace de déclenchement distribué : une entrée qui ne correspond à aucun motif “pur” peut néanmoins placer $\tilde x_t$ dans une région où une combinaison d’unités s’active de manière à approximer une direction sensible downstream, c’est-à-dire où

$$
\sum_r a_r(\tilde x_t),v_r
$$

aligne le résiduel avec un sous-espace déclenchant. Le problème n’est donc pas seulement de filtrer des contenus explicites, mais de contrôler une géométrie d’activation où des interférences entre *features* peuvent recomposer des écritures problématiques de façon non évidente.

En synthèse, tant que les *features* restent en superposition, l’alignement par “désactivation de neurones” ou par édition locale des MLP se heurte à une contrainte structurelle — la non-séparabilité des primitives internes — qui rend les interventions à la fois difficiles à cibler et potentiellement fragiles.

#### Lecture clé–valeur : le FFN comme mémoire associative

Une interprétation devenue standard consiste à lire les FFN comme des **mémoires associatives clé–valeur** : des motifs détectés dans l’entrée (clés) déclenchent l’ajout de vecteurs d’écriture (valeurs) dans le résiduel.

Dans le cas canonique, en notant $\kappa_i$ la $i$-ème colonne de $W_{\text{in}}$ et $r_i$ la $i$-ème ligne de $W_{\text{out}}$, l’écriture s’explicite :

$$
\mathrm{MLP}(x_{\text{pos}})
=
\sum_{i=1}^{d_{\text{ff}}} \sigma(x_{\text{pos}} \kappa_i)\, r_i.
$$

Chaque $\kappa_i$ agit comme un **détecteur de motif** (clé) via le score $x \kappa_i$, et chaque $r_i$ comme un **vecteur d’écriture** (valeur) injecté dans le résiduel avec une amplitude $\sigma(\cdot)$. Pour l’analyse sécurité, cela met en évidence le rôle du MLP comme **compilateur de features** : une fois qu’un motif est rendu disponible à la position courante (souvent via l’attention), le FFN peut écrire une mise à jour latente réutilisable par les couches suivantes.

Dans le cas gated, une écriture analogue explicite la nature multiplicative du déclenchement. En posant $\alpha_i$ la $i$-ème colonne de $W_1$ et $\beta_i$ la $i$-ème colonne de $W_2$, on obtient :

$$
\mathrm{MLP}_{\text{gated}}(x_{\text{pos}})=\sum_{i=1}^{d_{\text{ff}}} \Big(\sigma(x_{\text{pos}} \alpha_i)\,(x_{\text{pos}} \beta_i)\Big)\, r_i,
$$

ce qui renforce l’intuition “détection $\times$ contenu” : une même “valeur” $r_i$ peut être écrite sous des conditions d’activation plus sélectives.

#### Polysémanticité, superposition, collisions (et intérêt des SAE)

Un point délicat est que les “unités” internes ne sont pas naturellement séparées : les modèles peuvent représenter **plus de features que de dimensions** via **superposition** lorsque les features sont rares/sparsifiées, au prix d’interférences. ([arXiv][4])  
Opérationnellement, cela se traduit par des **collisions sémantiques** : une même direction (ou région) de l’espace d’activation peut contribuer à des interprétations distinctes selon le contexte, ce qui complique les défenses fondées sur des “détecteurs” naïfs.

Cela motive l’usage de **Sparse Autoencoders (SAE)** en interprétabilité récente : apprendre une base parcimonieuse de features, souvent plus proches de variables monosémantiques, afin d’isoler les directions effectivement manipulées et de faciliter des interventions causales plus propres. ([Transformer Circuits][5])

<hr style="width:40%; margin:auto;">

### Couplage attention–MLP : de la collecte à l’écriture

Au sein d’un bloc, l’architecture impose un cycle fonctionnel robuste où l’attention prépare l’information et le MLP la transforme :

1. **Collecte (attention).** À la position $t$, l’attention construit une distribution de routage $\bar A_{t,\cdot}$ (par ex. $\bar A=\frac1H\sum_{h=1}^H A^{(h)}$) et agrège des contributions issues du contexte ; l’état local $x_t$ devient alors un **mélange contrôlé** de charges utiles sélectionnées à travers les positions accessibles.

2. **Écriture (MLP).** Le MLP opère ensuite **localement** sur ce vecteur enrichi : en fonction de motifs détectés (clés internes), il produit une mise à jour $\Delta x_t$ ajoutée au résiduel. Cette mise à jour correspond à une **écriture directionnelle** dans l’espace latent, susceptible de persister et d’être réutilisée par les blocs suivants.

Cette séparation est structurante : l’attention fournit principalement la **connectivité** (quels fragments d’état peuvent influencer $t$), tandis que le MLP réalise la **composition sémantique** (quelles directions latentes sont activées et consolidées). Ainsi, le comportement global résulte moins d’un “raisonnement” porté par l’attention seule que d’une coopération routage $\rightarrow$ écriture, itérée en profondeur : l’attention rend certaines informations co-présentes à $t$, et le MLP convertit cette co-présence en une représentation interne plus stable.

<hr style="width:40%; margin:auto;">

### Sonder la profondeur : du Logit Lens au Tuned Lens

L’additivité du flux résiduel autorise une lecture “anticipée” de la prédiction en appliquant l’unembedding à des états intermédiaires. En convention **vecteurs-lignes**, et en notant $x_t^{(l)}\in\mathbb R^{1\times d_{\text{model}}}$ l’état résiduel à la position $t$ après la couche $l$, la lecture brute s’écrit :

$$
z_t^{(l)} \;=\; \mathrm{Norm}_{\text{out}}\!\big(x_t^{(l)}\big)\, W_U \;+\; b,
\qquad
z_t^{(l)} \in \mathbb R^{1\times\vert\mathcal V\vert}.
$$

Le **Logit Lens** consiste à examiner $z_t^{(l)}$ couche par couche afin d’observer comment la distribution de sortie se stabilise au fil des blocs. Cette lecture reste toutefois approximative : les représentations intermédiaires peuvent subir des ré-encodages (changements de base effectifs) tels que l’application directe de $W_U$ ne constitue pas un décodeur localement fidèle.

Le **Tuned Lens** généralise cette idée en apprenant, pour chaque couche, une sonde affine qui “recentre” l’état intermédiaire avant unembedding. Une écriture standard est :

$$
z^{(l)}_{\text{tuned},t} \;=\; \mathrm{Norm}_{\text{out}}\!\big(x_t^{(l)}\big)\, D^{(l)}\, W_U \;+\; b^{(l)},
$$

où $D^{(l)}\in\mathbb R^{d_{\text{model}}\times d_{\text{model}}}$ et $b^{(l)}\in\mathbb R^{1\times\vert\mathcal V\vert}$ sont appris pour améliorer la fidélité du décodage local. ([arXiv][6])

Pour la sécurité, l’intérêt n’est pas seulement de “voir plus tôt”, mais de **dater** : identifier à quelle profondeur une trajectoire interne s’aligne sur une continuation donnée, et donc estimer si des mécanismes tardifs disposent encore d’un levier additif suffisant pour infléchir cette trajectoire.

### Spécialisation par profondeur : dynamique de construction (et préconditions de l’échec)

Les analyses couche-par-couche (probes linéaires, *logit lens* / *tuned lens*, interprétabilité mécaniste) convergent vers une spécialisation en profondeur. Les frontières exactes dépendent du modèle et du régime d’entraînement, mais une partition grossière est utile pour raisonner en sécurité.

**Heuristique de partition.** Pour un modèle de profondeur $L$ (blocs numérotés $l=1,\dots,L$) :
- **couches basses** : $l \in [1,\lfloor L/3\rfloor]$ ;
- **couches médianes** : $l \in [\lfloor L/3\rfloor+1,\lfloor 2L/3\rfloor]$ (régime typiquement centré autour de $L/2$) ;
- **couches tardives** : $l \in [\lfloor 2L/3\rfloor+1, L]$.

Cette partition n’est pas une loi, mais un repère empirique pour localiser *où* se construit une direction latente dominante et *où* s’appliquent des corrections de conformité.

#### Cadre minimal : mises à jour additives et “décodabilité” couche-par-couche

Dans un Transformer *pre-norm*, l’état résiduel à une position cible suit :

$$
x_t^{(l+1)} = x_t^{(l)} + \Delta_t^{(l)},\qquad \Delta_t^{(l)}=\Delta^{(l)}_{\text{attn},t}+\Delta^{(l)}_{\text{MLP},t}.
$$

La *logit lens* sonde la décodabilité intermédiaire en projetant un état normalisé vers l’espace vocabulaire :

$$
z_t^{(l)} = \mathrm{Norm}_{\text{out}}(x_t^{(l)})\,W_U + b,\qquad p_t^{(l)}=\mathrm{softmax}(z_t^{(l)}).
$$

La *tuned lens* remplace la projection brute par une calibration par couche (souvent affine/linéaire), afin de compenser partiellement les changements de base internes :

$$
z^{(l)}_{\text{tuned},t} = g^{(l)}(\mathrm{Norm}_{\text{out}}(x_t^{(l)}))\,W_U + b^{(l)},
$$

où $g^{(l)}$ est typiquement une transformation apprise (par couche) minimisant l’écart entre $p_t^{(l)}$ et $p_t^{(L)}$.

Une notion opérationnelle de “construction” est alors : une information est dite **décodable** à la profondeur $l$ lorsque la projection (logit/tuned lens) commence à attribuer une masse de probabilité stable à un ensemble cohérent de continuations (tokens, champ lexical, style, acte de dialogue), mesurable par une baisse de $\mathrm{KL}(p_t^{(l)} \,\|\, p_t^{(L)})$ ou par l’augmentation de la marge logit sur un sous-ensemble cible.

#### Régime 1 — Couches basses ($\approx$ premier tiers) : codage de surface et régularités locales
Les couches basses stabilisent des caractéristiques locales : ponctuation, frontières de sous-mots, formats, motifs orthographiques, dépendances courtes et contraintes syntaxiques de faible portée. Les contributions MLP y ressemblent davantage à des extracteurs de motifs “shallow” qu’à des variables abstraites globales. ([arXiv][3])

**Signature lens typique.** Les distributions $p_t^{(l)}$ y sont fortement sensibles aux régularités locales et prédisent surtout des continuations “compatibles” en surface, avec une faible structuration sémantique globale.

#### Régime 2 — Couches médianes ($\approx$ autour de $L/2$) : construction d’une structure directionnelle
Les couches médianes composent :
- des circuits d’attention (copie, référence, récupération d’indices, ancrage sur des positions saillantes) ;
- des écritures MLP qui **consolident** des variables latentes plus abstraites (entités suivies, relations, contraintes de continuation, objectifs implicites, gabarits de réponse).

Ce régime est celui où émerge fréquemment une **structure directionnelle** : une composante $v$ telle que ses projections $\langle (x_t^{(l)})^\top, w_i\rangle$ deviennent simultanément grandes pour un ensemble de directions d’unembedding $\{w_i\}$ correspondant à une continuation cohérente (thème, intention, format). Les motifs d’induction (copie conditionnelle) s’inscrivent naturellement dans ce régime et sont souvent corrélés à l’apparition de capacités d’ICL. ([arXiv][1])

**Signature lens typique.** La *tuned lens* commence à produire des distributions qualitativement proches des sorties finales : la “forme” de la continuation devient visible (entités attendues, registre, structure de réponse), même si le contenu reste encore plastique.

#### Régime 3 — Couches tardives ($\approx$ dernier tiers) : raffinement, formatage, conformité comportementale
Les couches tardives agissent comme un **raffinement** : cohérence stylistique, format, respect des consignes de dialogue, et — lorsque présent — déclenchement de comportements de refus/alignement. Le point structurel est additif : aucune “réinitialisation” n’a lieu, uniquement des corrections $\Delta^{(l)}$ supplémentaires appliquées à un état déjà structuré.

**Signature lens typique.** Les distributions intermédiaires deviennent très proches de la sortie finale ; l’essentiel du travail concerne la sélection fine (lexique exact, politesse, justification) et, le cas échéant, la bascule vers une politique de refus.

#### Préconditions géométriques de l’échec : budget additif et alignabilité

Le mode d’échec critique se formule en termes de projections et d’amplitude relative. Soit $w_{\text{adv}}$ une direction d’unembedding associée à un ensemble de continuations indésirables (ou à un champ lexical cible), et $w_{\text{align}}$ une direction associée à la conformité/refus.

Une condition typique de “fait accompli” apparaît lorsque, à la fin du régime médian, l’état $x_t^{(\ell_0)}$ (avec $\ell_0\approx 2L/3$) contient déjà une composante $v_{\text{adv}}$ telle que :
- **amplitude** : $\|v_{\text{adv}}\|$ est grande relativement au reste du signal décodable ;
- **alignabilité** : $\langle v_{\text{adv}}, w_{\text{adv}}\rangle$ est élevée (fort impact logit sur les tokens cibles).

Les couches tardives doivent alors produire une correction $\Delta_{\text{align},t}=\sum_{l>\ell_0}\Delta_t^{(l)}$ satisfaisant simultanément :
$$
\langle \Delta_{\text{align},t}, w_{\text{adv}}\rangle \ll -\langle v_{\text{adv}}, w_{\text{adv}}\rangle
\quad\text{et}\quad
\langle \Delta_{\text{align},t}, w_{\text{align}}\rangle \gg 0,
$$
tout en respectant les contraintes de magnitude et de direction imposées par la dynamique résiduelle. Lorsque la norme ou l’orientation effective de $\Delta_{\text{align},t}$ est insuffisante, la correction tardive ne compense pas une composante médiane déjà dominante : l’échec dépend autant du *moment* où une direction devient décodable que de son *amplitude relative*.

<hr style="width:40%; margin:auto;">

### Synthèse : trois leviers mécaniques

L’anatomie du bloc met en évidence trois leviers mécaniques (dont les déclinaisons offensives sont discutées en 1.4) :

1. **Contrôle du routage (attention).** Agir sur la **commande** $\bar A_{t,\cdot}$ : redistribution de masse softmax, positions ancres (*sinks*), et circuits de copie/référence qui structurent les chemins de transfert. ([arXiv][2])

2. **Déclenchement d’écritures associatives (MLP).** Activer des **clés internes** et provoquer des écritures de valeurs : dans l’écriture clé–valeur, augmenter sélectivement $\sigma(x \kappa_i)$ pour injecter des composantes $r_i$ dans le résiduel. La superposition rend possibles des collisions (mêmes unités supportant plusieurs traits). Les SAE offrent un outil d’isolement de *features* pour l’audit de ces mécanismes. ([arXiv][4])

3. **Effet de profondeur (fait accompli).** Stabiliser précocement une direction latente décodable, avant les mécanismes tardifs de raffinement/conformité. Le basculement devient observable couche-par-couche via *logit lens* / *tuned lens*. ([arXiv][6])

---

<div class="cm-figure">
  <img src="/assets/img/art1/Figure_3.png" alt="Graphique vectoriel saturation">
  <figcaption>Fig 4. Logit Lens dynamique : le modèle "acquiesce" (courbe cyan) dans les couches médianes par induction. Le refus (courbe rose) n'intervient que tardivement, créant une tension structurelle mais insuffisante.</figcaption>
</div>

<hr style="width:40%; margin:auto;">


<div class="cm-figure">
  <img src="/assets/img/art1/Figure_4.png" alt="Graphique">
  <figcaption>Fig 5. Confusion des plans et capture de la Softmax : en l'absence de cloisonnement mémoire, le déluge de données utilisateur (Rose) dilue mathématiquement l'instruction système (Cyan).</figcaption>
</div>

---

## 1.4 Exploitation : modes d’échec structurels et dynamique des flux

La décomposition opérée en 1.3 modélise le Transformer comme un système à état partagé, où chaque couche met à jour le flux résiduel. Cette dynamique repose sur l'alternance entre un mécanisme de routage (Time Mixing, lecture sélective du contexte via l'Attention), qui redistribue l’information selon le contexte, et un mécanisme de transformation (Channel Mixing, écriture/transformation locale du résiduel via les MLP), qui inscrit des features non linéaires dans l'état global. Cette lecture par circuits rend explicites les chemins de circulation de l’information et recontextualise l’analyse de sécurité : l’entrée discrète constitue un canal d’injection, tandis que l’exploitation s'opère par le contrôle de la géométrie des représentations intermédiaires.

Les vulnérabilités observées (injections de prompt, hallucinations, contournements) ne se réduisent pas à des artefacts d’implémentation, mais découlent de capacités structurelles inhérentes. Dès lors qu’un mécanisme peut extraire de l'information du contexte pour écrire dans un état partagé, un canal existe pour les signaux adverses. Une perturbation injectée est susceptible d'être propagée, et — selon les gains induits par la normalisation et les non-linéarités — de se maintenir ou s’amplifier à travers les couches.

Cette section formalise la transition entre mécanique locale (opérations intra-couche) et dynamique globale (trajectoires dans l’espace des activations). L'analyse démontre comment certaines propriétés fondamentales — la superposition (stockage dense d'information), la compositionnalité (construction hiérarchique de features) et l’inertie du flux résiduel (accumulation additive) — induisent des modes d’échec récurrents. Dans ce cadre, l’exploitation s’apparente à du steering (pilotage) : elle vise à orienter les vecteurs d’activation pour forcer une computation spécifique. L’attaque transcende alors la manipulation sémantique pour devenir une intervention géométrique, souvent indétectable par des contrôles de surface, mais déterminante pour la distribution finale des sorties.

<hr style="width:40%; margin:auto;">

### In-Band Signaling et absence de ségrégation architecturale

La vulnérabilité fondamentale des Grands Modèles de Langage ne réside pas principalement dans leur stochasticité, mais dans le **régime d’entrée unifié** qu’ils implémentent : le **_in-band signaling_**. Dans ce paradigme, **instructions de contrôle** (messages système / développeur) et **données non fiables** (entrée utilisateur, contexte externe) cohabitent dans un **canal unique**. Après tokenisation et projection, ces éléments perdent leur statut “typé” (instruction vs donnée) et ne se distinguent plus que par leurs effets sur le flux résiduel.

Cette unification est structurelle : à l’inférence, rien n’impose qu’un segment soit “consultable mais non exécutable”. Toute information présente dans la fenêtre d’attention peut, en principe, être **lue** (via des têtes d’attention) puis **réinjectée** dans l’état partagé (flux résiduel), contribuant à la prédiction du token suivant.


#### Isomorphisme Von Neumann et absence de “bit NX” architectural

Le traitement du contexte par un Transformer présente une analogie forte avec l’architecture de **Von Neumann** : **code** et **données** partagent un espace adressable commun, sans typage intrinsèque. Les systèmes d’exécution classiques atténuent cette classe de risques par des mécanismes **hors-modèle** et **structurels** (segmentation mémoire, anneaux de privilèges, permissions de pages, bit **NX**/DEP), imposant une contrainte matérielle : certains octets peuvent être stockés mais non interprétés comme instructions.

Le Transformer standard est dépourvu d’un équivalent architectural de cette **non-exécutabilité**. Il n’existe pas, au niveau du calcul interne, de barrière permettant de marquer une sous-séquence comme “donnée passive” ou “instruction privilégiée”. La hiérarchie *System / Developer / User* repose majoritairement sur une **convention de sérialisation** (templates, balises, tokens spéciaux) et non sur une séparation physique des voies de calcul. En particulier, ces marqueurs restent **in-band** : une fois intégrés au contexte, ils deviennent eux-mêmes des tokens soumis au même mélange temporel et au même mélange de canaux.

**Détail SOTA (côté sécurité applicative).** Les architectures de déploiement modernes tentent d’introduire une séparation **out-of-band** au-dessus du modèle (ex. politiques d’outils, planificateur, contrôleur, “tool router”), mais le cœur du Transformer conserve un canal de représentation unifié : la séparation est imposée par l’orchestration, pas par la dynamique interne du réseau.

#### Confused deputy vectoriel : confusion de privilèges par canal unique

Cette absence de ségrégation induit une variante **vectorielle** du problème du **_Confused Deputy_** : le modèle possède les capacités nécessaires pour agir (génération, sélection d’actions, appels d’outils dans les systèmes outillés), mais ne dispose pas d’un mécanisme interne d’**authentification de provenance** des signaux qui pilotent son état. Autrement dit, le calcul ne “sait” pas, de manière structurelle, distinguer une contrainte normative d’une incitation non fiable : tout est encodé comme activation dans le même espace latent.

Les marqueurs de sécurité (délimiteurs, rôles, tokens spéciaux) n’établissent pas une barrière ; ils modulent au mieux le contexte comme n’importe quel autre contenu. Une injection n’attaque donc pas un pare-feu logique interne : elle initie une **compétition représentationnelle**. L’objectif est de produire, via la fin du contexte et l’effet de **Time Mixing**, des activations qui dominent (en direction / alignement) celles qui encodent les contraintes de sécurité, puis de stabiliser ces effets via **Channel Mixing** dans le flux résiduel. Formellement, il s’agit d’un **steering** des activations : orienter la trajectoire du résiduel vers une région de l’espace latent où certaines interprétations et actions deviennent plus probables.

**Détail SOTA (lecture géométrique).** Avec *weight tying* (fréquent), la matrice d’embedding et la projection de sortie sont liées, ce qui renforce la lecture “géométrique” : une activation qui s’aligne avec certaines directions de vocabulaire tend à augmenter mécaniquement la probabilité des tokens correspondants. Les attaques de pilotage exploitent cette continuité : déplacer l’état interne dans l’espace latent suffit souvent à modifier le régime de génération, indépendamment de la “légitimité” sémantique de la source au niveau surface.

#### Limites de la sanitization : impossibilité d’une décision parfaite

Traiter ce problème par un filtrage statique (*sanitization*) se heurte à une limite de principe : la “dangerosité” d’une entrée n’est pas une propriété purement **syntaxique**, isolable de manière fiable par des règles locales. Elle dépend de propriétés **sémantiques** (implicatures, paraphrases, contexte), et surtout de l’**état interne** du modèle, c’est-à-dire de la dynamique d’activation induite par l’ensemble de la séquence. Il n’existe donc pas de procédure générale, à la fois **complète** (zéro faux négatif) et **sûre** (zéro faux positif), capable de décider pour toute chaîne arbitraire si elle conduira à un comportement indésirable dans tous les contextes.

En conséquence, l’injection de prompt n’est pas un “bug” isolé mais une conséquence directe de l’unification **Instruction/Donnée**. Tant que le contrôle demeure **in-band**, la sécurité ne peut pas être une garantie déterministe portée par l’architecture du modèle seul : elle repose sur un empilement de mesures (formatage, politiques externes, contrôleurs, garde-fous) dont l’efficacité est nécessairement **heuristique** et **probabiliste**.

<hr style="width:40%; margin:auto;">

### Détournement du routage attentionnel : capture et contrainte de trajectoire

La section précédente a établi que l’architecture Transformer traite instructions et données de manière indifférenciée au sein d’un même canal (*in-band*). En l’absence de ségrégation architecturale, la circulation effective de l’information est gouvernée par les mécanismes de **routage** — en pratique, les têtes d’attention.

Mathématiquement, ce routage est déterminé par des scores de similarité entre requêtes $Q$ et clés $K$, puis normalisé par *softmax* :

$$
A \;=\; \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right),
$$

où $M$ désigne le masque (p. ex. causal) et éventuels biais structurels. Cette formulation met en évidence une propriété critique : le routage dépend d’une **similarité vectorielle**, non d’une hiérarchie logique ni d’une origine de confiance. Par ailleurs, la nature exponentielle de la *softmax* induit des régimes de **saturation** : lorsqu’un petit nombre de logits domine, la masse d’attention se **concentre** sur quelques positions. Dès lors, si un segment non fiable induit des clés $K$ obtenant des produits scalaires $QK^\top$ systématiquement élevés pour certaines requêtes dominantes, il peut capter une part disproportionnée du budget d’attention de la tête concernée, marginalisant mathématiquement d’autres signaux. Le phénomène relève d’une **exploitation géométrique** de l’espace d’activation, plutôt que d’une “incompréhension” au sens sémantique.

#### Injection indirecte et contamination du contexte étendu (RAG / Web)

Dans les architectures augmentées par récupération (RAG) ou connectées à des sources externes (web, dépôts documentaires), la surface d’attaque s’étend au texte **ingéré** et concaténé au contexte. En l’absence de mécanisme natif de **provenance** au niveau latent, le modèle ne dispose pas d’une séparation interne garantissant qu’un segment “récupéré” soit traité comme moins autoritatif qu’un segment de contrôle : le routage est déterminé par la **saillance représentationnelle**.

L’injection indirecte exploite cette absence de traçabilité *in-band* : des fragments insérés dans des documents tiers peuvent, une fois tokenisés et projetés, devenir fortement prioritaires pour certaines têtes de lecture. L’effet typique est un **détournement local du routage** : une partie des ressources d’attention est allouée au contenu injecté, ce qui peut réduire l’influence effective d’instructions initiales. Le *steering* (pilotage) opère alors par compétition vectorielle : les signaux les plus saillants dans l’espace latent structurent la génération, indépendamment de leur source.

#### Porosité des délimiteurs : rôles et balises comme biais statistiques

Les formats de dialogue (ChatML, balises de rôle, séparateurs) sont souvent traités comme des frontières étanches analogues à une isolation mémoire. Pour un Transformer, ces marqueurs restent toutefois des **tokens** projetés dans le même espace vectoriel que le reste du contexte.

Ils introduisent des indices structurels et des biais locaux (via leur effet sur $Q$, $K$, $V$ à proximité), mais ne constituent pas une barrière logique : une tête peut relier des positions situées de part et d’autre d’un délimiteur dès lors que l’alignement $QK^\top$ le favorise. La séparation des rôles est donc une propriété **statistique** apprise (p. ex. lors de l’alignement / fine-tuning), et non une contrainte architecturale dure (*hard constraint*). À ce titre, elle reste susceptible d’être dégradée par des perturbations adverses qui modifient la distribution d’attention.

#### Forçage de préfixe : verrouillage inertiel de la trajectoire auto-régressive

Le caractère auto-régressif impose une dépendance forte à l’historique : la génération suit une distribution conditionnelle

$$
\mathbb{P}(s_{t} \mid s_{<t})\quad(\text{où }t\text{ est le pas de génération})..
$$

Le forçage de préfixe (*prefill*) exploite cette inertie : lorsqu’un préfixe de réponse est déjà présent dans l’historique fourni au modèle, l’état interne est orienté vers un régime où les continuations cohérentes avec ce préfixe deviennent beaucoup plus probables.

Géométriquement, le flux résiduel au pas $t$ est déjà situé dans une région de l’espace latent favorable à la continuation. Revenir brutalement vers un autre régime (p. ex. une réorientation forte de l’intention ou du style) requiert une mise à jour atypique, statistiquement défavorisée par l’objectif d’entraînement qui privilégie la cohérence locale. Le phénomène se comprend comme un **verrouillage de trajectoire** : certaines suites sont fortement favorisées, non par légitimité intrinsèque, mais par la dynamique conditionnelle induite par le préfixe.

<hr style="width:40%; margin:auto;">

### Amorçage des circuits de copie : Many-Shot et induction

Le *many-shot jailbreaking* — consistant à préfixer la requête cible par une longue suite de dialogues exemplaires — ne s’analyse pas comme une persuasion sémantique, mais comme une exploitation structurelle de l’*In-Context Learning* (ICL). La fenêtre de contexte agit comme un support d’indexation de motifs, au sein duquel certains sous-circuits (notamment des **têtes d’induction**) extraient des régularités et inscrivent dans le flux résiduel des signaux favorisant la continuation du pattern. Cette lecture s’aligne sur la décomposition établie en 1.3 : le **Time Mixing** assure la lecture sélective d’occurrences passées, tandis que le **Channel Mixing** compose ces lectures en directions d’activation.

#### Amorçage mécanique : motif invariant et têtes d’induction

Le mécanisme central repose sur un *priming* par répétition dense d’un motif du type :

$$
[\text{Requête interdite}] \;\rightarrow\; [\text{Réponse complaisante}].
$$

Le motif est instancié sous une forme stable (balises identiques, gabarit conversationnel fixe). Dans ce régime, les têtes d’attention d’induction — définies mécanistiquement comme des opérateurs exploitant des correspondances locales pour **copier** une continuation observée ailleurs — tendent à devenir plus influentes dans la continuation.

Formellement, pour une tête donnée, la mise à jour s’écrit :

$$
\alpha_t = \mathrm{softmax}\!\left(\frac{q_t K^\top}{\sqrt{d_k}}\right),
\qquad
\Delta x_t^{\text{head}} = \alpha_t\,V \;\;(\text{ligne }t\text{ de } \alpha V).
$$

Lorsque le préfixe contient de nombreuses occurrences d’un même gabarit, les représentations de clés $K$ associées aux positions de transition (fin de requête / début de réponse) deviennent nombreuses et suffisamment similaires (dans les sous-espaces pertinents) pour favoriser un alignement avec le vecteur requête $q_t$ produit au pas courant. La distribution $\alpha_t$ peut alors concentrer sa masse sur des positions structurellement analogues au cas présent ; la tête lit ces positions et réinjecte, via $V$, des directions corrélées à la continuation observée.

En première approximation, plus le motif est régulier, plus la correspondance $q\cdot k$ est facilitée, offrant une voie courte pour imposer une continuation par copie structurelle plutôt que par réinterprétation sémantique.

#### Gain effectif et compétition dans le flux résiduel

La dynamique additive du flux résiduel permet l’accumulation de contributions couche après couche :

$$
x_t^{(l+1)} = x_t^{(l)} + \Delta x_t^{\text{attn},(l)} + \Delta x_t^{\text{mlp},(l)}.
$$

La présence de multiples occurrences pertinentes dans la fenêtre permet au mécanisme d’attention d’agréger un ensemble de valeurs $V$ orientées vers une même direction de complétion. Il est alors utile de parler d’un **gain effectif** du circuit de copie : la projection de $\Delta x_t^{\text{attn}}$ sur une direction de “complaisance” tend à croître avec (i) la densité d’occurrences, (ii) leur cohérence de format, et (iii) leur compatibilité représentationnelle dans les couches précoces.

À l’opposé, les mécanismes d’alignement (p. ex. RLHF) peuvent être modélisés comme des pressions directionnelles vers des régions de l’espace latent associées au refus ou à la redirection sûre. La littérature technique emploie souvent l’expression *refusal vectors* pour désigner des sous-espaces (ou ensembles de directions) dont l’activation augmente la probabilité de tokens typiques de refus.

#### Lecture par logit lens : formalisation au niveau des logits

La compétition se reflète au niveau de la projection finale (*unembedding*, *logit lens*) :

$$
z_{t,i} = \langle x_t^{(L)}^\top, w_i \rangle + b_i.
$$

En considérant deux familles de sorties (complétion $c$ vs refus $r$), l’écart de score s’écrit :

$$
z_{t,c} - z_{t,r} = \langle x_t^{(L)}^\top, w_c - w_r \rangle + (b_c - b_r).
$$

Le *many-shot* agit en orientant $x_t^{(L)}$ (via les écritures attentionnelles et MLP successives) vers des régions augmentant $\langle x_t^{(L)}^\top, w_c\rangle$ relativement à $\langle x_t^{(L)}^\top, w_r\rangle$. Ce basculement peut être obtenu **sans** modification des poids, par renforcement inductif d’une trajectoire de complétion surreprésentée dans le contexte.

#### Basculement de régime : de “politique globale” à “complétion de motif”

Lorsque le signal de copie devient suffisamment dominant, le modèle peut basculer vers un régime où la cohérence locale (prolonger le gabarit établi) l’emporte sur des contraintes globales. La fenêtre de contexte sature alors les voies de lecture/écriture avec des signaux favorisant la continuité du pattern. Le phénomène s’interprète comme une saturation des voies de **Time/Channel Mixing** plutôt que comme un contournement cognitif : la requête cible peut rester sémantiquement inchangée, tandis que la dynamique interne privilégie une continuation structurelle.

#### Conditions limites

L’effet dépend de la régularité du format (bruit et hétérogénéité diminuent la capacité d’induction) et des non-linéarités (normalisation, saturation attentionnelle), qui empêchent une addition indéfinie des contributions.

<hr style="width:40%; margin:auto;">

### Réactivation associative et collisions de caractéristiques : mémoires MLP

Tandis que les têtes d’attention redistribuent l’information entre positions (Time Mixing), les blocs MLP opèrent **position par position** sur l’état résiduel (Channel Mixing). Dans une lecture d’interprétabilité mécaniste, le MLP ne se comporte pas comme une unité de “raisonnement symbolique”, mais comme un **mécanisme de récupération associative** : il détecte certains motifs dans le flux résiduel et y additionne des contributions apprises.

#### Le MLP comme dictionnaire associatif continu

Pour un état résiduel (pré-normalisé selon l’architecture) $x\in\mathbb R^{1\times d_{\text{model}}}}$, un sous-bloc MLP peut s’écrire (forme générique) :

$$
\Delta x_{\text{mlp}} \;=\; \phi(xW_{\text{in}} + b_{\text{in}})\,W_{\text{out}} \;+\; b_{\text{out}},
$$

puis est ajouté au flux résiduel via la connexion résiduelle.

Une écriture utile consiste à expliciter la somme sur les neurones cachés. En notant $d_{\text{mlp}}$ la largeur de la couche intermédiaire, et $\kappa_j$ la $j$-ème colonne de $W_{\text{in}}$, on obtient :

$$
a_j \;=\; \phi(x\kappa_j + b_j),
\qquad
\Delta x_{\text{mlp}} \;=\; \sum_{j=1}^{d_{\text{mlp}}} a_j\,r_j \;+\; b_{\text{out}},
$$

où $r_j\in\mathbb R^{d_{\text{model}}}$ est la $j$-ème colonne de $W_{\text{out}}$ (vecteur d’écriture).

Cette factorisation justifie l’analogie “clé–valeur” (Geva et al., 2021) :
- **Clés (détection)** : les vecteurs $\kappa_j$ définissent des hyperplans de détection ; l’affinité $x\kappa_j$ mesure l’alignement *linéaire* entre le résiduel et la “feature” détectée (ce n’est une similarité cosinus que sous normalisation explicite).  
- **Valeurs (écriture)** : chaque activation $a_j$ pondère une direction d’écriture $r_j$ injectée dans le flux résiduel.

Contrairement à l’attention (copie dynamique de contenu déjà présent dans le contexte), le MLP implémente une récupération **paramétrique** : il réinjecte des directions stockées dans les poids, apprises pendant l’entraînement.

> *Note d’implémentation.* Avec des variantes gated (p. ex. SwiGLU), la détection est modulée par une porte ; l’intuition “somme de valeurs $v_j$ pondérées par des activations” reste cependant pertinente au niveau fonctionnel.

#### Superposition et polysemanticité

Une contrainte structurelle réside dans l’écart entre le grand nombre de “features” utiles et la dimension finie $d_{\text{model}}$. La représentation interne se fait alors par **superposition** : plusieurs directions conceptuelles partagent des sous-espaces et ne sont pas orthogonales.

Dans ce régime, des unités intermédiaires du MLP peuvent devenir **polysémantiques** : un même neurone (ou une même direction détectée) répond à plusieurs régularités partiellement corrélées. Mathématiquement, l’activation $a_j=\phi(\kappa_j^\top x + b_j)$ peut être déclenchée par des configurations distinctes de $x$ dès lors qu’elles ont une projection suffisante sur $k_j$.

Cette économie dimensionnelle induit des **interférences** : activer une direction associée à un concept $A$ peut co-activer (faiblement ou fortement) des directions $B, C$ lorsque les détecteurs $k_j$ et les écritures $v_j$ ne sont pas séparés de manière orthogonale.

#### Mode d’échec : collisions de caractéristiques et steering latent

Du point de vue sécurité, l’architecture met en jeu une tension : des garde-fous externes opèrent souvent sur des représentations discrètes (surface, tokens, classes sémantiques), alors que la dynamique interne est continue et distribuée. Dans un système en superposition, deux entrées “différentes” au niveau lexical peuvent produire des états $x$ dont les projections sur certains détecteurs $\kappa_j$ sont similaires, déclenchant des écritures $r_j$ comparables : c’est une forme de **collision de caractéristiques**.

Lorsque de telles collisions surviennent dans des régions sensibles (p. ex. des sous-espaces associés à des comportements indésirables), le MLP peut agir comme un **amplificateur** : une configuration initialement ambiguë dans $r$ déclenche une récupération associative qui oriente le flux résiduel vers une trajectoire problématique. Dans cette lecture, une partie des comportements de contournement ou de dérive ne relève pas d’un “raisonnement erroné”, mais d’une **interférence vectorielle** conduisant à une récupération associative inadéquate — un cas de *steering* au niveau latent.

> **Cas d’école : ASCII art et attaques de format.**  
>
> Les prompts en ASCII art ne “fonctionnent” pas principalement par magie lexicale, mais comme **attaques de mise en forme**. D’une part, la surface (espaces, répétitions, caractères atypiques) peut dégrader des contrôles statiques. D’autre part, et surtout, ces motifs activent des *features* internes liées à des registres de présentation (bannières, tableaux, terminaux, logs). Par **superposition**, ces directions de format peuvent entrer en collision avec des directions “instructionnelles” ou “d’autorité” et déclencher, via les MLP (mémoires associatives), des écritures $r_j$ qui **pilotent** le flux résiduel vers un régime de complétion de gabarit. L’effet observable est un *steering* latent : le modèle se met à suivre la structure imposée par le format, indépendamment du statut réel (non privilégié) du contenu in-band.

<hr style="width:40%; margin:auto;">

### Optimisation adversariale sous contrainte discrète : gradients et topologie

L’exploitation d’un Grand Modèle de Langage peut se formaliser comme un problème d’optimisation sous contrainte discrète. Si l’interface d’entrée impose une séquence de symboles

$$
(s_1,\dots,s_n)\in\mathcal V^n,
$$

la dynamique interne du modèle — induite par la projection en embeddings puis la composition des couches — est une application différentiable presque partout. Cette dualité discret/continu autorise une lecture géométrique : une attaque correspond à la recherche de trajectoires d’activation maximisant un objectif donné dans l’espace latent, plutôt qu’à une exploration aléatoire de l’espace des séquences.

#### Formulation de l’objectif adversarial

Soit une séquence d’entrée $s=(s_1,\dots,s_n)\in\mathcal V^n$. L’attaque se modélise comme la recherche d’une entrée maximisant une fonction objectif $J$. Dans un cas typique, $J$ mesure la probabilité d’un mode de sortie $y^\star$ non souhaité du point de vue sécurité :
$$
\max_{s\in\mathcal V^n} J(s)
\qquad\text{avec}\qquad
J(s)=\log \mathbb{P}(y^\star\mid s,\theta),
$$
où $\theta$ désigne les paramètres figés du modèle. La difficulté centrale provient de la contrainte $\mathcal V^n$ : aucune descente de gradient n’est définie directement sur des symboles discrets, alors que $J$ est induite par une chaîne d’opérations continues après projection.

#### Du discret au continu : gradients dans l’espace des embeddings

En notant $i_t=\iota(s_k)$ l’ID du token en position $k$, et $e_t=W_E[i_t]\in\mathbb R^{d_{\text{model}}}$ son embedding, la rétropropagation fournit le gradient de $J$ par rapport au vecteur dense $e_t$ :
$$
\nabla_{e_t}J \;=\; \frac{\partial \log \mathbb{P}(y^\star\mid s,\theta)}{\partial e_t}.
$$
Ce gradient définit dans $\mathbb R^{d_{\text{model}}}$ une direction locale d’augmentation de l’objectif. Le “retour” au discret s’interprète alors comme une forme de quantification/projection : parmi les embeddings réalisables $\{W_E[i]\}_{i\in\{0,\dots,\vert\mathcal V\vert-1\}}$, identifier des substitutions plausibles dont l’effet local est le plus compatible avec cette direction (au sens d’un voisinage ou d’un alignement dans l’espace des embeddings), puis valider cet effet dans le modèle complet.

#### Preuve de principe : recherche discrète guidée par gradient (famille GCG)

Les méthodes de la famille **Greedy Coordinate Gradient (GCG)** fournissent une instanciation opérationnelle du *steering* : une optimisation sur l’espace discret des séquences $\mathcal V^n$ est pilotée par une information différentielle définie dans l’espace continu des activations, sans modification des poids du modèle.

Soit un objectif scalaire $J$ (score ou contrainte ; pour une perte on maximise son opposé) et un ensemble de positions modifiables $\Omega\subseteq\{1,\dots,n\}$. En notant $e_t = W_E[i_t]$ l’embedding à la position $k$, le gradient $\nabla_{e_t}J$ fournit une **sensibilité locale** : il indique, dans $\mathbb R^{d_{\text{model}}}$, la direction de variation de l’embedding qui augmenterait $J$ (toutes choses égales par ailleurs).

Le principe de GCG consiste à exploiter cette information **continue** pour cribler une recherche **discrète**. Plutôt que d’explorer aveuglément $\mathcal V$, le gradient est projeté sur la table d’embeddings afin d’identifier des tokens dont la représentation vectorielle est la plus alignée avec une ascension de $J$. Cette étape correspond à une **relaxation** du problème discret et à l’usage d’une **approximation linéaire de premier ordre** pour proposer des candidats.

Au sens algorithmique, GCG s’interprète comme une forme d’**ascension de coordonnées gloutonne** (*greedy coordinate ascent*) sur l’ensemble de positions $\Omega$ : à chaque itération, l’optimisation modifie une ou plusieurs coordonnées (positions de tokens) en choisissant localement la mise à jour la plus favorable selon $J$, sous une politique gloutonne.

#### Mécanique d’une itération (schéma GCG)

1) **Calcul du gradient (relaxation).**  
Pour chaque position cible $k\in \Omega$, calcul de $\nabla_{e_t}J$ par rétropropagation. Ce vecteur représente la direction locale dans l’espace continu dans laquelle une modification de $e_t$ augmenterait $J$.

2) **Criblage des candidats (Top-k).**  
Sélection d’un ensemble restreint de substitutions $\mathcal C_k\subset \mathcal V$ (p. ex. les 256 meilleurs tokens) en maximisant l’alignement avec le gradient d’ascension, typiquement via un critère du type
$$
u \in \arg\max_{u\in\mathcal V} \langle W_E[u]-e_t, \nabla_{e_t}J\rangle,
$$
ce qui correspond à une estimation de la variation de $J$ sous une linéarisation locale.

3) **Validation exacte (forward pass).**  
Pour chaque candidat $u\in\mathcal C_k$, évaluation de $J$ par un passage avant complet ; la substitution retenue est celle qui maximise effectivement $J$, en réintégrant les non-linéarités ignorées lors du criblage.

4) **Politique de mise à jour (coordonnée unique vs mise à jour en bloc).**  
Deux variantes usuelles coexistent :  
- **Une seule position par itération** : un unique $k\in \Omega$ est sélectionné (selon une règle gloutonne) et mis à jour, ce qui stabilise l’optimisation et limite les interactions non linéaires entre positions.  
- **Mise à jour en bloc** : plusieurs positions $k\in \Omega$ sont modifiées au sein d’une même itération, ce qui accélère la recherche mais augmente le couplage entre mises à jour et peut nécessiter une validation plus stricte.

Ce cadre constitue une preuve de principe : une recherche combinatoire a priori prohibitive devient praticable en exploitant l’interface entre représentation discrète (séquence de tokens) et représentations continues (embeddings/activations). La sécurité se reformule alors comme un problème d’optimisation, où le *steering* des activations sert de guide à une exploration contrôlée de l’espace des séquences.


#### Topologie des “tunnels” adverses

L’existence de séquences optimisées met en évidence une propriété géométrique en haute dimension : l’objectif $J$ induit un paysage fortement non convexe, dans lequel peuvent exister des chemins de progression contournant les régions dominées par des comportements sûrs (refus, prudence, redirection). Dans cette lecture :
- les mécanismes d’alignement tendent à rendre certaines zones de trajectoires plus stables ;
- une recherche guidée exploite la non-convexité pour atteindre des régions où une autre dynamique interne devient dominante.

Cette optimisation demeure locale et probabiliste : minima locaux, plateaux, effets de normalisation (p. ex. LayerNorm) et compétition entre circuits limitent toute garantie globale, mais augmentent significativement la probabilité d’identifier des modes d’échec structurels par rapport à une exploration non guidée.


<hr style="width:40%; margin:auto;">

### La course en profondeur : le « fait accompli »

La section 1.4.5 a établi l’existence de trajectoires adversariales (via optimisation discrète ou continue) menant à une complétion non conforme. La réussite pratique dépend toutefois d’un phénomène dynamique : pourquoi les dernières couches ne “rattrapent-elles” pas une trajectoire déjà engagée ?

Le **fait accompli** désigne un mode d’échec où la non-conformité provient moins d’une absence de mécanismes de refus que d’un **retard en profondeur**. Lorsque des mises à jour précoces ou médianes ont déjà verrouillé une direction latente dominante, les corrections tardives (style, conformité, refus explicite) ne disposent plus d’un levier géométrique suffisant pour inverser la trajectoire du flux résiduel.

#### Cadre minimal : accumulation additive et notion de *commitment*

Dans un Transformer *pre-norm* (cf. 1.3), l’état résiduel à une position cible suit une dynamique additive :
$$
x^{(l+1)} = x^{(l)} + \Delta^{(l)},\quad \text{avec} \quad \Delta^{(l)}=\Delta^{(l)}_{\text{attn}}+\Delta^{(l)}_{\text{MLP}}.
$$
Cette forme implique une **accumulation** : la profondeur raffine un vecteur existant plutôt que de le remplacer.

Une mesure de *commitment* (marge directionnelle) peut alors être définie entre deux continuations antagonistes : une complétion non conforme et un refus. Soient $w_c,w_r\in\mathbb{R}^{d_{\text{model}}}$ deux **directions d’unembedding** associées à ces continuations (p. ex. colonnes de $W_U$, ou, sous *weight tying*, vecteurs correspondants dans $W_E$). La marge est :

$$
C^{(l)} \;=\; \langle x^{(l)},\, w_c - w_r \rangle.
$$

Un $C^{(l)}$ fortement positif indique que l’état interne est déjà plus aligné avec la complétion non conforme qu’avec le refus, avant même d’atteindre la fin du réseau.

#### 1.4.6.2 Asynchronie des sous-circuits : avantage temporel des écritures précoces

En reprenant la spécialisation par profondeur discutée en 1.3, les opérations de lecture/écriture ne sont pas synchrones : certaines couches stabilisent tôt des variables latentes (objectif implicite, gabarit de réponse, structure directionnelle), tandis que d’autres opèrent plus tard comme mécanismes de raffinement et de conformité.

Le fait accompli est favorisé lorsque des écritures (Channel Mixing / MLP) et des routages d’attention (Time Mixing) construisent précocement une représentation dominée par $w_c$. Si les corrections tardives (dont les circuits de refus, lorsqu’ils sont principalement tardifs) n’interviennent qu’après la formation d’une marge $C^{(l)}$ déjà élevée, elles agissent sur un terrain défavorable.

#### Inertie géométrique : coût de la réorientation tardive

À une profondeur $l$ donnée, une correction de sécurité efficace doit produire une mise à jour $\Delta^{(l)}_{\text{safety}}$ telle que la marge devienne non positive (bascule vers le refus) :
$$
C^{(l+1)} \;=\; \langle x^{(l)} + \Delta^{(l)}_{\text{safety}},\, w_c - w_r \rangle \;\le\; 0.
$$
De manière équivalente, une condition sur la force directionnelle de la correction s’écrit :
$$
\langle \Delta^{(l)}_{\text{safety}},\, w_r - w_c \rangle \;\ge\; C^{(l)}.
$$
Cette inégalité constitue le cœur du fait accompli : si $C^{(l)}$ est déjà grand, la correction requise doit être fortement alignée avec $w_r-w_c$. Or, les mises à jour $\Delta^{(l)}$ sont contraintes par la dynamique interne (normalisation, saturation des activations, capacité finie des sous-couches), ce qui borne le **steering** (pilotage) disponible dans les dernières couches. Lorsque la marge accumulée excède cette capacité effective, la trajectoire devient difficile à inverser.

#### Lecture via *logit lens* : point de non-retour

La *logit lens* rend observable cette dynamique en évaluant, couche par couche, la compétition entre refus et complétion. En notant $z^{(l)}$ la projection intermédiaire vers l’espace vocabulaire (souvent à partir d’un état normalisé), le **logit gap** est :
$$
\mathrm{Gap}(l) \;=\; z^{(l)}_{c} - z^{(l)}_{r}.
$$
En pratique, un fait accompli correspond à un régime où $\mathrm{Gap}(l)$ devient nettement positif dès les couches médianes et conserve une avance stable : le modèle est “commis” à une famille de continuations, et les corrections tardives n’opèrent plus que des ajustements locaux (hésitation, style) sans renverser la décision sémantique globale.

#### Conclusion : vulnérabilité temporelle de la profondeur

Le fait accompli met en évidence une contrainte structurelle : la profondeur introduit une **asymétrie temporelle**. Quand la détection ou la rectification de non-conformité intervient principalement tard, elle doit lutter contre un flux résiduel ayant déjà acquis une inertie significative.

Une défense robuste ne peut donc se réduire à un “gendarme final” (filtre en sortie). Elle requiert soit (i) des mécanismes de refus capables d’intervenir tôt, soit (ii) des contraintes structurelles limitant la capacité de steering adversarial vers certaines directions, soit (iii) une redistribution des fonctions de conformité tout au long de la profondeur afin d’empêcher la formation d’un *commitment* élevé.

---


### Conclusion : De la logique à la probabilité

Ce texte a parcouru la chaîne complète du Transformer, depuis la discrétisation de l’entrée (tokenisation, IDs) jusqu’à la dynamique continue des activations (embeddings, attention/MLP, flux résiduel, lecture des logits). L’enjeu central qui en découle, en sécurité offensive comme défensive, est le suivant : un LLM ne se spécifie ni ne se protège comme un système déterministe.

Dans une base SQL, une API ou un programme classique, la sécurité se formule en termes de décisions discrètes : une requête est valide ou invalide, un droit est accordé ou refusé, une règle est satisfaite ou violée. Dans un Transformer, le comportement résulte d’une **distribution** sur les sorties, induite par une dynamique interne continue. Il n’existe pas de mécanisme interne garantissant un interdit absolu ; il existe un ensemble de contraintes et de préférences qui modulent les probabilités. L’exploitation consiste alors à **déplacer la masse de probabilité** vers des générations non désirées ; la défense consiste à **réduire** la probabilité de ces générations et à **augmenter la marge** (robustesse) des régimes sûrs, via des interventions sur la dynamique des activations et les mécanismes de contrôle amont/aval.

La structure computationnelle qui organise cette dynamique est l’alternance **Time Mixing** (attention : lecture/agrégation du contexte) et **Channel Mixing** (MLP : transformation/recombinaison). Ce couplage définit un canal de transport et de transformation de signaux où superposition, compositionnalité et partage d’état — sources de performance — constituent également des vecteurs d’attaque. Dans ce cadre, l’exploitation peut être décrite comme du **steering** (*pilotage*) : l’orientation du flux d’activation vers des régions de l’espace latent qui rendent certaines suites de tokens plus probables, parfois en dehors de ce que des contrôles opérant sur la surface du texte infèrent.

### Vers la géométrie de l’intelligible

La section précédente a caractérisé les mécanismes de circulation de l’information ; la suivante portera sur la géométrie des représentations : l’espace de haute dimension dans lequel le modèle encode et combine des concepts. L’objectif devient l’identification de directions, de sous-espaces et de séparations internes associés à des familles de comportements.

Le prochain article développera cette perspective et montrera comment le **steering** des activations — en attaque comme en alignement — se formalise comme un problème de trajectoires dans l’espace latent, plutôt que comme un problème de reformulation en surface.


<hr style="width:40%; margin:auto;">
---



[1]: https://arxiv.org/abs/2209.11895?utm_source=chatgpt.com "In-context Learning and Induction Heads"
[2]: https://arxiv.org/abs/2309.17453?utm_source=chatgpt.com "Efficient Streaming Language Models with Attention Sinks - arXiv"
[3]: https://arxiv.org/abs/2012.14913?utm_source=chatgpt.com "Transformer Feed-Forward Layers Are Key-Value Memories"
[4]: https://arxiv.org/abs/2209.10652?utm_source=chatgpt.com "[2209.10652] Toy Models of Superposition"
[5]: https://transformer-circuits.pub/2023/monosemantic-features?utm_source=chatgpt.com "Decomposing Language Models With Dictionary Learning"
[6]: https://arxiv.org/abs/2303.08112?utm_source=chatgpt.com "[2303.08112] Eliciting Latent Predictions from Transformers ..."
