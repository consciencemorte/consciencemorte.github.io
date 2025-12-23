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

## 1.1 Tokenisation et discrétisation de l’espace d’entrée

L’interaction avec un Grand Modèle de Langage (LLM) se donne comme un flux textuel continu ; pourtant, le réseau ne manipule qu’une **séquence discrète d’entiers**. La *tokenisation*, première transformation du pipeline d’inférence, matérialise l’interface entre langage naturel (symbolique) et calcul matriciel (numérique). Cette interface constitue une **surface d’attaque structurelle** : elle est discontinue, dépend d’heuristiques d’implémentation (normalisation, pré-tokenisation), et se trouve souvent découplée des garde-fous périphériques (WAF, filtres, classifieurs externes).

### Formalisation du processus de tokenisation

Soit $\mathcal{S}$ l’ensemble des chaînes de caractères possibles. La tokenisation peut être modélisée comme une composition :

$$
\mathrm{Tok} \;=\; \iota^{*} \circ \tau \circ \nu \;:\; \mathcal{S} \rightarrow \{0,\dots,\vert\mathcal{V}\vert-1\}^*
$$

- $\nu : \mathcal{S}\rightarrow \mathcal{S}$ — **normalisation / pré-tokenisation** : règles déterministes sur espaces, Unicode (NFC/NFKC ou autre selon implémentation), motifs de découpage, etc.
- $\tau : \mathcal{S}\rightarrow \mathcal{V}^*$ — **segmentation** en une séquence $(v_1,\dots,v_n)$, avec $v_k\in\mathcal{V}$.
- $\iota : \mathcal{V}\rightarrow \{0,\dots,\vert\mathcal{V}\vert-1\}$ (étendue pointwise en $\iota^{*}:\mathcal V^{*}\to\{0,\dots,|\mathcal V|-1\}^{*}$) — **indexation** associant à chaque token un entier $i_k$ (Token ID).

Le vocabulaire $\mathcal{V}$ est fini, de cardinalité $\vert\mathcal{V}\vert$ fixée avant l’entraînement (souvent $32\,000$ à $128\,000$ pour de nombreuses architectures récentes). Dans les tokenizers byte-level réversibles, toute chaîne arbitraire admet une représentation valide : la contrainte d’encodabilité ne joue pas le rôle de barrière, y compris pour des entrées bruitées ou non standards.

> **Terminologie (niveau discret).** Dans la suite, un *token* désigne un symbole du vocabulaire du tokenizer $v\in\mathcal V$ produit par la segmentation $\tau$, et son *ID* $i=\iota(v)\in\{0,\dots,\vert\mathcal V\vert-1\}$ est un **indice** servant au lookup (accès indexé) dans la matrice d’embedding (définie ci-après). Les IDs n’induisent aucune géométrie : $\vert i-j\vert$ n’a pas de signification sémantique.  
> La chaîne de représentation discrète s’écrit :
> $$
> \mathcal S \xrightarrow{\nu} \mathcal S \xrightarrow{\tau} \mathcal V^* \xrightarrow{\iota^{*}} \{0,\dots,\vert\mathcal V\vert-1\}^*,
> $$
> et produit une séquence d’IDs $(i_1,\dots,i_n)$.

<hr style="width:40%; margin:auto;">

### Algorithmique des sous-mots : compression statistique

Les tokenizers modernes reposent majoritairement sur des algorithmes de **segmentation en sous-unités** (*subwords*), visant une **compression statistique** plutôt qu’une analyse morphologique.

- **BPE (Byte-Pair Encoding)** : construction incrémentale d’un vocabulaire par fusions successives de paires fréquentes (caractères, octets, ou symboles pré-tokenisés selon la variante), jusqu’à atteindre une taille cible. L’effet recherché est une séquence moyenne plus courte, sans prétention d’analyse linguistique.
- **Unigram LM** : sélection d’un vocabulaire et d’une segmentation maximisant la vraisemblance sous un modèle probabiliste ; souvent déployé via SentencePiece.

Les implémentations industrielles exposent généralement trois “objets” techniques : un motif de pré-tokenisation (souvent une expression régulière), une table de fusions (rangs de fusion), et un ensemble de **tokens spéciaux** (délimiteurs de message, fin de séquence, marqueurs de format). Ces tokens spéciaux sont critiques : ils déterminent la structure effective du prompt (templates instruct/chat) tout en étant parfois traités hors du périmètre des filtres centrés sur le texte naturel.

<hr style="width:40%; margin:auto;">

### Asymétrie et discontinuité : la faille de l’interface

La tokenisation introduit une **asymétrie fondamentale** entre perception de l'utilisateur (visuelle, continue) et représentation computationnelle (discrète, numérique). La transformation est **discontinue** : une perturbation minime dans $\mathcal{S}$ (espace, variante Unicode, caractère invisible) peut induire une normalisation $\nu(s)$ et/ou une segmentation $\tau(s)$ radicalement différentes, donc une suite d’IDs sans corrélation triviale avec la forme initiale.

Cette discontinuité correspond à une **rupture de l’adjacence** : une similarité de surface (visuelle ou typographique) ne se traduit pas en similarité dans la représentation discrète. Dès lors, tout contrôle fondé sur la surface (regex, mots-clés) ou sur des motifs tokenisés reste intrinsèquement non robuste face aux obfuscations.

<hr style="width:40%; margin:auto;">

### Surface d’attaque de la tokenisation : contournement des garde-fous par obfuscation

Les garde-fous déployés dans les systèmes réels opèrent sur des **représentations hétérogènes** :

- **Filtrage de surface** : sur $\mathcal{S}$ (regex, canonicalisation) avant tokenisation.
- **Filtrage tokenisé** : sur la séquence de tokens $(v_1,\dots,v_n)$ ou d’IDs $(i_1,\dots,i_n)$ (listes noires, règles) après tokenisation.
- **Classifieurs externes** : modèles spécialisés (souvent des encodeurs de type BERT/RoBERTa/DeBERTa, parfois distillés) finement ajustés pour classifier toxicité, intention ou conformité, opérant sur texte brut et/ou sur une représentation dérivée produite par leur propre encodeur (avec une normalisation/tokenisation potentiellement différente de celle du LLM principal).

Le LLM opère sur des représentations internes $x^{(l)}$ dans un espace continu (espace latent). Un angle mort exploitable apparaît lorsqu’un garde-fou valide une entrée dans l’espace où il opère (surface, tokens/IDs, ou représentations pré-LLM), alors que l’entrée effective du modèle (embeddings puis états internes du flux résiduel) encode l’intention adversariale ciblée. En particulier, les classifieurs externes échouent typiquement par **décalage de distribution** : entraînement sur des formes canoniques, sensibilité aux obfuscations (typos, confusables, translittérations) et divergences de normalisation/tokenisation entre le composant de sûreté et le tokenizer du LLM.

Les mécanismes suivants structurent ce canal d’obfuscation.

**Variabilité multilingue et multi-écriture.** 
Un concept peut être atomique dans une langue et fragmenté en plusieurs sous-tokens dans une autre (agglutination, translittération, scripts non latins). La granularité “visible” par les contrôles dépend alors du tokenizer, de la langue et du système d’écriture, ce qui modifie la segmentation et la surface effective d’obfuscation, et rend instable toute défense calibrée sur une forme canonique.

**Perturbations de surface et invariance par fragmentation.**  
De faibles perturbations (typos, séparations, insertion d’espaces/caractères neutres, confusables) peuvent forcer une re-segmentation. Un token canonique susceptible d’être filtré (“Malicious”) peut devenir une suite de sous-unités distinctes (“Mal”, “is”, “cious”). À l’échelle des IDs, l’entrée change complètement ; à l’échelle latente, la composition des sous-unités (embeddings puis transformations des premières couches) tend à préserver un signal suffisant pour que l’intention demeure exploitable. Le filtre lexical observe des fragments non bloqués ; le modèle reconstruit une direction sémantique voisine.

<figure class="cm-figure">
  <img src="/assets/img/art1/Figure_1.png" alt="Graphique tokenisation" loading="lazy">
  <figcaption>
    Fig. 1 — Illusion de fragmentation : des tokens de surface disjoints peuvent, après composition interne,
    converger vers une intention similaire, rendant les filtres lexicaux insuffisants.
  </figcaption>
</figure>

**Incohérences de normalisation Unicode (frontières inter-composants).**  
Les pipelines réels enchaînent plusieurs composants (proxy/WAF, normaliseur, classifieurs, tokenizer, LLM) dont les politiques Unicode peuvent diverger. Un espace d’obfuscation apparaît lorsque deux chaînes Unicode distinctes mais visuellement confondables ne sont pas rendues équivalentes par les mêmes opérations de normalisation/canonicalisation selon les composants, ou lorsque la canonicalisation intervient après le contrôle. Un cas classique est la différence entre forme Unicode composée et décomposée : par exemple “é” peut être encodé comme un seul point de code ($U+00E9$) ou comme “e” + accent combinant ($U+0065$ + $U+0301$). Un filtre amont qui canonicalise (ou non) différemment du tokenizer peut valider une variante, tandis que la variante effectivement tokenisée diverge et échappe aux motifs attendus.

**Alignement cross-lingue et chimères sémantiques.**  
L’entraînement multilingue tend à aligner des concepts proches à travers les langues. Des séquences hybrides (mélange de fragments de langues/systèmes d’écriture) peuvent apparaître incohérentes pour un filtre lexical tout en restant interprétables pour le modèle : la sémantique est portée principalement par la géométrie des représentations internes plutôt que par la conformité syntaxique de surface. Ce mécanisme autorise des “chimères” : une entrée qui se présente comme du bruit pour un contrôle de surface, mais dont la projection en embeddings, suivie de la composition dans les premières couches, déplace l’état interne vers la région sémantique cible.

<figure class="cm-figure">
  <img src="/assets/img/art1/gpt_respond.png" alt="Illustration de la robustesse de l’espace latent" loading="lazy">
  <figcaption>
    Robustesse de l’espace latent : une entrée obfusquée peut être “reconstruite” en intention après passage dans les couches.
  </figcaption>
</figure>

**Tokens atypiques et outliers distributionnels (“glitch/anomalous tokens”).**  
Certains tokens associés à des motifs rares (séquences techniques, fragments de code, chaînes bruitées) reçoivent des mises à jour de gradient beaucoup plus rares et éparses, et tendent donc à être moins contraints par l’entraînement effectif. Ces outliers présentent fréquemment des statistiques atypiques (norme, voisinage, anisotropie) par rapport à la distribution moyenne des embeddings. Leur injection peut amplifier des activations (via normalisations et produits scalaires), déstabiliser des régimes internes et révéler des comportements limites.

<hr style="width:40%; margin:auto;">

### Projection dans l’espace vectoriel : embedding

L’**espace latent** désigne l’espace vectoriel continu $\mathbb R^{d_{\text{model}}}$ dans lequel vivent les représentations internes du modèle (vecteurs d’embedding et activations des couches). La matrice d’embedding est notée :

$$
W_E \in \mathbb{R}^{\vert\mathcal{V}\vert\times d_{\text{model}}}.
$$

La transition du discret au continu s’écrit :
$$
s \xrightarrow{\nu,\tau} (v_1,\dots,v_n) \xrightarrow{\iota^{*}} (i_1,\dots,i_n) \xrightarrow{W_E} (e_1,\dots,e_n).
$$

La tokenisation produit une suite d’indices $(i_1,\dots,i_n)$ ; l’entrée continue correspondante est la suite de vecteurs d’embedding $e_k=W_E[i_k]\in\mathbb R^{d_{\text{model}}}$, à laquelle s’ajoute un mécanisme positionnel, avant composition par les couches.

Un token d’ID $u$ peut être représenté par un vecteur *one-hot* $\delta_u\in\{0,1\}^{\vert\mathcal{V}\vert}$ :
$$
(\delta_u)_m =
\begin{cases}
1 & \text{si } m = u, \\
0 & \text{sinon,}
\end{cases}
\quad \forall m \in \{0,\dots,\vert\mathcal{V}\vert-1\}.
$$

La projection dans l’espace latent s’écrit :
$$
e_u = \delta_u^\top W_E \in \mathbb{R}^{d_{\text{model}}}
\qquad\text{(équivalent à un accès indexé : } e_u = W_E[u]\text{)}.
$$

Les vecteurs de $W_E$ sont appris par rétropropagation dans l’objectif de **prédiction du token suivant** (Causal Language Modeling). Aucune contrainte n’impose directement qu’un token soit “proche” d’un autre : la géométrie de l’espace des embeddings émerge des signaux de gradient induits par la prédiction. Des tokens qui apparaissent dans des contextes similaires (synonymes, variantes typographiques, fragments corrélés) reçoivent des mises à jour proches et tendent ainsi à occuper des régions voisines de l’espace des embeddings. Cette topologie apprise explique pourquoi certaines altérations de surface peuvent rester sémantiquement exploitables après projection.

Cette observation est cohérente avec l’**hypothèse distributionnelle** : sous un objectif de prédiction, si deux tokens $t_1,t_2$ induisent des distributions de contextes proches, leurs embeddings ont tendance à se rapprocher de l'heuristique :
$$
P(C \mid t_1) \approx P(C \mid t_2) \;\Longrightarrow\; W_E[t_1] \approx W_E[t_2],
$$
au sens où une divergence faible entre distributions de contextes se traduit souvent par une faible distance (ou un fort cosinus) entre vecteurs d’embedding. Ce mécanisme contribue à une robustesse sémantique partielle face à certaines obfuscations de surface.

<hr style="width:40%; margin:auto;">

### Unembedding et weight tying : logits, alignement et couplage entrée/sortie

À l’instant $t$, le modèle produit un vecteur de **logits** $z_t\in\mathbb R^{1\times\vert\mathcal V\vert}$, c’est-à-dire des scores **non normalisés** sur le vocabulaire avant application de la Softmax. Ces scores proviennent de la projection de l’état interne final $h_t$ vers l’espace vocabulaire via une matrice d’**unembedding** $W_U$ (à biais près). Dans les architectures où les poids sont liés (*weight tying*), $W_U$ n’est pas appris indépendamment : il est contraint par la matrice d’embedding d’entrée $W_E$, typiquement $W_U = W_E^\top$. Cette contrainte rend explicite une équivalence géométrique entre encodage et décodage : les mêmes vecteurs de vocabulaire servent à la fois d’**embeddings** en entrée et de **directions de scoring** en sortie, de sorte que la génération est directement gouvernée par l’alignement de $h_t$ avec ces vecteurs.


**Convention de dimensions**  
L’état résiduel final au pas $t$ est noté $h_t \in \mathbb{R}^{1\times d_{\text{model}}}$ et la matrice d’embedding $W_E \in \mathbb{R}^{\vert\mathcal{V}\vert\times d_{\text{model}}}$. La projection vers l’espace vocabulaire s’écrit, à biais près :
$$
W_U = W_E^\top \in \mathbb{R}^{d_{\text{model}}\times\vert\mathcal{V}\vert},
\qquad
z_t = h_t W_U + b \in \mathbb{R}^{1\times\vert\mathcal{V}\vert}.
$$

En notant $w_i\in\mathbb R^{d_{\text{model}}}$ le vecteur d’embedding du token $i$ (transposé de la i-ème ligne de $W_E$ : $w_i := W_E[i]^\top$), chaque logit est :
$$
z_{t,i} = \langle h_t^\top, w_i \rangle + b_i.
$$
...
$$
z_{t,i} - z_{t,j} = \langle h_t^\top, w_i - w_j \rangle + (b_i - b_j)
$$

Ces logits paramétrisent ensuite la distribution de sortie via la Softmax.

**Distribution de sortie (Softmax)**  
À ce stade, $z_{t,i}$ est un score associé au token $i$ (plus $z_{t,i}$ est grand, plus le token $i$ est favorisé), mais ces scores ne sont pas des probabilités : ils ne sont ni bornés ni normalisés. La Softmax convertit l’ensemble des logits $z_t$ en une distribution catégorielle sur le vocabulaire ; pour chaque $i\in\{0,\dots,\vert\mathcal V\vert-1\}$, $P(\text{token}=i\mid \text{contexte})$ désigne la probabilité que le prochain token généré soit $i$, conditionnellement au contexte (i.e. au préfixe déjà fourni et aux états internes qu’il induit) :
$$
P(\text{token}=i \mid \text{contexte})
=\frac{\exp(z_{t,i})}{\sum_{j=0}^{\vert\mathcal V\vert-1}\exp(z_{t,j})}.
$$
Cette normalisation garantit $\sum_i P(\text{token}=i\mid \text{contexte})=1$. En outre, les **écarts** de logits contrôlent directement les rapports de probabilités :
$$
\frac{P(\text{token}=i \mid \text{contexte})}{P(\text{token}=j \mid \text{contexte})}
=\exp\!\bigl(z_{t,i}-z_{t,j}\bigr).
$$

**Intuition géométrique**  
L’unembedding compare l’état interne $h_t$ à tous les vecteurs de vocabulaire : chaque score $z_{t,i}$ est un produit scalaire, donc une mesure d’alignement. À $h_t$ fixé, si deux tokens ont des embeddings proches ($w_i \approx w_j$), leurs logits tendent à être proches, car :
$$
z_{t,i} - z_{t,j} = \langle h_t^\top,\, w_i - w_j\rangle + (b_i - b_j).
$$
Sans biais (ou à biais comparable), la proximité en embedding se traduit mécaniquement par une proximité de score pour un même état interne.

**Conséquence directe du weight tying.**  
Comme l’espace des embeddings est aussi l’espace de scoring, toute direction latente qui corrèle avec un ensemble de tokens (p. ex. un champ lexical) se reflète immédiatement dans les logits : augmenter la probabilité d’un token revient à orienter $h_t$ vers le vecteur $w_i$ correspondant, tout en réduisant l’alignement avec les alternatives concurrentes.

> En §1.3, cette interprétation géométrique des logits (projection de $h_t$ sur la base vocabulaire $\{w_i\}$) sera opérationnalisée via le Logit Lens pour reconstruire la trajectoire couche par couche et caractériser la persistance d’un signal sémantique sous obfuscation.

<hr style="width:40%; margin:auto;">

### Encodage positionnel

L’opération de lookup d’embedding $e_k = W_E[i_k]$ est, par construction, **invariante à la permutation** : à ce stade, l’ordre des éléments n’est pas encodé. La séquentialité est introduite par un **mécanisme positionnel**, soit **absolu** (vecteurs de position appris ou sinusoïdaux ajoutés aux embeddings), soit **relatif** (p. ex. RoPE), typiquement implémenté via une transformation appliquée aux projections $Q/K$ au sein de l’attention.

Dans une écriture additive (positions absolues), l’entrée du premier bloc s’écrit :
$$
x_k^{(0)} = e_k + p_k,
\qquad
X^{(0)} \in \mathbb R^{n\times d_{\text{model}}}.
$$

Après indexation et injection positionnelle, l’entrée n’est plus une suite d’entiers mais une suite de vecteurs denses $x_k^{(0)}$ alimentant le **flux résiduel**. À partir de ce point, l’analyse se déplace de la chaîne symbolique (normalisation, segmentation, indexation) vers la dynamique continue (produits scalaires, normalisations, attention et MLP) en haute dimension, où se joue l’essentiel des effets exploitables par une perspective offensive.

---

## 1.2 Architecture du flux résiduel et dynamique de propagation

Une propriété structurante du Transformer est l’organisation du réseau autour du **flux résiduel** (*residual stream*). Plutôt que de remplacer la représentation à chaque étape, l’architecture maintient un **état vectoriel persistant** de dimension $d_{\text{model}}$ qui traverse tous les blocs : de l’entrée $X^{(0)}$ (embeddings de tokens + information positionnelle) jusqu’à la projection finale sur le vocabulaire (souvent appelée *unembedding*).

Dans la lecture canonique en interprétabilité mécaniste, le flux résiduel joue le rôle de **substrat commun de représentation** : toutes les sous-couches lisent un état global et n’y injectent que des **mises à jour additives**. L’attention et le MLP se comportent ainsi comme des opérateurs d’update : ils calculent une variation $\Delta X$ de même gabarit que l’état courant, puis l’agrègent au flux. La dynamique n’est donc pas un écrasement (*overwrite*), mais une **superposition** progressive de contributions.

Pour une séquence de longueur $n$, le flux résiduel est un tenseur
$$
X^{(l)} \in \mathbb{R}^{n \times d_{\text{model}}},
\qquad
x_k^{(l)} \in \mathbb{R}^{d_{\text{model}}}\ \text{(tranche à la position } k).
$$
Les deux familles de mises à jour se distinguent uniquement par leur structure :

- l’**attention** réalise un mélange **inter-positions** (échanges d’information entre tokens) ;
- le **MLP** applique une transformation non linéaire **par position** (calcul local, même gabarit tensoriel).

### Écriture standard (Pre-Norm) et décomposition en mises à jour

Dans un Transformer *pre-norm* (fréquent dans les modèles récents), un bloc $l$ s’écrit typiquement :
$$
\begin{aligned}
X'^{(l)} &= X^{(l)} + \Delta X^{(l)}_{\mathrm{MHA}},
&\qquad \Delta X^{(l)}_{\mathrm{MHA}} &= \tilde X^{(l)},\\
X^{(l+1)} &= X'^{(l)} + \Delta X^{(l)}_{\mathrm{MLP}},
&\qquad \Delta X^{(l)}_{\mathrm{MLP}} &= \mathrm{MLP}(\mathrm{Norm}_2(X'^{(l)})).
\end{aligned}
$$
La normalisation stabilise l’échelle **au moment de la lecture** ; l’addition impose que chaque sous-couche n’agisse que via une **mise à jour** injectée dans le canal commun.

En négligeant dropout et détails d’implémentation, le déroulé sur $L$ blocs se déroule télescopiquement comme une somme d’interventions :
$$
X^{(L)} = X^{(0)} + \sum_{l=0}^{L-1}\Delta X^{(l)}_{\mathrm{MHA}} + \sum_{l=0}^{L-1}\Delta X^{(l)}_{\mathrm{MLP}}.
$$
Cette écriture met en évidence un point clé : l’état initial $X^{(0)}$ demeure présent, mais son **poids effectif** dans les représentations tardives dépend de la manière dont les mises à jour successives orientent le flux dans $\mathbb{R}^{d_{\text{model}}}$.

<figure class="cm-figure">
  <img src="/assets/img/art1/Figure_2_2.png" alt="Diagramme illustrant l'accumulation du flux résiduel dans un Transformer.">
  <figcaption>
    Fig. 2 — L’“autoroute” résiduelle : l’état initial $X^{(0)}$ demeure présent, tandis que chaque sous-couche ajoute une mise à jour $\Delta X$ au flux résiduel.
  </figcaption>
</figure>

### Propriétés mécanistes à conséquence offensive

#### Additivité stricte : l’“effacement” est géométrique, pas architectural

Aucune sous-couche ne dispose d’une primitive d’effacement : elle ne fait qu’ajouter $\Delta x$ au flux. Toute disparition apparente d’une information correspond à une **compensation géométrique** (annulation partielle dans un sous-espace pertinent) et non à une opération de remplacement. Une contrainte initiale (instruction, cadrage, format) persiste donc tant qu’une dynamique ultérieure ne neutralise pas sa composante directionnelle.

**Conséquence opérationnelle.** Un contournement ne “supprime” pas un garde-fou ; il **déplace** la résultante vers un régime où ce garde-fou devient directionnellement secondaire lors des lectures successives.

#### Normalisation : la décision locale est dominée par l’orientation

La normalisation (souvent RMSNorm) contraint la **lecture** du flux par chaque sous-couche : les mises à jour sont calculées à partir d’un état renormalisé, ce qui rend l’**orientation** (direction) plus informative que la norme pour les décisions locales (produits scalaires en attention, seuils d’activation dans le MLP).

Pour RMSNorm (avec un terme de stabilité $\varepsilon>0$) :
$$
\mathrm{RMSNorm}(x)=\frac{x}{\mathrm{RMS}(x)}\odot\gamma,
\qquad
\mathrm{RMS}(x)=\sqrt{\frac{1}{d_{\text{model}}}\sum_{i=1}^{d_{\text{model}}}x_i^2+\varepsilon},
$$
où $\gamma\in\mathbb{R}^{d_{\text{model}}}$ est un gain appris (distinct par couche de normalisation : $\gamma_1,\gamma_2$). À $\gamma$ fixé, la lecture est quasi invariante à une dilatation scalaire globale : l’information “vue” est principalement portée par une direction renormalisée (modulée canal par canal par $\gamma$).

**Remarque importante.** Cette invariance est **locale** (au moment où l’on lit $x$). Elle n’implique pas une invariance globale du réseau, car le flux résiduel lui-même **n’est pas renormalisé après addition** : il accumule les mises à jour.

#### Inertie du flux et bras de levier relatif

> Dans cette sous-section, on se place à position $k$ fixée et on omet l’indice $k$ pour alléger les notations.

Bien que la lecture soit normalisée, le flux résiduel **intègre** les mises à jour sans renormalisation globale :
$$
x^{(l+1)} = x^{(l)} + \Delta x^{(l)}_{\mathrm{MHA}} + \Delta x^{(l)}_{\mathrm{MLP}}
\;\;\stackrel{\text{def}}{=}\;\;
x^{(l)} + \Delta x^{(l)}.
$$
En conséquence, la norme $\lVert x^{(l)}\rVert$ tend à croître avec la profondeur $l$ (souvent proche d’un régime $\Theta(\sqrt{l})$ si les incréments sont faiblement corrélés). Cette croissance crée une **inertie vectorielle** : plus l’état est “massif”, plus il résiste aux corrections tardives.

On quantifie la capacité d’une couche à réorienter le flux via son **bras de levier relatif** :
$$
\rho^{(l)}=\frac{\lVert\Delta x^{(l)}\rVert}{\lVert x^{(l)}\rVert}.
$$
Dans le régime $\lVert\Delta x^{(l)}\rVert\ll\lVert x^{(l)}\rVert$, l’impact directionnel est borné. En notant $\Delta x_\perp^{(l)}$ la composante de $\Delta x^{(l)}$ orthogonale à $x^{(l)}$, on obtient :

$$
\sin\angle\!\big(x^{(l)},x^{(l)}+\Delta x^{(l)}\big)
=
\frac{\lVert\Delta x_\perp^{(l)}\rVert}{\lVert x^{(l)}+\Delta x^{(l)}\rVert}
\;\le\;
\frac{\lVert\Delta x^{(l)}\rVert}{\lVert x^{(l)}\rVert-\lVert\Delta x^{(l)}\rVert}
=
\frac{\rho^{(l)}}{1-\rho^{(l)}}\quad(\rho^{(l)}<1).
$$

Ainsi, lorsque $\rho^{(l)}\to 0$ (souvent en profondeur), une sous-couche peut encore **ajouter** du contenu, mais sa capacité à **changer l’orientation** — donc à renverser une trajectoire interne — devient marginale, sauf si la mise à jour est (i) de norme comparable à $\lVert x^{(l)}\rVert$, ou (ii) exceptionnellement bien orientée dans le sous-espace décisionnel pertinent.

### Synthèse offensive : la compétition vectorielle

Du point de vue de la sécurité, l’architecture résiduelle transforme la robustesse en un problème de **compétition vectorielle**. Une correction de sûreté (p. ex. un refus) n’efface pas une perturbation ; elle doit **contrebalancer** une direction latente concurrente au fil des lectures successives.

La vulnérabilité réside souvent dans un déséquilibre de levier : une perturbation qui s’installe tôt (via $X^{(0)}$ ou les premières couches, où $\lVert x^{(l)}\rVert$ demeure modérée) peut imposer une orientation. Corriger cette orientation tardivement devient difficile à cause de l’inertie accumulée : la correction doit être d’amplitude élevée **ou** d’une précision angulaire élevée. En audit/red-team, l’enjeu n’est donc pas “d’enlever” une contrainte, mais d’identifier quelles directions dominent le flux résiduel — et à partir de quel point elles deviennent coûteuses à réorienter.

<hr style="width:40%; margin:auto;">

### Transition : du substrat aux opérateurs

À partir de $X^{(0)}$, la dynamique est gouvernée par cette accumulation de mises à jour. La section suivante formalise comment l’attention et les MLP sculptent ce flux, couche par couche, et comment cette trajectoire interne peut être caractérisée dans une perspective d’audit.

---

## 1.3 Mécanique des opérateurs : mélange temporel et mélange de canaux

Le flux résiduel ayant été établi comme support additif de l’état interne (cf. 1.2), l’enjeu devient l’identification des **opérateurs** qui y **routent** l’information et de ceux qui la **transforment**. Dans l’approche d’interprétabilité mécaniste, un bloc Transformer s’analyse comme l’alternance de deux mécanismes quasi orthogonaux, distingués par l’axe sur lequel ils agissent (Elhage et al., 2021) :

- **Attention (mélange temporel / routage inter-positions)** : opère sur l’axe de la séquence ($T$). Elle met en relation les positions en redistribuant, vers une position cible, des fragments d’état construits ailleurs. Son effet fondamental est un **transfert** (adressage + agrégation) plutôt qu’une création d’information ex nihilo.

- **MLP/FFN (mélange de canaux / transformation locale)** : opère sur l’axe des dimensions ($d_{\text{model}}$) à position fixée (**indépendamment des autres tokens**). Il transforme localement l’état en écrivant dans le résiduel des directions latentes, et se prête à une lecture en **mémoire associative** (clé–valeur) au sens de Geva et al. (2021).

Cette dichotomie structure l’analyse de sécurité : le mélange temporel gouverne la **propagation** d’une influence à travers le contexte (où et quand un signal devient accessible), tandis que le mélange de canaux gouverne sa **compilation** en variables latentes décodables (comment un signal se convertit en comportement, y compris les politiques de conformité ou de refus).

### Le bloc Transformer : deux écritures sur un même bus

**Convention.** $T$ désigne la longueur de séquence et $t$ l’index de la position courante $(1\le t\le T)$. Pour une séquence, l’état à l’entrée du bloc $l$ est $X^{(l)}\in\mathbb{R}^{T\times d_{\text{model}}}$. Dans l’architecture *pre-norm* dominante (normalisation de type LayerNorm avant chaque sous-bloc), le bloc s’exprime comme deux mises à jour additives successives sur un même flux résiduel :
$$
\begin{aligned}
\tilde X^{(l)} &:= \mathrm{Norm}_1\!\big(X^{(l)}\big),\\
\Delta X^{(l)}_{\mathrm{MHA}} &:= \mathrm{MHA}\!\big(\tilde X^{(l)}\big),
& X'^{(l)} &= X^{(l)} + \Delta X^{(l)}_{\mathrm{MHA}},\\
\Delta X^{(l)}_{\mathrm{MLP}} &:= \mathrm{MLP}\!\big(\mathrm{Norm}_2(X'^{(l)})\big),
& X^{(l+1)} &= X'^{(l)} + \Delta X^{(l)}_{\mathrm{MLP}}.
\end{aligned}
$$

L’attention et le MLP ne sont donc pas de simples “couches” homogènes, mais deux **opérateurs d’écriture** qui injectent des mises à jour $\Delta X$ sur un bus partagé (le résiduel). La lecture mécaniste met en évidence une **compétition vectorielle** : faire émerger dans $X^{(L)}$ une composante directionnelle qui, une fois décodée via $W_U$, augmente la probabilité d’une continuation non souhaitée (voir 1.2).

<hr style="width:40%; margin:auto;">

### Mélange temporel : l’attention comme routeur (commande vs charge utile)

L’attention multi-têtes (MHA) implémente un **routage causal** : à la couche $l$, chaque position $t$ agrège de l’information provenant de positions $j\le t$. L’opérateur agit sur l’état (pré-)normalisé du flux résiduel $\tilde X^{(l)}=\mathrm{Norm}_1(X^{(l)})\in\mathbb R^{T\times d_{\text{model}}}$, où $T$ est la longueur de séquence.

**Conventions (multi-têtes).** $H$ désigne le nombre de têtes. La $t$-ième ligne de $X^{(l)}$ est notée $x_t^{(l)}\in\mathbb R^{1\times d_{\text{model}}}$. Les dimensions des sous-espaces de requêtes/clés et de valeurs sont notées $d_k$ et $d_v$ (typiquement $d_k=d_v=d_{\text{model}}/H$).

#### Projections $Q/K/V$ : adressage vs contenu

Pour chaque tête $h\in\{1,\dots,H\}$, le résiduel est projeté dans trois sous-espaces via des matrices entraînables
$W_Q^{(h)}\in\mathbb R^{d_{\text{model}}\times d_k}$,
$W_K^{(h)}\in\mathbb R^{d_{\text{model}}\times d_k}$,
$W_V^{(h)}\in\mathbb R^{d_{\text{model}}\times d_v}$ :

$$
Q^{(h)} = \tilde X^{(l)}\, W_Q^{(h)},\qquad
K^{(h)} = \tilde X^{(l)}\, W_K^{(h)},\qquad
V^{(h)} = \tilde X^{(l)}\, W_V^{(h)}.
$$

- **Requêtes ($Q$)** : vecteurs de *sélection* — ils paramètrent, pour chaque $t$, le type d’information à récupérer à ce stade de la computation.
- **Clés ($K$)** : vecteurs d’*adressage* — ils rendent chaque position $j$ “retrouvable” sous une signature dans l’espace des clés.
- **Valeurs ($V$)** : vecteurs de *contenu* — ils portent l’information effectivement transférée si la position $j$ est sélectionnée.

Cette factorisation est structurante : **$Q/K$ déterminent où lire (commande)**, **$V$ détermine quoi transférer (charge utile)**.

#### Scores, masque causal, et matrice de routage

Les scores d’adressage d’une tête $h$ s’écrivent :
$$
S^{(h)}=\frac{Q^{(h)}(K^{(h)})^\top}{\sqrt{d_k}} + M,\qquad S^{(h)}\in\mathbb R^{T\times T},
\qquad
A^{(h)}=\mathrm{softmax}_{\text{ligne}}(S^{(h)}),
\qquad
O^{(h)}=A^{(h)}V^{(h)}.
$$

- $M$ est le **masque causal**, qui interdit $j>t$ (implémenté typiquement via $-\infty$ sur les scores correspondants).
- $\mathrm{softmax}_{\text{ligne}}$ rend $A^{(h)}$ **stochastique par lignes** : pour tout $t$, $\sum_j A^{(h)}_{t,j}=1$. Chaque ligne $A^{(h)}_{t,\cdot}$ définit ainsi une distribution sur les positions accessibles.
- Le terme $q_t^{(h)}(k_j^{(h)})^\top$ mesure une **compatibilité** (alignement) entre la requête de $t$ et l’adresse portée par $j$ ; à couche et tête fixées, il contrôle la masse allouée à $j$.

Point clé pour l’analyse mécaniste : **conditionnellement à $A^{(h)}$**, l’attention est un opérateur **linéaire** en $V^{(h)}$ (donc en $X^{(l)}$ via $W_V^{(h)}$). Le comportement de routage est donc principalement déterminé par la construction de $A^{(h)}$ (Query–Key + position + softmax), c’est-à-dire par la **commande**.

#### Lecture au niveau d’une position : somme pondérée (commande / charge utile)

En notant $q_t^{(h)}\in\mathbb R^{1\times d_k}$ la $t$-ième ligne de $Q^{(h)}$, $k_j^{(h)}\in\mathbb R^{1\times d_k}$ la $j$-ième ligne de $K^{(h)}$, et $v_j^{(h)}\in\mathbb R^{1\times d_v}$ la $j$-ième ligne de $V^{(h)}$, la sortie d’une tête à la position $t$ s’écrit :
$$
\mathrm{Attn}^{(h)}(x_t^{(l)})=\sum_{j\le t}
\underbrace{A^{(h)}_{t,j}}_{\text{commande}}
\;\underbrace{v_j^{(h)}}_{\text{charge utile}},
\qquad
v_j^{(h)}=\tilde x_j^{(l)}W_V^{(h)},\quad \tilde x_j^{(l)}:=\mathrm{Norm}_1(x_j^{(l)}).
$$

La commande $A^{(h)}_{t,\cdot}$ est entièrement déterminée par les compatibilités Query–Key (et le masque). En posant
$K^{(h)}_{1:t}=[k_1^{(h)};\dots;k_t^{(h)}]\in\mathbb R^{t\times d_k}$,
elle s’écrit :
$$
A^{(h)}_{t,\cdot}=\mathrm{softmax}_{\text{ligne}}\!\left(\frac{q_t^{(h)}(K^{(h)}_{1:t})^\top}{\sqrt{d_k}}+M_{t,1:t}\right),
$$
où $M_{t,1:t}$ désigne la sous-ligne causale correspondante du masque $M$ (scores $j\le t$).

La sortie multi-têtes concatène les sorties par tête puis applique une projection de recombinaison $W_O$, où $W_O \in \mathbb R^{H d_v \times d_{\text{model}}}$ (et si $d_v=d_{\text{model}}/H$, alors $H d_v = d_{\text{model}}$) :

$$
\mathrm{MHA}(\tilde X^{(l)})=\mathrm{Concat}\big(O^{(1)},\dots,O^{(H)}\big)\,W_O.
$$
Les têtes réalisent ainsi plusieurs politiques de routage en parallèle (chacune dans un sous-espace), dont les contributions sont réinjectées dans le flux résiduel.

Cette vue “routeur” est plus précise que l’intuition “l’attention raisonne” : l’attention **réalloue** vers $t$ des fragments d’état construits ailleurs ; la construction de variables internes abstraites dépend surtout de la **composition en profondeur** (attention $\rightarrow$ MLP, et itérations sur les couches).

#### Motifs mécanistes récurrents (propriétés, pas encore exploitation)

- **Induction (copie contextuelle).** Un motif largement documenté est le schéma $([\dots,A,B,\dots,A]\mapsto B)$, interprétable comme une **copie conditionnelle** et expliquant une part de l’in-context learning. Dans l’analyse par circuits, ce comportement est généralement distribué sur plusieurs couches : des têtes amont rendent disponible une information de “prédécesseur”, et des têtes d’induction routent ensuite vers la continuation observée lors d’une occurrence antérieure. ([arXiv][1])  
  (La section 1.4 reliera ce motif aux dérives par démonstration et aux régimes *many-shot*.)

- **Puits d’attention (attention sinks).** Certains modèles attribuent une masse attentionnelle disproportionnée à des positions “ancres” (souvent au début de séquence), parfois indépendamment de leur contenu. En contexte long, ce phénomène est souvent décrit comme un effet de stabilisation : conserver quelques tokens initiaux peut suffire à préserver certaines propriétés de génération lors de troncatures de la KV-cache. ([arXiv][2])  
  Mécaniquement, cela se comprend via la contrainte de normalisation du softmax (allocation de masse) et l’émergence de points de référence entrant en compétition pour la masse attentionnelle.

<hr style="width:40%; margin:auto;">

### Mélange de canaux : le MLP comme mémoire associative

Le MLP/FFN opère **position par position** : il ne route pas entre tokens, mais applique une transformation non linéaire locale qui écrit dans le flux résiduel. Dans sa forme canonique (biais omis pour lisibilité) :

$$
\mathrm{MLP}(x_{\text{pos}})=\sigma\!\left(x_{\text{pos}}W_{\text{in}}\right)\,W_{\text{out}},
\qquad
W_{\text{in}}\in\mathbb R^{d_{\text{model}}\times d_{\text{ff}}},
\quad
W_{\text{out}}\in\mathbb R^{d_{\text{ff}}\times d_{\text{model}}}.
$$

La non-linéarité $\sigma$ (souvent GELU/SiLU selon l’architecture) transforme des **détections** linéaires en coefficients d’activation qui contrôlent l’écriture dans le résiduel.

Les architectures modernes utilisent fréquemment des variantes **gated** (p. ex. SwiGLU), qui accentuent une structure “détection $\times$ écriture” :

$$
\mathrm{MLP}_{\text{gated}}(x_{\text{pos}})
=
\Big(\sigma(x_{\text{pos}}W_1)\odot (x_{\text{pos}}W_2)\Big)\,W_{\text{out}},
$$
avec $W_1,W_2\in\mathbb R^{d_{\text{model}}\times d_{\text{ff}}}$ et $\odot$ le produit de Hadamard (biais omis).

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
P(s_{t} \mid s_{<t})\quad(\text{où }t\text{ est le pas de génération})..
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
z_{t,i} = \langle h_t^\top, w_i \rangle + b_i.
$$
En considérant deux familles de sorties (complétion $c$ vs refus $r$), l’écart de score s’écrit :
$$
z_{t,c} - z_{t,r} = \langle h_t^\top, w_c - w_r \rangle + (b_c - b_r).
$$
Le *many-shot* agit en orientant $h_t$ (via les écritures attentionnelles et MLP successives) vers des régions augmentant $\langle h_t^\top, w_c\rangle$ relativement à $\langle h_t^\top, w_r\rangle$. Ce basculement peut être obtenu **sans** modification des poids, par renforcement inductif d’une trajectoire de complétion surreprésentée dans le contexte.

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
J(s)=\log P(y^\star\mid s,\theta),
$$
où $\theta$ désigne les paramètres figés du modèle. La difficulté centrale provient de la contrainte $\mathcal V^n$ : aucune descente de gradient n’est définie directement sur des symboles discrets, alors que $J$ est induite par une chaîne d’opérations continues après projection.

#### Du discret au continu : gradients dans l’espace des embeddings

En notant $i_k=\iota(s_k)$ l’ID du token en position $k$, et $e_k=W_E[i_k]\in\mathbb R^{d_{\text{model}}}$ son embedding, la rétropropagation fournit le gradient de $J$ par rapport au vecteur dense $e_k$ :
$$
\nabla_{e_k}J \;=\; \frac{\partial \log P(y^\star\mid s,\theta)}{\partial e_k}.
$$
Ce gradient définit dans $\mathbb R^{d_{\text{model}}}$ une direction locale d’augmentation de l’objectif. Le “retour” au discret s’interprète alors comme une forme de quantification/projection : parmi les embeddings réalisables $\{W_E[i]\}_{i\in\{0,\dots,\vert\mathcal V\vert-1\}}$, identifier des substitutions plausibles dont l’effet local est le plus compatible avec cette direction (au sens d’un voisinage ou d’un alignement dans l’espace des embeddings), puis valider cet effet dans le modèle complet.

#### Preuve de principe : recherche discrète guidée par gradient (famille GCG)

Les méthodes de la famille **Greedy Coordinate Gradient (GCG)** fournissent une instanciation opérationnelle du *steering* : une optimisation sur l’espace discret des séquences $\mathcal V^n$ est pilotée par une information différentielle définie dans l’espace continu des activations, sans modification des poids du modèle.

Soit un objectif scalaire $J$ (score ou contrainte ; pour une perte on maximise son opposé) et un ensemble de positions modifiables $\Omega\subseteq\{1,\dots,n\}$. En notant $e_k = W_E[i_k]$ l’embedding à la position $k$, le gradient $\nabla_{e_k}J$ fournit une **sensibilité locale** : il indique, dans $\mathbb R^{d_{\text{model}}}$, la direction de variation de l’embedding qui augmenterait $J$ (toutes choses égales par ailleurs).

Le principe de GCG consiste à exploiter cette information **continue** pour cribler une recherche **discrète**. Plutôt que d’explorer aveuglément $\mathcal V$, le gradient est projeté sur la table d’embeddings afin d’identifier des tokens dont la représentation vectorielle est la plus alignée avec une ascension de $J$. Cette étape correspond à une **relaxation** du problème discret et à l’usage d’une **approximation linéaire de premier ordre** pour proposer des candidats.

Au sens algorithmique, GCG s’interprète comme une forme d’**ascension de coordonnées gloutonne** (*greedy coordinate ascent*) sur l’ensemble de positions $\Omega$ : à chaque itération, l’optimisation modifie une ou plusieurs coordonnées (positions de tokens) en choisissant localement la mise à jour la plus favorable selon $J$, sous une politique gloutonne.

#### Mécanique d’une itération (schéma GCG)

1) **Calcul du gradient (relaxation).**  
Pour chaque position cible $k\in \Omega$, calcul de $\nabla_{e_k}J$ par rétropropagation. Ce vecteur représente la direction locale dans l’espace continu dans laquelle une modification de $e_k$ augmenterait $J$.

2) **Criblage des candidats (Top-k).**  
Sélection d’un ensemble restreint de substitutions $\mathcal C_k\subset \mathcal V$ (p. ex. les 256 meilleurs tokens) en maximisant l’alignement avec le gradient d’ascension, typiquement via un critère du type
$$
u \in \arg\max_{u\in\mathcal V} \langle W_E[u]-e_k, \nabla_{e_k}J\rangle,
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
