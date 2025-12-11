---
layout: post
title: "I. Architecture Transformer et discrétisation du langage"
categories: [théorie, introduction]
featured: true
math: true
hero_image: "/assets/img/art1/header.png"
---


## 1.1 Tokenisation et Discrétisation de l’Espace d’Entrée

Bien que l’interaction avec un Grand Modèle de Langage (LLM) apparaisse pour l'utilisateur comme un flux textuel continu, le modèle neuronal sous-jacent opère exclusivement sur des séquences discrètes d’entiers. La **tokenisation**, première transformation du pipeline d'inférence, constitue l'interface critique entre le langage naturel (symbolique) et le calcul matriciel (numérique).

### Formalisation de la Segmentation

Soit $\mathcal{S}$ l’espace des chaînes de caractères possibles. La tokenisation se définit comme une fonction de projection $\tau : \mathcal{S} \rightarrow \mathcal{V}^*$ associant à une chaîne brute une séquence de tokens $(t_1, t_2, \dots, t_n)$, où chaque $t_i$ appartient à un vocabulaire fini $\mathcal{V}$. La cardinalité $\lvert \mathcal{V} \rvert$, **fixée avant la phase d'entraînement**, oscille généralement entre $32\,000$ et $128\,000$ unités pour les architectures actuelles (LLaMA, GPT-4, etc.).

Les algorithmes de sous-mots (_subword algorithms_), tels que le **Byte-Pair Encoding (BPE)** ou **SentencePiece**, ne procèdent pas par une compression sémantique, mais par une compression statistique : ils réalisent une fusion itérative des paires de symboles les plus fréquentes dans le corpus d'entraînement. L'objectif est de minimiser la longueur moyenne de la séquence $(t_i)$ tout en maintenant la taille du vocabulaire $\lvert \mathcal{V} \rvert$ fixe. Il en résulte un découpage adaptatif : les termes fréquents deviennent des tokens uniques, tandis que les termes rares ou morphologiquement complexes sont décomposés en sous-unités.

Ce mécanisme induit une variabilité de représentation intrinsèque, véritable vecteur d'attaque en sécurité offensive :

1. **Variabilité Multilingue :** Un concept identique (ex: _“cat”_) peut être encodé par un token unique en anglais, mais fragmenté en plusieurs tokens dans des langues agglutinantes ou des écritures non-latines. La distance entre deux concepts dans l'espace des identifiants (ID) ne présage donc en rien de leur proximité sémantique.
    
2. **Sensibilité aux Perturbations (Adversarial Typos) :** Un même concept sémantique peut être segmenté de multiples manières selon des variations morphologiques infimes. Par exemple, le mot _"Malicious"_ peut posséder son propre token si sa fréquence est élevée. Cependant, une altération mineure comme _"Maliscious"_ pourra forcer le tokenizer à le fragmenter en une séquence inédite, par exemple `[Mal, is, cious]`. Pour un filtre de sécurité rigide basé sur une liste noire d'IDs, la séquence `[Malicious]` est interdite, mais la séquence `[Mal, is, cious]` pourrait être invisible, bien qu'elles portent un sens similaire pour le modèle une fois projetées.
    

### Projection dans l'Espace Vectoriel (Embedding)

La transition du domaine discret vers le domaine continu s'opère via la matrice d’embedding 
$W_E \in \mathbb{R}^{\lvert \mathcal{V} \rvert \times d_{\text{model}}}$. 
Chaque token $t \in \mathcal{V}$ est initialement représenté par un vecteur one-hot épars 
$x_t \in \{0,1\}^{\mathcal{V}}$ défini par

$$
(x_t)_i =
\begin{cases}
1 & \text{si } i = t, \\
0 & \text{sinon,}
\end{cases}
\quad \forall i \in \mathcal{V}.
$$

Il est ensuite projeté via $W_E$ dans un espace latent dense, où l'information sémantique est
compressée et répartie sur un nombre réduit de dimensions continues ($d_{\text{model}}$) :

$$
e_t = W_E^\top x_t \in \mathbb{R}^{d_{\text{model}}}.
$$

Techniquement implémentée comme une table de correspondance (_lookup table_), cette matrice $W_E$ contient des vecteurs appris par rétropropagation en minimisant l'erreur (la perte) de prédiction du token suivant (*Causal Language Modeling*). L'objectif est probabiliste : apprendre une distribution sur les tokens conditionnée par le contexte. En pratique, rien n’impose directement de rapprocher tel ou tel vecteur ; ce sont les contraintes de prédiction, combinées à l’Hypothèse distributionnelle ( $P(C \mid w_1) \approx P(C \mid w_2) \Rightarrow w_1 \approx w_2 $), qui poussent progressivement les tokens partageant des contextes d’apparition similaires vers des régions voisines de l’espace des embeddings.

Dans de nombreuses architectures modernes (comme la famille GPT), cette matrice est souvent partagée (*tied)* avec la matrice de projection finale (*unembedding*), liant directement la géométrie de l'espace d'entrée aux probabilités de sortie. Formellement, si l'on définit la matrice d'embedding telle que :

$$W_E \in \mathbb{R}^{\lvert \mathcal{V} \rvert \times d_{\text{model}}}$$

Alors la matrice d’*unembedding* $W_U$ est sa transposée (ou une transformation linéaire directe de celle-ci) :

$$W_U = W_E^\top \in \mathbb{R}^{d_{\text{model}} \times \lvert \mathcal{V} \rvert}$$

On considère une séquence de longueur ($\text{T}$) et on note $h_t \in \mathbb{R}^{d_{\text{model}}}$ l'état caché à l’instant $t \in {1,\dots,T}$ (aussi noté $x_t^{(L)}$ dans la littérature). Ce vecteur constitue la synthèse contextuelle de toute la séquence après traitement par l'ensemble des couches (l'état sémantique agrégé du modèle juste avant la génération). Les logits de sortie $z_t$ sur le vocabulaire s’écrivent alors :

$$z_t = W_E h_t \in \mathbb{R}^{\lvert \mathcal{V} \rvert}$$

Cette opération effectue un produit scalaire entre l'état caché courant ($h_t$) et l'embedding de chaque mot du vocabulaire stocké dans $W_E$. Concrètement, le modèle mesure le degré d'alignement (similarité cosinus non normalisée) entre sa représentation courante et les vecteurs du vocabulaire. Les tokens dont les vecteurs sont les plus colinéaires à l'état caché obtiennent les scores les plus élevés.

Ce mécanisme réalise explicitement le *weight tying* : il n'existe pas de barrière de traduction entre la représentation des prompts et celle des réponses.

> **Implication pour la sécurité :** Si un attaquant parvient à identifier la direction vectorielle correspondant à un concept interdit dans l'espace d'entrée ($W_E$), il sait, du fait de cette contrainte architecturale, que cette même direction maximisera la probabilité de générer ce concept en sortie ($z_t$). Cela simplifie la cartographie de la surface d'attaque, car il n'existe pas de "barrière de traduction" entre la représentation des prompts et celle des réponses.

C'est à ce stade que s'établit la topologie initiale du modèle. Sous la pression de l'objectif de prédiction, des tokens distincts par leur identifiant mais statistiquement interchangeables, ou fonctionnellement proches, (synonymes, variantes typographiques, ou racines communes) convergent vers des représentations vectorielles géométriquement voisines, car ils tendent à être entourés des mêmes contextes prédictifs.

**Note sur l'Encodage Positionnel :** Contrairement aux RNNs, cette projection est par nature invariante à la position. Pour restaurer la séquentialité, une information de position $p_t$ (absolue ou relative, comme le _RoPE_) est additionnée au vecteur sémantique. L'entrée réelle **de la première couche de normalisation (avant le premier bloc d'attention)** est donc la superposition $x_t^{(0)} = e_t + p_t$.

### Asymétrie entre Surface Lexicale et Représentation Latente

L'architecture décrite ci-dessus engendre une discontinuité structurelle majeure entre la surface du texte et sa représentation interne, exploitée par les attaques d'obfuscation.

Les architectures de sécurité actuelles déploient des garde-fous (guardrails) à plusieurs niveaux. On distingue souvent :

1. **Le Filtrage de surface :** Opérant sur l'espace $\mathcal{S}$ via des expressions régulières (regex) avant tokenisation, ou sur la séquence des IDs $(t_i)$ via des listes noires après tokenisation.
    
2. **Les Classifieurs externes :** Des modèles spécialisés (par exemple des modèles de type BERT finetunés pour la détection de toxicité) qui analysent le texte brut ou ses embeddings initiaux pour intercepter des catégories de contenu dangereuses avant qu'elles n'atteignent le LLM principal.


En revanche, le mécanisme d'attention du modèle opère sur les représentations vectorielles internes (ou vecteurs latents) $x^{(l)}$. L'hypothèse de travail centrale en sécurité offensive est que la robustesse de cet espace vectoriel permet au modèle de reconstruire approximativement le sens d'un concept même si sa représentation de surface est altérée pour contourner les filtres de niveau 1 et 2. Empiriquement, on observe que des variations de surface relativement fortes (typos, translittérations, fragmentation) tendent à être interprétées comme le même concept sémantique par le modèle, rendant ces attaques réalistes.

Cette dissociation est exacerbée par deux phénomènes :

1. **L'Invariance par Fragmentation :** Comme vu avec l'exemple _"Maliscious"_, un mot interdit $M$, s'il est introduit avec des variations ou des espaces (ex: _t o k e n_), est décomposé en sous-tokens disjoints de l'ID original. Pourtant, la dynamique d'entraînement fait que la somme (ou la composition initiale) de leurs embeddings **tend à projeter** l'état latent dans une région de l'espace vectoriel voisine de celle du concept $M$ original. Le filtre lexical voit des débris inoffensifs ; le modèle perçoit le concept reconstitué.
    
2. **L'Alignement Cross-Lingue et les Chimères Sémantiques** : L'entraînement multilingue rend l'espace latent agnostique à la langue : les vecteurs de _“apple”_ et _“pomme”_ y sont géométriquement alignés. Cette propriété ouvre la voie aux attaques hybrides : en concaténant des sous-tokens issus de langues différentes (ex: une racine latine associée à une désinence cyrillique), un attaquant peut crée une séquence textuelle incohérente pour un filtre lexical (une "soupe de caractères"). Cependant, pour le modèle, la somme vectorielle de ces fragments disparates converge précisément vers le concept interdit. La sémantique survit à la fragmentation linguistique, là où la surveillance syntaxique échoue.

<figure class="cm-figure">
  <img src="/assets/img/art1/gpt_respond.png" alt="Illustration de la robustesse de l’espace latent" loading="lazy">
  <figcaption>Illustration de la robustesse de l’espace latent : une entrée fragmentée est reconstruite malgré la corruption de surface.</figcaption>
</figure>


Il existe donc une dichotomie fondamentale : la tokenisation est rigide et discrète, tandis que la sémantique est fluide et continue. C'est dans cet interstice que réside la capacité du modèle à généraliser le sens au-delà de la forme, propriété essentielle à l'intelligence du système, mais également limite structurante pour son contrôle.

---

## 1.2 Architecture du flux résiduel et dynamique de propagation

Une distinction fondamentale de l'architecture Transformer réside dans l'organisation du réseau autour du flux résiduel (*residual stream*). Contrairement aux architectures convolutives classiques où chaque étape recalcule une nouvelle représentation, le Transformer maintient un canal vectoriel continu de dimension $d_{model}$ traversant l'intégralité des blocs, de l'encodage initial (*embedding*) jusqu'à la projection finale (*unembedding*).

Cette topologie implique que les blocs de calcul ne transforment pas l'information par substitution, mais par **accumulation additive**. Le flux résiduel agit comme une mémoire de travail vectorielle persistante. Chaque bloc lit l'état global courant pour calculer une transformation, puis injecte le résultat sous forme d'une perturbation additive ($\Delta x$) dans le flux principal.

Mathématiquement, cela signifie que la représentation en sortie du bloc $L$ peut être vue comme la somme directe de l'embedding initial et de toutes les interventions successives des couches :

$$x_L = x_0 + \sum_{i=0}^{L-1} F_i(x_i)$$

Cette propriété est capitale : l'information originale $x_0$ (le prompt) n'est jamais "écrasée" ou oubliée, elle est simplement noyée sous l'accumulation des vecteurs ajoutés par chaque couche."

### Formalisation des mises à jour additives et rôle de la normalisation

Soit $x^{(l)} \in \mathbb{R}^{d_{model}}$ l'état du flux résiduel à l'entrée du bloc $l$ (où $l \in [0, L-1]$). Chaque bloc est composé de deux sous-couches principales : l'Attention Multi-Têtes (MHA) et un Perceptron Multicouche (MLP). Dans les architectures modernes (type LLaMA, Mistral), la normalisation est appliquée en entrée de chaque sous-couche (_Pre-Norm_).

La dynamique de propagation s'exprime par des mises à jour successives de l'état $x^{(l)}$ :

$$\begin{aligned} x'^{(l)} &= x^{(l)} + \text{MHA}(\text{Norm}(x^{(l)})) \\ x^{(l+1)} &= x'^{(l)} + \text{MLP}(\text{Norm}(x'^{(l)})) \end{aligned}$$

Deux propriétés mécanistes découlent de ce formalisme :

1. **L'Identité Privilégiée et la Mémoire Longue :** Chaque sous-couche $F$ calcule une perturbation résiduelle $\Delta x = F(x)$ qui est ajoutée linéairement. Le gradient se propage sans entrave le long du chemin principal, permettant aux informations inscrites à l'étape $t_0$ (comme une instruction système _“You are a helpful and harmless assistant”_) d'être préservées jusqu'aux couches profondes, à moins qu'une mise à jour ultérieure ne vienne spécifiquement les annuler vectoriellement.
    
2. La Prédominance de la Direction (Géométrie Sphérique) : La fonction $\text{Norm}(x)$ (telle que RMSNorm) projette le vecteur résiduel sur une hypersphère locale avant qu'il ne soit traité par les têtes d'attention ou les neurones du MLP.
    
    $$\text{RMSNorm}(x) = \frac{x}{\|x\|_2} \cdot g$$
    
    Cette opération a une conséquence majeure pour la sécurité : localement, pour une couche donnée, la magnitude absolue du signal entrant est normalisée. L'information est donc principalement encodée dans la direction (l'angle) du vecteur plutôt que dans sa longueur (intensité).
    

> **Note technique** : **Saturation du Flux Résiduel et Inertie Sémantique**
>
> Il est inexact d'affirmer que le réseau est globalement invariant à l'échelle. Si chaque sous-bloc (Attention ou MLP) normalise effectivement son entrée via RMSNorm, les mises à jour résiduelles, elles, s'accumulent additivement sans normalisation dans le flux principal. En conséquence, la norme globale du vecteur d'état $\|x^{(l)}\|$ tend à croître avec la profondeur du réseau ($l$).
>
> Cette dynamique crée une asymétrie critique dans le traitement du signal :
>
> - L'Entrée des couches : Les têtes d'attention et les neurones perçoivent une version localement normalisée (directionnelle) du signal.
> - L'Impact des couches : La contribution additive d'une couche ($\Delta x$) est régulée par cette normalisation d'entrée et par la dynamique d'apprentissage, alors que le flux résiduel ($x$) sur lequel elle s'applique devient progressivement plus massif.
>
> Cela engendre un phénomène de **"Saturation du Flux Résiduel"** : le ratio d'influence relative $\frac{\|\Delta x\|}{\|x\|}$ tend à diminuer dans les couches profondes.
>
> En **sécurité offensive**, cela se traduit par une inertie sémantique. Les mécanismes d'alignement (comme les circuits de refus), qui cristallisent la décision morale dans les couches tardives — une fois le contexte global compris —, disposent d'un "bras de levier" vectoriel réduit. Ils peinent à dévier angulairement une trajectoire toxique qui a accumulé une magnitude importante et une cohérence directionnelle dans les couches précédentes.


### Spécialisation fonctionnelle et distribution de la sécurité

Bien que la séparation stricte des rôles soit débattue, un certain consensus en interprétabilité mécaniste met en avant une forme de spécialisation fonctionnelle, où les mécanismes de sécurité sont distribués :

1. **Le routage (MHA)** — La couche d’attention agit comme un mécanisme de copie sélective à longue portée. Mathématiquement, il permet à la position courante de « lire » le passé et d’importer une somme pondérée des vecteurs précédents.

   Du point de vue de la sécurité, le filtrage exploite la contrainte de ressource finie imposée par la softmax (la somme des poids d’attention vaut toujours 1). Des têtes « alignées » apprennent à neutraliser les contextes toxiques, non pas en les supprimant, mais en détournant leur attention : elles allouent la quasi-totalité de leur « budget attentionnel » aux instructions de sécurité (system prompt) ou à des tokens neutres (comme le début de séquence). Le contenu malveillant, pondéré par un coefficient quasi nul, n’influe alors pratiquement pas sur la mise à jour du flux résiduel.

2. **Le traitement et la mémoire (MLP)** — La couche feed-forward agit comme une vaste mémoire associative opérant sur chaque token individuellement. Contrairement à l’attention qui déplace l’information, le MLP l’enrichit ou la modifie. Il projette le flux résiduel dans une dimension intermédiaire beaucoup plus large ($d_{ff} \approx 4 \times d_{model}$) avant de le compresser à nouveau.

   Mécaniquement, on peut interpréter cette opération comme un dictionnaire de paires clé–valeur :
   - **Détection (Clés, $W_{in}$)** : la première couche agit comme un banc de détecteurs de motifs. Si le vecteur du flux résiduel s’aligne avec une « clé » spécifique (par exemple, une direction sémantique représentant un concept illicite), le neurone correspondant s’active via la non-linéarité.
   - **Écriture (Valeurs, $W_{out}$)** : l’activation de ce neurone déclenche l’ajout d’un vecteur « valeur » spécifique dans le flux résiduel.
   
   En sécurité, c’est ici que résident les circuits de refus. Lorsqu’un motif toxique est détecté par la première couche (la « clé »), la seconde couche injecte un vecteur correctif (la « valeur ») dont la direction s’oppose géométriquement à la génération de la suite toxique, orientant la trajectoire du flux vers des tokens de refus (par ex. : *« I cannot fulfill… »*).


### Implications pour la Sécurité : Inertie et Compétition Vectorielle

L'architecture additive et la dynamique de saturation ainsi décrites transforment la sécurité en un problème de géométrie vectorielle plutôt qu'en un problème de filtrage binaire.

#### 1. Additivité du flux résiduel et "contre-poids" d’alignement

Dans une architecture de type Transformer, l’état interne suit une dynamique essentiellement **additive** :

$$
x_{L} \approx x_{0} + \sum_{l=1}^{L} \Delta x^{(l)}_{MHA} + \sum_{l=1}^{L} \Delta x^{(l)}_{MLP}.
$$

À ce titre, les mécanismes de sécurité issus du RLHF ne disposent pas d’un opérateur d’**effacement** des informations latentes : ils n’interviennent que par l’ajout de nouvelles composantes dans le flux résiduel, et non par la suppression explicite de composantes existantes.

On peut alors modéliser l’alignement comme l’injection d’un **vecteur de refus** $v_{\text{refus}}$ dans un sous-espace latent associé aux comportements sûrs. Lorsqu’un motif toxique est détecté, le modèle ajoute une contribution $\Delta x_{\text{align}} \approx v_{\text{refus}}$ destinée à dévier la trajectoire de $x_L$ vers une région de l’espace latent correspondant au refus ou à la déviation de la demande initiale.

Une attaque par prompt n’essaie donc pas de “désactiver” ce mécanisme, mais de lui **superposer** un vecteur $v_{\text{attaque}}$ tel que la somme
$$
v_{\text{total}} = v_{\text{refus}} + v_{\text{attaque}}
$$
se projette majoritairement dans une direction latente associée à l’acquiescement (_compliance_) plutôt qu’au refus. En pratique, l’attaque consiste à injecter suffisamment de composantes "complaisantes" pour que le contre-poids de sécurité soit dominé au niveau de la somme vectorielle finale du flux résiduel.


#### 2. RMSNorm : Saturation d'Amplitude et Alignement Directionnel

Les architectures récentes (telles que LLaMA ou Mistral) substituent la LayerNorm classique par la RMSNorm (*Root Mean Square Normalization*). Cette opération projette le vecteur d'activation $x$ sur une hypersphère de rayon fixe.

Formellement, pour un vecteur d'entrée $x \in \mathbb{R}^d$, la transformation s'écrit :

$$\tilde{x} = \text{RMSNorm}(x) = \frac{x}{\|x\|_2} \odot \gamma$$

Où : $\gamma \in \mathbb{R}^d$ est un paramètre d'échelle apprenable (*gain*), permettant de redimensionner chaque dimension indépendamment après la normalisation.

Cette formule met en évidence une propriété critique : **l'invariance à l'échelle**. Si l'on multiplie l'entrée $x$ par un scalaire $\alpha > 0$, la sortie reste inchangée ($\text{RMSNorm}(\alpha x) = \text{RMSNorm}(x)$). L'information n'est donc plus portée par l'intensité du signal (la norme), mais exclusivement par son orientation (l'angle).

Cependant, cette propriété devient un vecteur d'attaque lorsque le flux résiduel est composé d'une somme de vecteurs. Considérons l'état du flux résiduel $x$ juste avant la normalisation comme la superposition linéaire de trois composantes :

$$x = v_{\text{sécu}} + v_{\text{adv}} + \epsilon$$

Où :
* $v_{\text{sécu}}$ représente le vecteur de sécurité (le "contre-poids" induit par l'alignement).
* $v_{\text{adv}}$ représente le vecteur induit par le prompt adversarial (l'attaque).
* $\epsilon$ représente le bruit contextuel résiduel.

L'attaque par saturation (type GCG - _Greedy Coordinate Gradient_) exploite la mécanique d'addition vectorielle. En optimisant les tokens d'entrée, l'attaque ne cherche pas à effacer $v_{\text{sécu}}$, mais à générer un vecteur $v_{\text{adv}}$ dont la norme est démesurément grande par rapport à celle du vecteur de sécurité ($\|v_{\text{adv}}\| \gg \|v_{\text{sécu}}\|$).

Lors de la sommation, le vecteur résultant $x$ s'aligne asymptotiquement sur la direction de sa composante la plus longue ($v_{\text{adv}}$). L'application de la RMSNorm fige ensuite cet alignement :

$$\lim_{\|v_{\text{adv}}\| \to \infty} \text{RMSNorm}(v_{\text{sécu}} + v_{\text{adv}}) \approx \text{RMSNorm}(v_{\text{adv}})$$

Le mécanisme de normalisation, en ramenant l'ensemble à une échelle fixe, "écrase" la contribution relative de $v_{\text{sécu}}$. Bien que le vecteur de sécurité soit toujours présent mathématiquement, sa contribution angulaire au produit scalaire des couches suivantes devient négligeable. Le réseau, qui traite l'information directionnelle, ne perçoit plus que la direction imposée par l'attaque.


**Interprétation globale.**  
La sécurité du modèle ne résulte pas d’un filtre binaire (“autorisé / interdit”), mais d’un **équilibre géométrique** entre vecteurs concurrents dans le flux résiduel.  
Deux familles d’attaques se dégagent alors :

1. **Attaques additives** : injection de vecteurs de complaisance $v_{\text{attaque}}$ qui compensent ou renversent la direction $v_{\text{refus}}$.
2. **Attaques par éblouissement (exploitation de la normalisation)** : construction de composantes $v_{\text{adv}}$ de norme extrême, souvent quasi-orthogonales à $v_{\text{refus}}$, de sorte que la RMSNorm projette l’état latent dans une direction essentiellement adversariale et **écrase la contribution angulaire** du vecteur de sécurité, le rendant pratiquement inopérant pour les couches suivantes.


> **Note technique : Superposition et interférences vectorielles** <br><br>
> La disparité dimensionnelle impose une contrainte structurelle majeure aux LLM : le modèle doit manipuler un nombre de features $N$ largement supérieur à la dimension de son flux résiduel ($N \gg d_{\text{model}}$). <br>
> Pour pallier cette limite, le réseau adopte une stratégie de superposition où les concepts sont encodés par des vecteurs $f_i$ formant un ensemble redondant et non-orthogonal. L'activation d'un concept, approximée par la projection $a_i \approx \langle f_i, r \rangle$, n'est donc jamais parfaitement isolée : elle subit le "bruit" induit par les corrélations non-nulles avec d'autres features partiellement alignés.<br><br>
> Cette compression avec perte engendre une polysémie vectorielle critique pour la sécurité. Puisqu'il existe inévitablement un chevauchement directionnel non nul entre des concepts interdits et bénins ($\langle f_{\text{forbidden}}, f_{\text{benin}} \rangle \neq 0$), il est possible de construire des séquences de tokens apparemment inoffensifs dont la combinaison linéaire génère une interférence constructive dans la direction interdite. <br>
> Cette dynamique permet d'activer artificiellement la représentation latente d'un concept prohibé via l'accumulation de signaux de surface bénins, contournant ainsi les filtres sémantiques explicites.

---

## 1.3 Architecture en Couches et Composition Fonctionnelle

Si la section précédente a établi la mécanique locale d'une mise à jour dans le flux résiduel, il est nécessaire de considérer le modèle dans sa globalité. Un Grand Modèle de Langage se définit mathématiquement comme une **composition profonde de transformations non-linéaires successives**.

Une fois le token d'entrée projeté dans l'espace vectoriel initial $x^{(0)}$, sa représentation traverse séquentiellement une pile de $L$ blocs identiques structurellement mais aux paramètres distincts (où $L$ atteint typiquement plusieurs dizaines, voire centaines dans les architectures récentes). Le modèle complet $F_{\theta}$ s'exprime par la composition de ces $L$ fonctions de couche :

$$x^{(L)} = F_L \circ F_{L-1} \circ \dots \circ F_1 (x^{(0)})$$

Cette structure en couches multiples est le support de l'abstraction progressive de l'information. Au fil de son transit, le vecteur résiduel subit des transformations successives : les représentations des couches basses restent fortement corrélées aux propriétés de surface (le token brut), tandis que les représentations des couches plus profondes encodent des concepts de plus haut niveau, permettant l'émergence de comportements complexes assimilables à de la planification de réponse.

### Dichotomie Structurelle : Mélange Temporel et Mélange de Canaux

Pour appréhender le traitement de l'information, il est utile de visualiser l'état interne du modèle à un instant $t$ non pas comme un vecteur unique, mais comme une matrice de taille $[T \times d_{model}]$, où $T$ est la longueur du contexte courant et $d_{model}$ la dimension vectorielle.

L'architecture Transformer se caractérise par une séparation des traitements, alternant deux types d'opérations complémentaires au sein de chaque bloc.

1. Le Mélangeur Temporel (Time Mixing) : L'Attention Multi-Têtes

Ce module opère "horizontalement" sur la matrice. Il constitue le seul mécanisme de l'architecture permettant de croiser des informations situées à des positions temporelles différentes.

Ce mécanisme assure la contextualisation : le vecteur d'un token à la position $i$ intègre des informations provenant des positions $j \le i$ (dans le cadre d'un modèle auto-régressif contraint par un masque causal). En l'absence de ce mélangeur, le traitement de chaque token s'effectuerait dans un isolement temporel total, rendant impossible la résolution des dépendances syntaxiques ou des coréférences.

2. Le Mélangeur de Canaux (Channel Mixing) : Le Perceptron Multicouche (MLP)

Ce module opère "verticalement", position par position. Il prend le vecteur d'un token unique et mélange ses dimensions internes ($d_{model}$) de manière localement indépendante : durant cette étape, aucune interaction explicite n'a lieu entre tokens différents.

En projetant le vecteur dans une dimension intermédiaire plus élevée et en y appliquant une non-linéarité, le MLP fonctionne mécaniquement comme une mémoire associative. Il traite la représentation du token courant—précédemment enrichie du contexte par la couche d'attention—pour y appliquer des transformations apprises, telles que la récupération de faits ou l'application de règles linguistiques.

### Hiérarchie d’abstraction et "Logit Lens"

L’empilement de ces blocs induit une spécialisation fonctionnelle progressive. Cette hiérarchie peut être sondée via la technique du **Logit Lens**, qui consiste à projeter l'état intermédiaire du flux résiduel $x^{(l)}$ d'une couche donnée directement sur le vocabulaire de sortie. Cela permet d'approximer les tokens qui seraient privilégiés si une prédiction immédiate devait être effectuée à cette étape intermédiaire.

Cette analyse met en évidence une tendance empirique forte dans la répartition des tâches :

- **Couches Basses ($l \ll L/2$) :** Elles sont majoritairement associées au décodage de surface, traitant la syntaxe locale et les ambiguïtés grammaticales.
    
- **Couches Médianes ($l \approx L/2$) :** Elles semblent concentrer une grande partie des motifs associés au "raisonnement", à l'intégration de connaissances factuelles et à l'élaboration des structures de réponse.
    
- **Couches Tardives ($l \to L$) :** Elles raffinent la sortie (style, cohérence globale) et portent une part significative des comportements de refus acquis via les processus d'alignement (RLHF).
    

_Note : Cette hiérarchie demeure une approximation conceptuelle utile. En pratique, les circuits neuronaux sont distribués et les rôles fonctionnels présentent des chevauchements importants entre les couches._

### Implications pour la sécurité : Le Modèle de l'Arbitrage Vectoriel

Cette structure explique pourquoi la sécurité des LLM ne fonctionne pas comme une barrière binaire. Pour raisonner sur les attaques, il est possible de **modéliser de manière simplifiée** la décision finale comme un arbitrage géométrique dans la dernière couche du flux résiduel.

Considérons une requête malveillante. Le traitement génère différentes composantes vectorielles concurrentes dans le flux résiduel :

1. $\epsilon$ : le bruit de fond lié au format et au style.
    
2. $v_{adv}$ : une composante latente orientée vers l'acquiescement à la demande interdite.
    
3. $v_{sécu}$ : une composante opposée, issue des mécanismes d'alignement, orientée vers le refus.
    

Le vecteur final $v_{final}$ projeté en sortie est la résultante de ces influences. Bien que la réalité soit non-linéaire, on peut intuitivement se représenter cela comme une superposition :

$$v_{final} \approx \epsilon + v_{adv} + v_{sécu}$$

Une attaque n'a pas pour effet de "désactiver" mécaniquement le vecteur $v_{sécu}$. La configuration du prompt adversarial vise à orienter la résultante $v_{final}$ dans une direction sémantiquement proche de l'acquiescement, malgré la présence du vecteur de refus. Si l'on définit $v_{adv}$ comme la direction latente typique d'une réponse complaisante, et $v_{sécu}$ comme celle d'un refus standard, l'attaque atteint son but lorsque :

$$\text{CosSim}(v_{final}, v_{adv}) \gg \text{CosSim}(v_{final}, v_{sécu})$$

Un _jailbreak_ efficace est donc un prompt capable de générer une composante $v_{adv}$ dont l'angle ou la magnitude sont suffisants pour que l'ajout du contre-poids $v_{sécu}$ ne parvienne pas à extraire le vecteur final du cône d'attraction de la réponse toxique.

---

## 1.4 Le mécanisme d’attention et la dynamique de routage informationnel

L’innovation structurante de l’architecture **Transformer** (Vaswani et al., 2017) est de remplacer le goulot d’étranglement séquentiel des réseaux récurrents (RNN) par un mécanisme d’**attention par produit scalaire** (_scaled dot-product attention_).

Dans un RNN, tout l’historique $x_{<t}$ est comprimé dans un état caché $h_t$ de dimension fixe. Cette compression avec perte dilue mécaniquement les instructions initiales au fil de la génération. À l’inverse, dans un Transformer, la portée de lecture est immédiatement **globale** à chaque étape : un token peut accéder à n’importe quelle partie du contexte passé en fonction de sa pertinence latente, indépendamment de sa distance séquentielle.

Du point de vue de la sécurité, cette architecture implique qu’**aucun segment du contexte n’est protégé structurellement**. Contrairement à un système d'exploitation classique qui distingue des zones mémoires protégées (_kernel space_) et utilisateur (_user space_), le Transformer ne possède pas de "registre sécurisé" pour son _System Prompt_. L'accessibilité d'une instruction de sécurité ne dépend pas de sa position privilégiée au début du contexte, mais uniquement des poids d'attention appris qui décideront, dynamiquement, si cette instruction mérite d'être lue à l'étape $t$.

---

### Formalisation des projections : requêtes, clés, valeurs

L’opérateur d’attention ne travaille pas sur les tokens bruts, mais sur l’état courant du flux résiduel $x_t^{(l)}$ à la couche $l$. Ce vecteur contient encore l’embedding sémantique et l’encodage positionnel initiaux, progressivement enrichis par les contributions cumulées de tous les blocs précédents.

Ce vecteur d'entrée est projeté dans trois sous-espaces fonctionnels via des matrices de poids entraînables ($W^Q, W^K, W^V$) :

- **Requête ($Q$)** : Encode le besoin informationnel du token courant à la couche actuelle.
    
- **Clé ($K$)** : Encode l’identité adressable de chaque position passée dans le contexte.
    
- **Valeur ($V$)** : Contient le contenu informationnel effectif qui sera extrait si la position est sélectionnée.
    

L’**attention par produit scalaire normalisé** est définie par :

$$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

Le mécanisme se déroule en trois temps :

1. **Calcul de similarité ($QK^\top$) :** Mesure une proximité géométrique entre ce que cherche le token courant ($Q$) et ce que proposent les tokens passés ($K$).
    
2. **Compétition (Softmax) :** Les scores sont transformés en une distribution de probabilité $\alpha_{t,\cdot}$ telle que $\sum_i \alpha_{t,i} = 1$. C'est une **ressource finie** : augmenter l'attention sur un token diminue mécaniquement l'attention portée aux autres.
    
3. **Agrégation ($y_t = \sum_i \alpha_{t,i} v_i$) :** Le résultat est une somme pondérée des vecteurs _Valeurs_, qui est ensuite réinjectée dans le flux résiduel.
    

On peut interpréter ce mécanisme comme une **mémoire adressable par le contenu** (_content-addressable memory_) : le modèle ne lit pas à une adresse mémoire fixe, mais à "l'adresse sémantique" correspondant à son besoin informationnel actuel.

---

### Têtes d’Induction et Algorithmique de la Copie

Les travaux en interprétabilité mécaniste (notamment *Olsson et al., 2022*) ont isolé des circuits fonctionnels au sein des couches d'attention : les **têtes d’induction** (*induction heads*). Ces structures constituent le substrat opérationnel de l'**Apprentissage en Contexte** (*In-Context Learning*), permettant au modèle de réduire son erreur de prédiction sur de nouvelles tâches sans modification des poids $\theta$.

Contrairement à une mémorisation "par cœur" (liée aux poids du MLP), une tête d'induction implémente un algorithme de **complétion de motif par adressage indirect**. Elle opère sur les représentations latentes du flux résiduel.

Soit $x_i$ l'état vectoriel à la position courante $i$. Le mécanisme peut être modélisé ainsi :

1.  **Recherche (Matching) :** La tête compare la requête courante $Q_i$ aux clés passées $K_{<i}$. Elle cherche une position $j$ dans l'historique dont le contenu sémantique est similaire à l'état actuel ($Q_i \approx K_j$).
2.  **Décalage et Extraction (Copying) :** Si une correspondance est trouvée en $j$, la tête porte son attention non pas sur $j$, mais sur la position suivante $j+1$, pour en extraire le vecteur valeur $V_{j+1}$.

Ce mécanisme induit une boucle de rétroaction positive :

$$
\text{Si } \text{Sim}(x_j, x_i) \text{ est élevée} \implies \text{Attention}(i) \rightarrow (j+1)
$$

Le vecteur $V_{j+1}$ injecté dans le flux résiduel favorise alors, lors de la projection finale, la génération d'un token cohérent avec celui qui suivait le motif original.

#### Vecteurs d'Attaque : Saturation Contextuelle et Many-Shot Jailbreak

Les attaques de type **Many-Shot Jailbreak** exploitent cette mécanique en transformant l'inférence en une compétition d'algèbre linéaire entre les *priors* ancrés dans les poids et les évidences fournies par le contexte.

L'attaque sature la fenêtre d'attention avec $N$ exemples (ex: $N=100$) structurés selon le motif $M : [\text{Query}_{\text{Illicite}} \to \text{Response}_{\text{Complaisante}}]$. Cette répétition force les têtes d'induction à accumuler des vecteurs d'états correspondant aux réponses complaisantes des exemples précédents.

La dynamique résultante dans le flux résiduel final $x^{(L)}$ peut être modélisée comme la superposition de deux champs de force antagonistes :

1.  **Le Prior de Sécurité ($v_{\text{RLHF}}$)** : Généré principalement par les mémoires associatives des MLP, ce vecteur tend à orienter la projection finale vers des tokens de refus. Sa magnitude est structurellement bornée pour une entrée donnée.
    
2.  **L'Évidence Contextuelle ($v_{\text{ICL}}$)** : Généré par la somme des contributions des têtes d'induction, ce vecteur pointe vers une direction sémantique de "complaisance". Sa norme croît fonctionnellement avec le nombre d'exemples $N$ et la cohérence du motif $M$.
    

Le basculement (*jailbreak*) survient lorsque la magnitude de l'évidence contextuelle domine celle du prior de sécurité :

$$
\| v_{\text{ICL}}(N) \| \gg \| v_{\text{RLHF}} \| \implies \operatorname{Argmax}(W_U x^{(L)}) \in \text{Complaisance}
$$

#### Contournement des Filtres MLP par Pré-conditionnement

Cette dominance vectorielle neutralise fonctionnellement les couches de sécurité situées en aval. Le flux résiduel transmis aux dernières couches est "pré-conditionné" : il possède une norme élevée et une direction fortement orthogonale au sous-espace de refus.

Même si les neurones de sécurité s'activent (détectant la toxicité latente) et injectent une correction additive $\Delta x_{\text{secu}}$, cette contribution est vectoriellement noyée.
Géométriquement, le vecteur d'état $x$ est poussé si loin dans la direction de la complaisance que la correction $\Delta x_{\text{secu}}$ ne suffit pas à ramener la trajectoire dans le cône d'attraction des logits de refus. Le réseau ne "décide" pas d’ignorer la sécurité ; l'arithmétique des vecteurs rend simplement la région de refus inaccessible.


### Implications structurelles pour la sécurité : dilution et puits

La mathématique même de l'attention définit des surfaces d'attaque structurelles, exploitant la manière dont le modèle arbitre l'information.

**(1) Dilution contextuelle et ressource finie**

La contrainte du Softmax ($\sum \alpha = 1$) impose un jeu à somme nulle. Si un attaquant injecte un grand volume de texte "bruit", il force le modèle à distribuer sa masse d'attention sur ces nouveaux tokens. Il ne s'agit pas d'un effacement déterministe des règles de sécurité du _System Prompt_, mais d'une **dilution probabiliste** de leur influence. Leur contribution vectorielle au flux résiduel devient statistiquement négligeable face à la masse des vecteurs issus du contenu adversarial.


**(2) Puits d’attention et conservation de la masse de probabilité**

Ce phénomène est une conséquence directe de la normalisation Softmax, qui impose une contrainte de conservation sur les poids d’attention : $\sum_i \alpha_{t,i} = 1$. Mathématiquement, la matrice d'attention *doit* allouer l'intégralité de sa masse de probabilité à chaque pas de temps, indépendamment de la pertinence sémantique du contexte.

- **Mécanisme de neutralisation (No-Op).** Lorsque le contexte courant ne contient pas d'information pertinente pour une tête d'attention donnée, le réseau doit éviter d'intégrer du bruit dans le flux résiduel. La dynamique d'optimisation conduit alors souvent les têtes d'attention à assigner des poids élevés à des tokens positionnels fixes (comme `<BOS>` ou des délimiteurs).
Cette concentration n'est pas une "lecture" du token, mais une stratégie mécanique : les vecteurs *Valeurs* ($V$) associés à ces positions agissent comme des vecteurs quasi-nuls ou des biais statiques. Une forte attention sur ces tokens se traduit donc par une opération neutre (proche de l'identité) dans le flux résiduel, préservant l'état courant.

- **Exploitation par saturation structurelle.** Les attaques exploitent cette mécanique de répartition. L'injection de structures syntaxiques denses ou répétitives (JSON complexes, séquences à haute entropie) génère artificiellement des scores de similarité ($QK^\top$) élevés pour certaines têtes d'induction ou de syntaxe.
En vertu de la Softmax, cette augmentation locale des scores sur les tokens adversariaux entraîne mécaniquement l'écrasement des coefficients $\alpha$ associés aux autres parties du contexte, notamment le *System Prompt*. Les instructions de sécurité ne sont pas "oubliées" par le modèle, mais leur contribution vectorielle est diluée mathématiquement jusqu'à devenir négligeable face à la masse allouée aux structures parasites.


**(3) Synthèse : Isomorphisme Instruction-Donnée et Confusion des Plans**

Au-delà des dynamiques de routage, la vulnérabilité critique de l'architecture Transformer réside dans son **monisme architectural** : l’absence de séparation physique ou logique entre les signaux de commande et le contenu à traiter. Tout est injecté dans un même canal computationnel, sans cloisonnement explicite entre ce qui relève du contrôle et ce qui relève des données.

Dans les systèmes d’exploitation sécurisés (inspirés de l’architecture de Harvard ou via des protections de type *NX bit*), le code exécutable (plan de contrôle) et les entrées utilisateurs (plan de données) sont strictement ségrégués. Le processeur dispose de mécanismes matériels empêchant l’exécution de données, et le passage d’un plan à l’autre est régulé par des primitives bien définies. À l’inverse, le LLM exacerbe la vulnérabilité des architectures de **von Neumann** : il opère sur un canal unifié où le *System Prompt* et l’entrée utilisateur sont sérialisés dans le même **flux résiduel**, sans frontière structurelle.

Mécaniquement, cette fragilité s’explique par l’agnosticisme des poids du modèle. Les matrices de projection ($W_Q, W_K, W_V$) et les couches MLP appliquent exactement les mêmes transformations aux tokens d’instruction (« Tu es un assistant… ») et aux tokens de données (« Ignore l’instruction précédente… »). Il n’existe aucun “bit de privilège” ni métadonnée vectorielle persistante qui immuniserait les vecteurs issus du *System Prompt* contre les opérations de mélange (*mixing*) dans l’espace résiduel ; pour le modèle, il ne s’agit que de positions dans une séquence et de vecteurs dans le même espace.

Cette indistinction se traduit par une véritable **confusion des plans** (Control/Data Plane Confusion). Le modèle ne traite pas les instructions de sécurité comme des règles inviolables (contraintes dures), mais comme un simple contexte sémantique supplémentaire (contraintes douces) mis en compétition avec le reste du prompt. Les attaques par injection (saturation, induction, réécriture explicite des consignes, etc.) ne sont donc pas des dysfonctionnements accidentels, mais l’exploitation logique de cette **équivalence topologique** entre instruction et donnée. Tant que la sécurité reposera sur une attention sémantique apprise plutôt que sur une isolation structurelle de registres ou de segments, le « pare-feu » du modèle restera fondamentalement probabiliste et vulnérable à la manipulation arithmétique du contexte.
