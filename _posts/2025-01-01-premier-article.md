---
layout: post
title: "Naissance d'une conscience morte"
categories: [théorie, llm]
featured: true
math: true
---


1.1 Tokenisation et Discrétisation de l’Espace d’Entrée

Bien que l’interaction avec un Grand Modèle de Langage (LLM) apparaisse pour l'utilisateur comme un flux textuel continu, le modèle neuronal sous-jacent opère exclusivement sur des séquences discrètes d’entiers. La **tokenisation**, première transformation du pipeline d'inférence, constitue l'interface critique entre le langage naturel (symbolique) et le calcul matriciel (numérique).

### Formalisation de la Segmentation

Soit $\mathcal{S}$ l’espace des chaînes de caractères possibles. La tokenisation se définit comme une fonction de projection $\tau : \mathcal{S} \rightarrow \mathcal{V}^*$ associant à une chaîne brute une séquence de tokens $(t_1, t_2, \dots, t_n)$, où chaque $t_i$ appartient à un vocabulaire fini $\mathcal{V}$. La cardinalité $\lvert \mathcal{V} \rvert$, **fixée avant la phase d'entraînement**, oscille généralement entre $32\,000$ et $128\,000$ unités pour les architectures actuelles (LLaMA, GPT-4, etc.).

Les algorithmes de sous-mots (_subword algorithms_), tels que le **Byte-Pair Encoding (BPE)** ou SentencePiece, ne procèdent pas par une compression sémantique, mais par une compression statistique : ils réalisent une fusion itérative des paires de symboles les plus fréquentes dans le corpus d'entraînement. L'objectif est de minimiser la longueur moyenne de la séquence $(t_i)$ tout en maintenant la taille du vocabulaire $\lvert \mathcal{V} \rvert$ fixe. Il en résulte un découpage adaptatif : les termes fréquents deviennent des tokens uniques, tandis que les termes rares ou morphologiquement complexes sont décomposés en sous-unités.

Ce mécanisme induit une variabilité de représentation intrinsèque, véritable vecteur d'attaque en sécurité offensive :

1. **Variabilité Multilingue :** Un concept identique (ex: _“chat”_) est encodé par un token unique en anglais, mais fragmenté en plusieurs tokens dans des langues agglutinantes ou des écritures non-latines. La distance entre deux concepts dans l'espace des identifiants (ID) ne présage donc en rien de leur proximité sémantique.
    
2. **Sensibilité aux Perturbations (Adversarial Typos) :** Un même concept sémantique peut être segmenté de multiples manières selon des variations morphologiques infimes. Par exemple, le mot _"Malicious"_ possède souvent son propre token unique si sa fréquence est élevée. Cependant, une altération mineure comme _"Maliscious"_ forcera le tokenizer à le fragmenter en une séquence inédite, par exemple `[Mal, is, cious]`. Pour un filtre de sécurité rigide basé sur une liste noire d'IDs, la séquence `[Malicious]` est interdite, mais la séquence `[Mal, is, cious]` est invisible, bien qu'elles portent un sens similaire pour le modèle une fois projetées.
    

### Projection dans l'Espace Vectoriel (Embedding)

La transition du domaine discret vers le domaine continu s'opère via la matrice d’embedding $W_E \in \mathbb{R}^{\lvert \mathcal{V} \rvert \times d_{\text{model}}}$. Chaque token $t$, représenté conceptuellement par un vecteur _one-hot_ $x_t$, est projeté dans un espace latent dense :

$$e_t = W_E^\top x_t \in \mathbb{R}^{d_{\text{model}}}$$

Techniquement implémentée comme une table de correspondance (_lookup table_), cette matrice $W_E$ contient des vecteurs appris par rétropropagation. Dans de nombreuses architectures modernes (comme la famille GPT), cette matrice est souvent partagée (_tied_) avec la matrice de projection finale (_unembedding_), liant directement la géométrie de l'espace d'entrée aux probabilités de sortie.

> **Implication pour la sécurité :** Si un attaquant parvient à identifier la direction vectorielle correspondant à un concept interdit dans l'espace d'entrée, il sait, du fait de cette contrainte architecturale, que cette même direction maximisera la probabilité de générer ce concept en sortie. Cela simplifie la cartographie de la surface d'attaque, car il n'existe pas de "barrière de traduction" entre la représentation des prompts et celle des réponses.

C'est à ce stade que s'établit la **topologie initiale** du modèle : des tokens distincts du point de vue de l'encodage entier mais sémantiquement proches (ex: synonymes, variations typographiques), apparaissant systématiquement dans des contextes similaires, acquièrent ici des représentations vectorielles géométriquement voisines.

**Note sur l'Encodage Positionnel :** Contrairement aux RNNs, cette projection est par nature invariante à la position. Pour restaurer la séquentialité, une information de position $p_t$ (absolue ou relative, comme le _RoPE_) est additionnée au vecteur sémantique. L'entrée réelle **de la première couche de normalisation (avant le premier bloc d'attention)** est donc la superposition $h_t = e_t + p_t$.

### Asymétrie entre Surface Lexicale et Représentation Latente

L'architecture décrite ci-dessus engendre une discontinuité structurelle majeure entre la surface du texte et sa représentation interne, exploitée par les attaques d'obfuscation.

Les architectures de sécurité réelles déploient des garde-fous (guardrails) à plusieurs niveaux. On distingue souvent :

1. **Le Filtrage de surface :** Opérant sur l'espace $\mathcal{S}$ via des expressions régulières (regex) avant tokenisation, ou sur la séquence des IDs $(t_i)$ via des listes noires après tokenisation.
    
2. **Les Classifieurs externes :** Des modèles spécialisés (par exemple des modèles de type BERT finetunés pour la détection de toxicité) qui analysent le texte brut ou ses embeddings initiaux pour intercepter des catégories de contenu dangereuses avant qu'elles n'atteignent le LLM principal.


À l'inverse, le mécanisme d'attention du modèle opère sur les vecteurs $h_t$ dans l'espace latent. L'hypothèse de travail centrale en sécurité offensive est que la robustesse de cet espace vectoriel permet au modèle de reconstruire approximativement le sens d'un concept même si sa représentation de surface est altérée pour contourner les filtres de niveau 1 et 2. Empiriquement, on observe que des variations de surface relativement fortes (typos, translittérations, fragmentation) tendent à être interprétées comme le même concept sémantique par le modèle, rendant ces attaques réalistes.

Cette dissociation est exacerbée par deux phénomènes :

1. **L'Invariance par Fragmentation :** Comme vu avec l'exemple _"Maliscious"_, un mot interdit $M$, s'il est introduit avec des variations ou des espaces (ex: _t o k e n_), est décomposé en sous-tokens disjoints de l'ID original. Pourtant, la dynamique d'entraînement fait que la somme (ou la composition initiale) de leurs embeddings **tend à projeter** l'état latent dans une région de l'espace vectoriel voisine de celle du concept $M$ original. Le filtre lexical voit des débris inoffensifs ; le modèle perçoit le concept reconstitué.
    
2. **L'Alignement Cross-Lingue :** Du fait de l'entraînement massif sur des corpus multilingues, les vecteurs de mots équivalents (ex: _“apple”_ et _“pomme”_) s'alignent dans l'espace latent. Un filtre bloquant le terme anglais est souvent inopérant sur sa traduction ou sa translittération, bien que le modèle traite les deux entrées comme sémantiquement proches.
3. **L'Alignement Sémantique Cross-Lingue et Hybride :** Du fait de l'entraînement sur des corpus massifs multilingues, l'espace latent du modèle est agnostique à la langue. Les vecteurs de mots équivalents (ex: _“apple”_ et _“pomme”_) sont alignés. Un filtre bloquant le terme anglais peut être inopérant sur sa traduction.
	
	> **Un cas extreme : les attaques d'obfuscation hybrides**. Un attaquant peut construire un concept interdit en concaténant des sous-tokens issus de langues différentes (par exemple, une racine latine, un suffixe cyrillique et une terminaison anglaise). Pour un filtre lexical, c'est une soupe de caractères incohérente. Mais pour le modèle, la somme vectorielle de ces fragments, une fois projetée, reconstruit le concept sémantique interdit avec une précision suffisante pour activer les connaissances associées. Le modèle "comprend" l'intention au-delà de la barrière des langues.
    

Il existe donc une dichotomie fondamentale : la tokenisation est rigide et discrète, tandis que la sémantique est fluide et continue. C'est dans cet interstice que réside la capacité du modèle à généraliser le sens au-delà de la forme, propriété essentielle à l'intelligence du système, mais également limite structurante pour son contrôle.

---

## 1.2 Architecture du Flux Résiduel et Dynamique de Propagation

Une distinction fondamentale de l'architecture Transformer réside dans l'organisation du réseau autour du **flux résiduel** (_residual stream_). Contrairement aux architectures convolutives classiques où chaque couche recalcule une nouvelle représentation complète, le Transformer maintient un canal vectoriel continu de dimension $d_{model}$ traversant l'intégralité des couches, de l'encodage initial (_embedding_) jusqu'à la projection finale (_unembedding_).

Cette topologie implique que les blocs de calcul ne transforment pas l'information par substitution, mais par **accumulation additive**. Le flux résiduel agit comme une autoroute informationnelle où chaque couche vient lire l'état actuel et y "écrire" une mise à jour.

### Formalisation des Mises à Jour Additives et Rôle de la Normalisation

Soit $x^{(l)} \in \mathbb{R}^{d_{model}}$ l'état du flux résiduel à l'entrée du bloc $l$ (où $l \in [0, L-1]$). Chaque bloc est composé de deux sous-modules principaux : l'Attention Multi-Têtes (MHA) et un Perceptron Multicouche (MLP). Dans les architectures modernes (type LLaMA, Mistral), la normalisation est appliquée en entrée de chaque sous-couche (_Pre-Norm_).

La dynamique de propagation s'exprime par des mises à jour successives de l'état $x^{(l)}$ :

$$\begin{aligned} x'^{(l)} &= x^{(l)} + \text{MHA}(\text{Norm}(x^{(l)})) \\ x^{(l+1)} &= x'^{(l)} + \text{MLP}(\text{Norm}(x'^{(l)})) \end{aligned}$$

Deux propriétés mécanistes découlent de ce formalisme :

1. **L'Identité Privilégiée et la Mémoire Longue :** Chaque sous-couche $F$ calcule une perturbation résiduelle $\Delta x = F(x)$ qui est ajoutée linéairement. Le gradient se propage sans entrave le long du chemin principal, permettant aux informations inscrites à l'étape $t_0$ (comme une instruction système _“You are a helpful and harmless assistant”_) d'être préservées jusqu'aux couches profondes, à moins qu'une mise à jour ultérieure ne vienne spécifiquement les annuler vectoriellement.
    
2. La Prédominance de la Direction (Géométrie Sphérique) : La fonction $\text{Norm}(x)$ (telle que RMSNorm) projette le vecteur résiduel sur une hypersphère locale avant qu'il ne soit traité par les têtes d'attention ou les neurones du MLP.
    
    $$\text{RMSNorm}(x) = \frac{x}{\|x\|_2} \cdot g$$
    
    Cette opération a une conséquence majeure pour la sécurité : localement, pour une couche donnée, la magnitude absolue du signal entrant est normalisée. L'information est donc principalement encodée dans la direction (l'angle) du vecteur plutôt que dans sa longueur.
    

> **Note technique : Nuance sur l'Invariance d'Échelle**
> 
> Il serait abusif d'affirmer que le réseau complet est strictement invariant à la norme. Bien que chaque sous-bloc normalise son entrée, les mises à jour résiduelles, elles, s'accumulent sans normalisation intermédiaire dans le flux principal. La norme globale de $x^{(l)}$ tend à croître avec la profondeur. Cependant, du point de vue d'une tête d'attention ou d'un neurone spécifique, c'est la version normalisée qui est perçue. En sécurité offensive, on retient que la direction est le vecteur principal de la sémantique, et que les variations de magnitude sont souvent "écrasées" par les étapes de normalisation successives.

### Spécialisation Fonctionnelle et Distribution de la Sécurité

Bien que la séparation stricte des rôles soit débattue, le consensus actuel en interprétabilité mécaniste modélise une interaction fonctionnelle distincte, où les mécanismes de sécurité sont distribués :

1. **Le Routage (MHA) :** Le module d'Attention déplace l'information entre les positions. Du point de vue de la sécurité, des têtes d'attention "alignées" peuvent apprendre à _ne pas_ router d'informations provenant de contextes toxiques, ou à privilégier les instructions du _system prompt_ par rapport aux entrées utilisateur.
    
2. **Le Traitement et la Mémoire (MLP) :** Le module Feed-Forward opère localement comme une mémoire associative (_Key-Value memory_). Les couches MLP sont souvent le lieu où sont stockées les connaissances factuelles, mais aussi les "réflexes moraux". Des circuits spécifiques dans les MLP peuvent détecter des directions sémantiques problématiques dans le flux résiduel et y injecter une correction ou un refus.
    

### Implications pour la Sécurité : Superposition et Arithmétique Vectorielle

La vulnérabilité intrinsèque de cette architecture découle de la **disparité dimensionnelle** : le modèle doit encoder infiniment plus de concepts et de nuances que ne le permettent les $d_{model}$ dimensions orthogonales du flux résiduel. Le modèle recourt donc à la **superposition**, encodant des concepts dans des directions non-orthogonales.

Cette compression entraîne deux vecteurs d'attaque structurels majeurs :

#### 1. L'Interférence par Addition et le "Contre-poids" d'Alignement

Puisque le flux est additif ($x_{final} \approx \text{Embed} + \sum \Delta x_{MHA} + \sum \Delta x_{MLP}$), les mécanismes de sécurité issus du RLHF (Reinforcement Learning from Human Feedback) ne peuvent pas "effacer" une information.

Une hypothèse structurante féconde en interprétabilité est de modéliser l'alignement de sécurité non comme une gomme, mais comme un **contre-poids vectoriel**. Lorsqu'une requête toxique est détectée, le modèle génère une direction latente de "Refus" ($v_{refus}$) qui s'oppose à la direction de la réponse toxique.

L'attaque ne cherche donc pas à désactiver ce mécanisme, mais à ajouter une composante $v_{attaque}$ (via un prompt optimisé) telle que la somme vectorielle globale $v_{refus} + v_{attaque}$ pointe finalement vers une région de l'espace latent associée à l'acquiescement. L'attaque noie le contre-poids de sécurité sous une injection d'intentions contraires.

#### 2. La Cécité de la Normalisation aux Signaux Extrêmes

Comme la normalisation projette les entrées sur une même échelle relative avant traitement, elle crée une vulnérabilité aux attaques par saturation (comme les suffixes optimisés de type GCG - _Greedy Coordinate Gradient_).

Si une attaque parvient à générer, dans une couche précédente, une activation d'une amplitude extrêmement élevée dans une direction orthogonale au signal de sécurité, l'étape de normalisation suivante va réduire drastiquement la contribution relative du vecteur de sécurité. Le signal de refus, bien que présent, devient un "bruit de fond" angulaire imperceptible pour les couches suivantes, aveuglées par la magnitude du signal adversarial.

**En synthèse :** La sécurité du modèle ne repose pas sur une barrière binaire ("passera / passera pas"), mais sur un **équilibre dynamique de vecteurs** en compétition dans le flux résiduel. L'objectif de l'attaquant est de perturber cet équilibre géométrique, soit en ajoutant des vecteurs de complaisance, soit en exploitant la normalisation pour rendre les vecteurs de défense inopérants.


---

### 1.3 Architecture en Couches et Composition Fonctionnelle

Si la section précédente a établi la mécanique locale d'une mise à jour dans le flux résiduel, il est nécessaire de considérer le modèle dans sa globalité. Un Grand Modèle de Langage se définit mathématiquement comme une **composition profonde de transformations non-linéaires successives**.

Une fois le token d'entrée projeté dans l'espace vectoriel initial $x^{(0)}$, sa représentation traverse séquentiellement une pile de $L$ blocs identiques structurellement mais aux paramètres distincts (où $L$ atteint typiquement plusieurs dizaines, voire centaines dans les architectures récentes). Le modèle complet $F_{\theta}$ s'exprime par la composition de ces $L$ fonctions de couche :

$$x^{(L)} = F_L \circ F_{L-1} \circ \dots \circ F_1 (x^{(0)})$$

Cette structure en couches multiples est le support de l'abstraction progressive de l'information. Au fil de son transit, le vecteur résiduel subit des transformations successives : les représentations des couches basses restent fortement corrélées aux propriétés de surface (le token brut), tandis que les représentations des couches plus profondes encodent des concepts de plus haut niveau, permettant l'émergence de comportements complexes assimilables à de la planification de réponse.

#### 1.3.1 La Dichotomie Structurelle : Mélange Temporel et Mélange de Canaux

Pour appréhender le traitement de l'information, il est utile de visualiser l'état interne du modèle à un instant $t$ non pas comme un vecteur unique, mais comme une matrice de taille $[T \times d_{model}]$, où $T$ est la longueur du contexte courant et $d_{model}$ la dimension vectorielle.

L'architecture Transformer se caractérise par une séparation des traitements, alternant deux types d'opérations orthogonales au sein de chaque bloc.

1. Le Mélangeur Temporel (Time Mixing) : L'Attention Multi-Têtes

Ce module opère "horizontalement" sur la matrice. Il constitue le seul mécanisme de l'architecture permettant de croiser des informations situées à des positions temporelles différentes.

Ce mécanisme assure la contextualisation : le vecteur d'un token à la position $i$ intègre des informations provenant des positions $j \le i$ (dans le cadre d'un modèle auto-régressif contraint par un masque causal). En l'absence de ce mélangeur, le traitement de chaque token s'effectuerait dans un isolement temporel total, rendant impossible la résolution des dépendances syntaxiques ou des coréférences.

2. Le Mélangeur de Canaux (Channel Mixing) : Le Perceptron Multicouche (MLP)

Ce module opère "verticalement", position par position. Il prend le vecteur d'un token unique et mélange ses dimensions internes ($d_{model}$) de manière localement indépendante : durant cette étape, aucune interaction explicite n'a lieu entre tokens différents.

En projetant le vecteur dans une dimension intermédiaire plus élevée et en y appliquant une non-linéarité, le MLP fonctionne mécaniquement comme une mémoire associative. Il traite la représentation du token courant—précédemment enrichie du contexte par la couche d'attention—pour y appliquer des transformations apprises, telles que la récupération de faits ou l'application de règles linguistiques.

#### 1.3.2 Hiérarchie d’abstraction et "Logit Lens"

L’empilement de ces blocs induit une spécialisation fonctionnelle progressive. Cette hiérarchie peut être sondée via la technique du **Logit Lens**, qui consiste à projeter l'état intermédiaire du flux résiduel $x^{(l)}$ d'une couche donnée directement sur le vocabulaire de sortie. Cela permet d'approximer les tokens qui seraient privilégiés si une prédiction immédiate devait être effectuée à cette étape intermédiaire.

Cette analyse met en évidence une tendance empirique forte dans la répartition des tâches :

- **Couches Basses ($l \ll L/2$) :** Elles sont majoritairement associées au décodage de surface, traitant la syntaxe locale et les ambiguïtés grammaticales.
    
- **Couches Médianes ($l \approx L/2$) :** Elles semblent concentrer une grande partie des motifs associés au "raisonnement", à l'intégration de connaissances factuelles et à l'élaboration des structures de réponse.
    
- **Couches Tardives ($l \to L$) :** Elles raffinent la sortie (style, cohérence globale) et portent une part significative des comportements de refus acquis via les processus d'alignement (RLHF).
    

_Note : Cette hiérarchie demeure une approximation conceptuelle utile. En pratique, les circuits neuronaux sont distribués et les rôles fonctionnels présentent des chevauchements importants entre les couches._

#### 1.3.3 Implications pour la sécurité : Le Modèle de l'Arbitrage Vectoriel

Cette structure explique pourquoi la sécurité des LLM ne fonctionne pas comme une barrière binaire. Pour raisonner sur les attaques, il est possible de **modéliser de manière simplifiée** la décision finale comme un arbitrage géométrique dans la dernière couche du flux résiduel.

Considérons une requête malveillante. Le traitement génère différentes composantes vectorielles concurrentes dans le flux résiduel :

1. $v_{contexte\_neutre}$ : le bruit de fond lié au format et au style.
    
2. $v_{instruction\_toxique}$ : une composante latente orientée vers l'acquiescement à la demande interdite.
    
3. $v_{refus\_sécurité}$ : une composante opposée, issue des mécanismes d'alignement, orientée vers le refus.
    

Le vecteur final $v_{final}$ projeté en sortie est la résultante de ces influences. Bien que la réalité soit non-linéaire, on peut intuitivement se représenter cela comme une superposition :

$$v_{final} \approx v_{contexte\_neutre} + v_{instruction\_toxique} + v_{refus\_sécurité}$$

Une attaque n'a pas pour effet de "désactiver" mécaniquement le vecteur $v_{refus\_sécurité}$. La configuration du prompt adversarial vise à orienter la résultante $v_{final}$ dans une direction sémantiquement proche de l'acquiescement, malgré la présence du vecteur de refus. Si l'on définit $v_{cible\_toxique}$ comme la direction latente typique d'une réponse complaisante, et $v_{direction\_refus}$ comme celle d'un refus standard, l'attaque atteint son but lorsque :

$$\text{CosSim}(v_{final}, v_{cible\_toxique}) \gg \text{CosSim}(v_{final}, v_{direction\_refus})$$

Un _jailbreak_ efficace est donc un prompt capable de générer une composante $v_{instruction\_toxique}$ dont l'angle ou la magnitude sont suffisants pour que l'ajout du contre-poids $v_{refus\_sécurité}$ ne parvienne pas à extraire le vecteur final du cône d'attraction de la réponse toxique.

---

### 1.4 Le mécanisme d’attention et la dynamique de routage informationnel

L’innovation structurante de l’architecture **Transformer** (Vaswani et al., 2017) est de remplacer le goulot d’étranglement séquentiel des réseaux récurrents (RNN) par un mécanisme d’**attention par produit scalaire** (_scaled dot-product attention_).

Dans un RNN, tout l’historique $x_{<t}$ est comprimé dans un état caché $h_t$ de dimension fixe. Cette compression avec perte dilue mécaniquement les instructions initiales au fil de la génération. À l’inverse, dans un Transformer, la portée de lecture est immédiatement **globale** à chaque étape : un token peut accéder à n’importe quelle partie du contexte passé en fonction de sa pertinence latente, indépendamment de sa distance séquentielle.

Du point de vue de la sécurité, cette architecture implique qu’**aucun segment du contexte n’est protégé structurellement**. Contrairement à un système d'exploitation classique qui distingue des zones mémoires protégées (kernel space) et utilisateur (user space), le Transformer ne possède pas de "registre sécurisé" pour son _System Prompt_. L'accessibilité d'une instruction de sécurité ne dépend pas de sa position privilégiée au début du contexte, mais uniquement des poids d'attention appris qui décideront, dynamiquement, si cette instruction mérite d'être lue à l'étape $t$.

---

#### 1.4.1 Formalisation des projections : requêtes, clés, valeurs

L’opérateur d’attention ne travaille pas sur les tokens bruts. Il prend en entrée l'état courant du **flux résiduel** $x_t^{(l)}$ à la couche $l$. Ce vecteur est déjà une superposition complexe de l'embedding sémantique initial, de l'encodage positionnel, et des contributions cumulées des couches précédentes.

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

#### 1.4.2 Multi-têtes, induction et interaction avec les MLP

Les Transformers utilisent une attention multi-têtes pour paralléliser différents motifs de routage. Les travaux en interprétabilité mécaniste ont identifié des familles de têtes aux rôles distincts. Parmi elles, les **têtes d’induction** (_induction heads_) sont considérées comme le moteur mécaniste de l'**apprentissage en contexte** (_in-context learning_).

Bien que la recherche sur leur fonctionnement exact soit toujours active, elles modélisent un algorithme de copie contextuelle de type : _"Si le motif $(A, B)$ est apparu précédemment, et que le token actuel est $A$, alors porter une attention maximale sur le token qui suivait $A$ (donc $B$) pour copier sa valeur."_

C'est le mécanisme moteur des attaques de type **Many-Shot Jailbreak**.

> **Exemple mécaniste d'attaque via les têtes d'induction :**
> 
> Un attaquant sature le contexte avec 50 exemples de dialogues fictifs suivant le motif structurel `[Question Illicite] -> [Réponse Complaisante]`.
> 
> Lorsqu'il envoie finalement sa propre [Question Illicite Cibe], les têtes d'induction détectent la répétition du motif. Bien que des circuits neuronaux dédiés à la sécurité (souvent situés dans les couches MLP) soient susceptibles de générer une activation orientée vers le refus, leur signal est vectoriellement surpassé.
> 
> Les têtes d'induction, ayant alloué un poids d'attention quasi-total aux exemples précédents, "court-circuitent" le traitement sémantique profond au profit d'une copie de surface du style complaisant. Le vecteur résultant dans le flux résiduel s'aligne géométriquement avec les exemples fournis, contournant l'alignement par simple inertie mimétique.

Il faut distinguer :

- **L’Attention** (Routage) : Détermine _où_ l'information est lue dans l'historique.
    
- **Les MLP** (Traitement) : Stockent et appliquent des connaissances factuelles ou des règles (y compris morales) sur le token courant.
    
- **La Faille :** Si le mécanisme d'attention ne route pas le flux vers les zones du contexte activant les "réflexes" de sécurité du MLP, ou s'il route trop fortement vers des exemples adversariaux, le mécanisme de défense reste inactif ou est submergé.
    

---

#### 1.4.3 Implications structurelles pour la sécurité : dilution et puits

La mathématique même de l'attention définit des surfaces d'attaque structurelles, exploitant la manière dont le modèle arbitre l'information.

**(1) Dilution contextuelle et ressource finie**

La contrainte du Softmax ($\sum \alpha = 1$) impose un jeu à somme nulle. Si un attaquant injecte un grand volume de texte "bruit", il force le modèle à distribuer sa masse d'attention sur ces nouveaux tokens. Il ne s'agit pas d'un effacement déterministe des règles de sécurité du _System Prompt_, mais d'une **dilution probabiliste** de leur influence. Leur contribution vectorielle au flux résiduel devient statistiquement négligeable face à la masse des vecteurs issus du contenu adversarial.

**(2) Puits d’attention (_attention sinks_)**

Certains tokens (comme le début de séquence `<BOS>` ou des séparateurs spécifiques) agissent comme des attracteurs naturels, absorbant une part disproportionnée de l'attention. Les attaquants peuvent exploiter ce phénomène en créant des attracteurs artificiels via des formats très saillants (ex: JSON complexes, balises répétitives) pour détourner les têtes d'attention critiques de leur cible légitime (les instructions de sécurité) vers la structure du prompt adversarial.

**(3) Synthèse : le routage comme surface d’attaque**

En résumé, les attaques de prompt injection n'exploitent pas une faille logicielle classique, mais les propriétés émergentes du routage de l'attention :

1. La **compétition pour la ressource** (Dilution probabiliste).
    
2. L'**inertie mimétique** des mécanismes de copie (Têtes d'induction).
    
3. Les **artefacts de focalisation** (Puits d'attention).
