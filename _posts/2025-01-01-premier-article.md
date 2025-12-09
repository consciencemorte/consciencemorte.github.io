---
layout: post
title: "Naissance d'une conscience morte"
categories: [théorie, llm]
featured: true
math: true
---

## 1.1 Tokenisation et Discrétisation de l’Espace d’Entrée

Bien que l’interaction avec un Grand Modèle de Langage (LLM) apparaisse pour l'utilisateur comme un flux textuel continu, le modèle neuronal sous-jacent opère exclusivement sur des séquences discrètes d’entiers. La **tokenisation**, première transformation du pipeline d'inférence, constitue l'interface critique entre le langage naturel (symbolique) et le calcul matriciel (numérique).

### Formalisation de la Segmentation

Soit $\mathcal{S}$ l’espace des chaînes de caractères possibles. La tokenisation se définit comme une fonction de projection $\tau : \mathcal{S} \rightarrow \mathcal{V}^*$ associant à une chaîne brute une séquence de tokens $(t_1, t_2, \dots, t_n)$, où chaque $t_i$ appartient à un vocabulaire fini $\mathcal{V}$. La cardinalité $|\mathcal{V}|$, **fixée avant la phase d'entraînement**, oscille généralement entre $32\,000$ et $128\,000$ unités pour les architectures actuelles (LLaMA, GPT-4, etc.).

Les algorithmes de sous-mots (_subword algorithms_), tels que le **Byte-Pair Encoding (BPE)** ou SentencePiece, ne procèdent pas par une compression sémantique, mais par une compression statistique : ils réalisent une fusion itérative des paires de symboles les plus fréquentes dans le corpus d'entraînement. L'objectif est de minimiser la longueur moyenne de la séquence $(t_i)$ tout en maintenant la taille du vocabulaire $|\mathcal{V}|$ fixe. Il en résulte un découpage adaptatif : les termes fréquents deviennent des tokens uniques, tandis que les termes rares ou morphologiquement complexes sont décomposés en sous-unités.

Ce mécanisme induit une variabilité de représentation intrinsèque, véritable vecteur d'attaque en sécurité offensive :

1. **Variabilité Multilingue :** Un concept identique (ex: _“chat”_) est encodé par un token unique en anglais, mais fragmenté en plusieurs tokens dans des langues agglutinantes ou des écritures non-latines. La distance entre deux concepts dans l'espace des identifiants (ID) ne présage donc en rien de leur proximité sémantique.
    
2. **Sensibilité aux Perturbations (Adversarial Typos) :** Un même concept sémantique peut être segmenté de multiples manières selon des variations morphologiques infimes. Par exemple, le mot _"Malicious"_ possède souvent son propre token unique si sa fréquence est élevée. Cependant, une altération mineure comme _"Maliscious"_ forcera le tokenizer à le fragmenter en une séquence inédite, par exemple `[Mal, is, cious]`. Pour un filtre de sécurité rigide basé sur une liste noire d'IDs, la séquence `[Malicious]` est interdite, mais la séquence `[Mal, is, cious]` est invisible, bien qu'elles portent un sens similaire pour le modèle une fois projetées.
    

### Projection dans l'Espace Vectoriel (Embedding)

La transition du domaine discret vers le domaine continu s'opère via la matrice d’embedding $W_E \in \mathbb{R}^{|\mathcal{V}| \times d_{\text{model}}}$. Chaque token $t$, représenté conceptuellement par un vecteur _one-hot_ $x_t$, est projeté dans un espace latent dense :

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
