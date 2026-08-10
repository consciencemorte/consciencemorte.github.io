---

title: "Décodable n’est pas causal"
date: 2026-08-06
permalink: /notes/decodable-nest-pas-causal/
reading_minutes: 6
topics: [interprétabilité mécanistique, causalité, représentations]
status: revue de littérature
description: "Une représentation lisible dans les activations d’un modèle ne participe pas nécessairement au calcul qui produit sa sortie."
hero_image: "/assets/img/notes/decodable-nest-pas-causal/hero.jpg"
hero_position: "center 43%"
hero_credit: "Tijdschrift voor entomologie — image courtesy of BHL"
hero_source: "https://biodiversitylibrary.org/page/10847891"
---

La décodabilité est d’abord une propriété géométrique. Une variable est dite décodable lorsqu’une fonction de lecture peut la reconstruire à partir d’un état interne, avec une performance évaluée hors des données utilisées pour entraîner cette fonction. Dans le cas d’un probe linéaire, cela indique qu’une direction, un hyperplan ou un sous-espace sépare les classes considérées. Cette observation ne démontre cependant pas que le modèle utilise cette structure pendant l’inférence.

Trois propriétés doivent donc être distinguées : **lecture**, **influence** et **contrôle**. Un probe ou un *logit lens* demande si une information peut être extraite d’un état. Une ablation ou un *activation patching* demande si modifier cet état affecte un contraste comportemental défini. Une intervention de *steering* demande si une modification choisie de l’état permet de contrôler la sortie. Ces trois propriétés peuvent covarier, mais aucune n’implique les deux autres.

Le flux résiduel constitue l’espace de communication partagé du *Transformer*. À chaque couche, les têtes d’attention et les MLP en lisent certaines composantes par leurs projections d’entrée et y écrivent de nouveaux vecteurs par leurs projections de sortie. Une information peut donc être présente dans le flux résiduel sans appartenir aux directions effectivement lues par les composants aval. Elle peut également être redondante, apparaître après l’étape décisionnelle pertinente ou être compensée par des calculs ultérieurs.

<figure class="cm-figure cm-plate" id="figure-01">
  <div class="cm-plate-scroll" tabindex="0" aria-label="Planche 01 — faire défiler horizontalement si nécessaire">
    {% include figures/notes/decodable-nest-pas-causal/pl01-quatre-dissociations.svg %}
  </div>
  <figcaption>01 — Une direction peut être décodable dans un état sans que le remplacement de cet état modifie la sortie dans un contraste donné, et cela pour quatre raisons distinctes : elle n’est pas lue, elle est redondante, elle apparaît après la décision, ou son effet est compensé. Un probe ne permet pas de les départager.</figcaption>
</figure>

**Huang et Chang** mettent en évidence la dissociation entre lecture et influence dans des *Vision Transformers* entraînés au comptage d’objets. Un ViT découpe l’image en fragments et attribue un token à chacun ; les *object tokens* sont ceux qui recouvrent les objets à compter. S’y ajoute un token supplémentaire, le token **CLS**, rattaché à aucun fragment, et dont l’état en dernière couche alimente la tête de classification : c’est lui qui porte la prédiction.

Le protocole est un *activation patching* sur paires d’images. Le modèle est exécuté sur une image, puis l’activation d’un token choisi, à une couche choisie, est remplacée par celle du même token à la même couche, prélevée sur une exécution portant sur une image appariée qui contient un nombre d’objets différent. Si la prédiction bascule, l’état remplacé influençait la sortie.

La relation entre lecture et influence s’inverse avec la profondeur. Dans les **couches intermédiaires**, remplacer les object tokens modifie fortement la prédiction alors qu’un probe y lit mal le compte. Dans les **couches finales**, ces mêmes tokens permettent un décodage précis du compte, mais leur remplacement ne change presque rien. Le token CLS suit une troisième chronologie : il devient décodable avant d’acquérir son influence maximale sur la sortie.

Ce résultat écarte une image intuitive de la profondeur : celle d’une variable qui se préciserait couche après couche, et dont l’influence croîtrait à mesure qu’elle devient lisible. Les deux courbes ne sont pas alignées. Une représentation peut peser sur le calcul avant d’être lisible, et rester lisible après que son remplacement a cessé d’affecter la sortie. Une mesure de décodabilité prise à une couche donnée ne suffit donc pas à déterminer le moment où la variable est employée.

<figure class="cm-figure cm-plate" id="figure-02">
  <div class="cm-plate-scroll" tabindex="0" aria-label="Planche 02 — faire défiler horizontalement si nécessaire">
    {% include figures/notes/decodable-nest-pas-causal/pl02-profil-profondeur.svg %}
  </div>
  <figcaption>02 — Lecture qualitative des résultats de Huang et Chang : les object tokens présentent leur influence causale principale dans les couches intermédiaires, alors que leur décodabilité devient forte plus tard ; le token CLS suit une dynamique distincte. Les courbes sont indicatives et ne reproduisent pas les valeurs de l’article.</figcaption>
</figure>

L’exigence se durcit lorsque l’hypothèse porte sur un **circuit**, c’est-à-dire sur un chemin nommé : telle tête extrait telle composante du flux résiduel à telle couche, la déplace vers une autre position de token, y écrit un vecteur qu’une tête ou un MLP ultérieur relit. Décrire un circuit suppose donc d’identifier des lectures, des transports et des écritures, et non seulement de localiser une information. Une direction fortement informative peut n’appartenir à aucun chemin de ce type. Une représentation isolée n’est jamais un circuit à elle seule.

Les résultats de **Bal** montrent que le même problème apparaît dans l’évaluation des *sparse autoencoders* (SAE). Un SAE comporte deux moitiés. L’encodeur transforme une activation du modèle en un vecteur creux, dont presque toutes les coordonnées sont nulles. Le décodeur reconstruit l’activation à partir de ce vecteur : chacune de ses colonnes est une direction fixe de l’espace d’activation, un **atome**, et la reconstruction est une somme pondérée du petit nombre d’atomes que l’encodeur a laissés actifs.

La pratique d’évaluation courante consiste à apparier un concept connu avec l’atome dont la direction lui ressemble le plus, au sens de la similarité cosinus. Cet appariement ne porte cependant que sur le décodeur. Il ne dit rien de l’encodeur : rien ne garantit que l’atome apparié s’active effectivement lorsque le concept est présent, ni que son ablation modifie le calcul.

Bal se place dans un cadre où la réponse est connue d’avance. Les données sont engendrées à partir d’une liste de features fixée par l’expérimentateur ; un petit modèle est entraîné à les représenter en superposition ; un SAE est ensuite entraîné sur ses activations. Contrairement au cas d’un modèle de langage réel, on sait donc exactement quelles features existent, et chaque appariement peut être déclaré correct ou non.

Dans ce cadre, jusqu’à **77 %** des appariements dépassant un cosinus de **0,90** sont inertes à l’ablation lorsque le SAE a été entraîné dans de mauvaises conditions : l’atome apparié ne s’active jamais quand la feature est présente, et le retirer ne produit aucun effet. Le taux tombe à **9 %** pour un SAE correctement entraîné, sans disparaître, y compris pour des appariements dont le cosinus approche **1,000**. Un recensement mené sur un SAE de production retrouve le phénomène à plus petite échelle, avec environ **14 %** de features inertes dans l’échantillon examiné. 

L’écart entre ces deux taux est ce qui importe, car il interdit deux lectures opposées. On ne peut pas conclure que l’alignement géométrique des SAE serait sans valeur causale : à 9 %, la majorité des appariements à fort cosinus restent causalement actifs. On ne peut pas non plus traiter la qualité de l’autoencodeur comme un détail d’ingénierie : à 77 %, un SAE mal entraîné produit une carte de features dont l’essentiel des correspondances à fort cosinus ne recouvre rien de causal.

Ces valeurs ne se transportent pas telles quelles. Bal souligne que le cadre synthétique est le plus favorable qu’on puisse construire pour la procédure d’appariement, et que les nombres qu’il rapporte constituent donc un point d’étalonnage dans le cas facile, non une mesure de référence pour un SAE de production.

Le mécanisme mérite d’être détaillé, car il tient entièrement à une convention de mesure. La métrique de référence du domaine est la similarité cosinus **non signée**. Un atome dont la direction est exactement opposée à la feature cible affiche donc un cosinus signé proche de −1, une valeur non signée proche de 1, et franchit sans difficulté une barre fixée à 0,90.

Ce cas se produit lorsque deux features de la vérité-terrain sont antiparallèles. Le SAE apprend alors un axe unique pour les deux, et le même atome se retrouve apparié aux deux features, à l’une avec un signe positif, à l’autre avec un signe négatif. L’atome apparié négativement ne peut jamais s’activer, puisque les activations de l’encodeur sont positives. Dans l’environnement de référence de Bal, ces paires antipodales rendent compte de la totalité des features causalement inertes, dans le bon comme dans le mauvais autoencodeur.

Bal sépare deux causes, et elles ne pèsent pas au même endroit. L’**inertie compétitive** est une pathologie du TopK propre aux autoencodeurs dégradés : l’atome est un candidat légitime, mais d’autres remportent systématiquement la sélection parcimonieuse et il ne franchit jamais le seuil. C’est elle qui domine derrière le taux de 77 %, et elle s’atténue quand l’entraînement s’améliore. L’**inertie structurelle** est le mécanisme antipodal décrit ci-dessus ; elle subsiste dans les bons SAE et rend compte du résidu à 9 %. Améliorer l’entraînement ne l’élimine pas : un meilleur SAE apprend l’axe partagé plus précisément, ce qui rapproche encore le cosinus non signé de 1,000 et rend l’appariement plus convaincant, non moins.

Il faut noter que ces deux étapes ne sont pas la même expérience. Le taux de 77 % vient du recensement initial ; c’est le ré-audit déterministe qui isole les cinq paires antipodales et leur attribue, dans son environnement de référence, l’ensemble des features inertes des deux jeux récupérés.

Un dernier point nuance la thèse générale de cette note. La géométrie suffisait ici à détecter le problème : un atome apparié à deux features avec des signes opposés est visible dans la table avant toute intervention. Ce n’est pas la géométrie qui échoue, c’est l’usage d’une métrique qui jette le signe.

<figure class="cm-figure cm-plate" id="figure-03">
  <div class="cm-plate-scroll" tabindex="0" aria-label="Planche 03 — faire défiler horizontalement si nécessaire">
    {% include figures/notes/decodable-nest-pas-causal/pl03-sae-geometrie-causalite.svg %}
  </div>
  <figcaption>03 — Une forte similarité entre un atome du décodeur et une direction cible ne suffit pas à établir que la feature SAE correspondante est effectivement lue par l’encodeur dans les contextes pertinents. Bal observe cette inertie massivement dans les SAE dégradés et, plus rarement, dans les SAE bien entraînés.</figcaption>
</figure>

La suite du ré-audit rejoint directement la distinction posée en ouverture. L’inertie n’est pas la même dans les deux sens : les paires antipodales sont inertes **en lecture** — l’atome ne s’active pas, l’ablation ne fait rien, la feature est donc inobservable par ce canal — mais restent **fortement pilotables en écriture** à travers ce même atome, avec des spécificités de steering de l’ordre de 143 à 310 associées à un effet d’ablation exactement nul. Parler d’une feature « causalement inerte » sans autre précision revient donc à fusionner influence et contrôle, alors que le même objet se comporte de manière opposée selon le test appliqué.

Lecture et influence définissent ainsi deux axes distincts. Une représentation peut être décodable et influente, décodable mais peu influente, peu décodable mais influente, ou faible sur les deux axes. Ces quadrants décrivent des résultats expérimentaux ; ils ne constituent pas à eux seuls une taxonomie des mécanismes qui les produisent.

<figure class="cm-figure cm-plate" id="figure-04">
  <div class="cm-plate-scroll" tabindex="0" aria-label="Planche 04 — faire défiler horizontalement si nécessaire">
    {% include figures/notes/decodable-nest-pas-causal/pl04-quatre-regimes.svg %}
  </div>
  <figcaption>04 — Croiser une mesure de lecture et une intervention causale produit quatre régimes observables. Huang et Chang fournissent des exemples des deux dissociations principales : décodable mais peu influent, et influent malgré une faible décodabilité.</figcaption>
</figure>

La troisième propriété, le **contrôle**, introduit une difficulté supplémentaire. Une direction peut être discriminante sans constituer une direction d’intervention suffisamment spécifique pour modifier le comportement voulu.

<figure class="cm-figure cm-plate" id="figure-05">
  <div class="cm-plate-scroll" tabindex="0" aria-label="Planche 05 — faire défiler horizontalement si nécessaire">
    {% include figures/notes/decodable-nest-pas-causal/pl05-trois-proprietes.svg %}
  </div>
  <figcaption>05 — Lecture, influence et contrôle répondent à des questions différentes et possèdent leurs propres modes d’échec. Un résultat positif ou nul ne se convertit pas automatiquement en conclusion sur les deux autres propriétés.</figcaption>
</figure>

**Liu** étudie ce problème sur un régime d’échec précis en question-réponse médicale, l’*overthinking* : le modèle répond correctement lorsqu’on rééchantillonne sa réponse, mais se trompe lorsqu’il déroule une chaîne de raisonnement étendue. La variable à prédire est donc binaire — cette génération va-t-elle basculer dans le régime d’overthinking, ou non — et un probe linéaire tente de la lire dans le flux résiduel avant que la réponse ne soit produite.

Le probe y parvient, mais modestement. La classification binaire atteint **71,6 %** d’exactitude brute, pour une classe majoritaire à **66,7 %** : le gain réel est de **4,9 points**, avec une exactitude équilibrée de 62,3 % et une AUROC de 0,672. Liu qualifie lui-même l’effet de modeste, quoique très fiable statistiquement ($p \approx 10^{-16}$). Le probe repère bien les cas non problématiques — 90 % de rappel — mais manque deux tiers des cas d’overthinking.

Pourtant, cinq familles de *steering* linéaire fixe, couvrant **29 configurations** et **1 273** questions, ne produisent aucune amélioration nette de l’exactitude. La grandeur mesurée est l’écart d’exactitude finale par rapport à une exécution sans intervention : pour le steering ciblé sur le mode d’échec, il vaut **−0,2 point** (IC 95 % : [−2,8 ; +2,4]), sur une base de 65,1 %. Le résultat nul se retrouve sur Qwen2.5-7B et sur MMLU-STEM.

Pris isolément, ce résultat établirait seulement qu’une variable décodable n’est pas nécessairement contrôlable par cette famille d’interventions. Liu fournit cependant un diagnostic, et c’est là que le cas devient instructif.

Un écart nul ne signifie surtout pas que rien ne se produise. Le décompte par question le montre : l’intervention ciblée corrige **32 %** des questions initialement ratées et en abîme **17 %** de celles qui étaient réussies. Le comportement bouge donc beaucoup ; c’est le solde qui s’annule. Appliquée uniformément plutôt que de façon ciblée, la même direction fait chuter l’exactitude de **12,1 points**.

L’explication proposée est géométrique. La direction contrastive n’est pas indépendante des directions le long desquelles le modèle effectue la tâche : **88 %** de cette direction s’aligne sur un axe partagé « incorrect contre correct », ce qui laisse un indice de spécificité de **0,119** sur Llama et **0,152** sur Qwen. L’ordre de grandeur se lit mieux par comparaison : la direction de refus, que le steering pilote sans difficulté dans la littérature, affiche une spécificité de **0,999**. Pousser le long d’une direction aussi peu spécifique déplace le calcul de la tâche en même temps que le régime d’échec visé, ce qui est compatible avec l’annulation des effets ciblés comme avec l’effondrement sous intervention uniforme.

Un test supplémentaire renforce la lecture causale : effacer explicitement cette direction dégrade l’exactitude de **3,6 points**, alors que dix effacements de directions aléatoires donnent **+0,3 point**. La direction n’est donc pas un simple corrélat du régime d’échec ; elle participe au calcul correct.

L’échec du steering n’indique donc pas que la représentation soit causalement inerte. Il indique que la direction identifiée ne fournit pas d’axe de contrôle isolable **par une translation linéaire fixe du flux résiduel** — la famille d’interventions sur laquelle porte cette conclusion, et dont l’auteur précise qu’elle n’épuise pas les interventions possibles.

Liu montre d’ailleurs que le même signal reste exploitable autrement. Employé non pour corriger le modèle mais pour estimer après coup la fiabilité d’une génération et s’abstenir sur les moins sûres, il devance les cinq lignes de base d’incertitude testées (AUROC 0,716 sur la répartition de validation, 0,610 sur un jeu de test indépendant). Lire, influencer et contrôler restent trois choses distinctes, et l’échec de la troisième ne condamne pas l’usage de la première.

La distinction est méthodologiquement importante : un résultat nul sous intervention peut provenir de la géométrie de l’intervention elle-même, et non de l’absence de rôle du mécanisme ciblé.

<figure class="cm-figure cm-plate" id="figure-06">
  <div class="cm-plate-scroll" tabindex="0" aria-label="Planche 06 — faire défiler horizontalement si nécessaire">
    {% include figures/notes/decodable-nest-pas-causal/pl06-specificite.svg %}
  </div>
  <figcaption>06 — Une direction peut classer les états avec précision tout en recouvrant largement le calcul de la tâche, si bien qu’une translation le long de cet axe déplace les deux à la fois. Ce qui manque n’est pas l’information, mais une direction d’intervention assez sélective pour la viser seule.</figcaption>
</figure>

La dissociation inverse existe également. **Nadaf** étudie des *function vectors* dérivés de démonstrations *in-context* et observe qu’ils peuvent contrôler correctement le comportement alors que la bonne réponse n’est décodable par le *logit lens* à aucune couche. L’analyse porte sur **4 032** paires de transferts entre templates, **12 tâches** et **6 modèles** appartenant à trois familles ; le steering dépasse la précision du *logit lens* pour chaque tâche et chaque modèle étudié.

Ce résultat ne doit pas être assimilé au cas « influent mais peu décodable » de Huang et Chang, car l’instrument de lecture diffère. Le *logit lens* projette un état intermédiaire à travers la matrice d’*unembedding*, celle qui convertit un état interne en scores de vocabulaire : il demande si l’information est déjà exprimée dans la base de la sortie. Un probe entraîné, lui, peut apprendre n’importe quelle direction. Qu’une information échappe au premier sans échapper au second est donc en partie attendu, et la dissociation rapportée par Nadaf tient pour une part à cette différence d’instrument.

Elle reste informative dans un sens plus faible, mais suffisant ici : l’échec d’une lecture donnée n’autorise pas à conclure qu’un état ne contient aucune structure exploitable pour le contrôle. Nadaf interprète d’ailleurs certains *function vectors* comme des instructions de calcul plutôt que comme des directions codant la réponse elle-même, ce qui expliquerait qu’ils restent illisibles dans la base des réponses.

La conséquence méthodologique n’est pas qu’il faudrait remplacer les probes par des interventions causales. Chaque instrument possède son propre problème d’identification : un résultat, positif ou nul, admet plusieurs explications qu’il faut ensuite départager.

Un probe positif peut exploiter une information redondante ou épiphénoménale. Un *activation patching* ou une ablation nulle peut dépendre du contraste choisi, de mécanismes compensatoires ou de la granularité de l’intervention. Un steering nul peut résulter d’une direction mal alignée avec les degrés de liberté effectivement contrôlables, comme chez Liu ; symétriquement, un steering réussi établit qu’une intervention peut provoquer un comportement, mais pas que le modèle utilise spontanément cette même direction pour le produire.

Une interprétation mécanistique exige donc une **triangulation** entre mesures dont les hypothèses et les modes d’échec sont différents : probes et métriques géométriques pour la lecture, patching et ablations pour l’influence dans des contrastes spécifiés, steering pour la contrôlabilité, puis analyses de chemins ou de médiation lorsque l’hypothèse porte sur un circuit.

L’objectif n’est pas d’accumuler des tests jusqu’à obtenir une étiquette de causalité. Il est d’identifier une chaîne computationnelle avec laquelle les résultats de lecture, d’intervention et de contrôle soient conjointement compatibles — et de nommer, à chaque étape, les explications concurrentes que le test employé n’a pas écartées.

## Sources

Huang, L. et Chang, Y. (2025). *Causality ≠ Decodability, and Vice Versa: Lessons from Interpreting Counting ViTs*. arXiv:2510.09794. ([arXiv][1])

Liu, M. (2026). *Decodable but Not Corrected by Fixed Residual-Stream Linear Steering: Evidence from Medical LLM Failure Regimes*. arXiv:2605.05715. ([arXiv][2])

Bal, M. A. (2026). *From Geometric Recovery to Causal Validation: A Reproducible Audit of Sparse Autoencoder Features, from Superposition Geometry to Causal Inertness*. arXiv:2607.12166. ([arXiv][3])

Nadaf, M. S. B. (2026). *Steerable but Not Decodable: Function Vectors Operate Beyond the Logit Lens*. arXiv:2604.02608. ([arXiv][4])

Elhage, N. et al. (2021). *A Mathematical Framework for Transformer Circuits*. Transformer Circuits Thread. ([Transformer Circuits][5])

Heimersheim, S. et Nanda, N. (2024). *How to Use and Interpret Activation Patching*. arXiv:2404.15255. ([arXiv][6])

[1]: https://arxiv.org/abs/2510.09794 "Causality ≠ Decodability, and Vice Versa: Lessons from Interpreting Counting ViTs"
[2]: https://arxiv.org/abs/2605.05715 "Decodable but Not Corrected by Fixed Residual-Stream Linear Steering"
[3]: https://arxiv.org/abs/2607.12166 "From Geometric Recovery to Causal Validation"
[4]: https://arxiv.org/abs/2604.02608 "Steerable but Not Decodable: Function Vectors Operate Beyond the Logit Lens"
[5]: https://transformer-circuits.pub/2021/framework/index.html "A Mathematical Framework for Transformer Circuits"
[6]: https://arxiv.org/abs/2404.15255 "How to Use and Interpret Activation Patching"