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

La décodabilité est d’abord une propriété géométrique. Une variable est dite décodable lorsqu’une fonction de lecture peut la reconstruire à partir d’un état interne, avec une performance évaluée hors des données utilisées pour entraîner cette fonction. Dans le cas d’un probe linéaire, cela indique qu’une direction, un hyperplan ou un sous-espace sépare les classes considérées. Cette observation ne démontre pas que le modèle utilise cette structure pendant l’inférence.

Trois propriétés doivent donc être distinguées : **lecture**, **influence** et **contrôle**. Un probe ou un *logit lens* demande si une information peut être extraite d’un état. Une ablation ou un *activation patching* demande si modifier cet état affecte un contraste comportemental défini. Une intervention de *steering* demande si une modification choisie de l’état permet de contrôler la sortie. Ces trois propriétés peuvent covarier, mais aucune n’implique les deux autres.

Le flux résiduel constitue l’espace de communication partagé du *Transformer*. À chaque couche, les têtes d’attention et les MLP en lisent certaines composantes par leurs projections d’entrée et y écrivent de nouveaux vecteurs par leurs projections de sortie. Une information peut donc être présente dans le flux résiduel sans appartenir aux directions effectivement lues par les composants aval. Elle peut également être redondante, apparaître après l’étape décisionnelle pertinente ou être compensée par des calculs ultérieurs.

<figure class="cm-figure cm-plate" id="figure-01">
  <div class="cm-plate-scroll" tabindex="0" aria-label="Planche 01 — faire défiler horizontalement si nécessaire">
    {% include figures/notes/decodable-nest-pas-causal/pl01-quatre-dissociations.svg %}
  </div>
  <figcaption>01 — Une direction peut être décodable tout en restant sans effet pour quatre raisons distinctes : elle n’est pas lue, elle est redondante, elle apparaît après la décision ou son effet est compensé. Un probe ne permet pas de les départager.</figcaption>
</figure>

**Huang et Chang** mettent directement en évidence la dissociation entre lecture et influence dans des *Vision Transformers* entraînés au comptage d’objets. Le remplacement des *object tokens* dans les **couches intermédiaires** modifie fortement la prédiction alors que le compte y reste faiblement décodable. Dans les couches finales, la relation s’inverse : les mêmes tokens permettent un décodage précis du compte, mais leur remplacement produit peu d’effet causal. Le token CLS présente une autre chronologie encore, devenant décodable avant d’acquérir son influence maximale sur la sortie.

Le résultat exclut une lecture simple de la profondeur du réseau dans laquelle une variable deviendrait progressivement plus « présente », puis plus causale à mesure que sa décodabilité augmente. Une représentation peut contribuer au calcul avant d’être facilement lisible par un probe et rester lisible après que son rôle causal principal a été transféré à d’autres états.

<figure class="cm-figure cm-plate" id="figure-02">
  <div class="cm-plate-scroll" tabindex="0" aria-label="Planche 02 — faire défiler horizontalement si nécessaire">
    {% include figures/notes/decodable-nest-pas-causal/pl02-profil-profondeur.svg %}
  </div>
  <figcaption>02 — Lecture qualitative des résultats de Huang et Chang : les object tokens présentent leur influence causale principale dans les couches intermédiaires, alors que leur décodabilité devient forte plus tard ; le token CLS suit une dynamique distincte. Les courbes sont indicatives et ne reproduisent pas les valeurs de l’article.</figcaption>
</figure>

Cette distinction devient plus stricte lorsqu’une hypothèse porte sur un circuit. Passer d’une représentation décodable à une description de circuit exige d’identifier comment cette représentation est lue, transportée ou transformée le long d’un chemin fonctionnel. Une direction peut être fortement informative sans appartenir à ce chemin ; inversement, un état faiblement aligné avec une variable sous une métrique de lecture donnée peut néanmoins jouer un rôle nécessaire dans son calcul.

Les résultats de **Bal** montrent que le même problème apparaît dans l’évaluation des *sparse autoencoders*. Une pratique courante consiste à associer une feature connue à l’atome du décodeur SAE présentant la similarité cosinus la plus élevée. Cette correspondance géométrique ne garantit pourtant pas que la feature correspondante s’active lorsque le concept cible est présent, ni que son ablation affecte le calcul.

Dans le cadre contrôlé de Bal, où la vérité-terrain des features est connue, jusqu’à **77 %** des correspondances dépassant un cosinus de **0,90** sont causalement inertes pour un SAE dégradé. Le taux tombe à **9 %** pour un SAE correctement entraîné, mais ne disparaît pas, y compris pour certaines correspondances dont le cosinus approche **1,000**. L’audit d’un SAE de production retrouve le phénomène à plus petite échelle, avec environ **14 %** de features inertes dans l’échantillon étudié.

Le contraste 77/9 est important pour l’interprétation. Il ne montre pas que l’alignement géométrique des SAE est généralement dépourvu de valeur causale ; il montre que cette correspondance peut devenir massivement trompeuse dans un SAE dégradé et qu’un résidu de dissociation subsiste même lorsque l’autoencodeur est bien entraîné. Bal distingue notamment une inertie compétitive, liée à la dynamique de l’encodeur, d’une inertie structurelle qui peut persister dans de meilleurs SAE.

<figure class="cm-figure cm-plate" id="figure-03">
  <div class="cm-plate-scroll" tabindex="0" aria-label="Planche 03 — faire défiler horizontalement si nécessaire">
    {% include figures/notes/decodable-nest-pas-causal/pl03-sae-geometrie-causalite.svg %}
  </div>
  <figcaption>03 — Une forte similarité entre un atome du décodeur et une direction cible ne suffit pas à établir que la feature SAE correspondante est effectivement lue par l’encodeur dans les contextes pertinents. Bal observe cette inertie massivement dans les SAE dégradés et, plus rarement, dans les SAE bien entraînés.</figcaption>
</figure>

Lecture et influence définissent ainsi deux axes distincts. Une représentation peut être décodable et causale, décodable mais causalement inerte, peu décodable mais influente, ou faible sur les deux mesures. Ces quadrants décrivent des résultats expérimentaux ; ils ne constituent pas à eux seuls une taxonomie des mécanismes qui les produisent.

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

**Liu** étudie ce problème dans un régime d’*overthinking* sur des tâches médicales. L’état est linéairement décodable dans le flux résiduel avec une exactitude équilibrée de **71,6 %** ((p < 10^{-16})). Pourtant, cinq familles de *steering* linéaire fixe, couvrant **29 configurations** et **1 273** observations, produisent toutes un effet proche de zéro ; le résultat est également retrouvé sur Qwen2.5-7B et sur MMLU-STEM.

Pris isolément, ce résultat établirait seulement qu’une variable décodable n’est pas nécessairement contrôlable par cette famille d’interventions. Liu fournit cependant un diagnostic supplémentaire : la direction associée à l’*overthinking* présente un recouvrement de **85–88 %** avec le calcul critique pour la tâche et un ratio de spécificité **≤ 0,152**. Des interventions non sélectives sur cette structure dégradent la performance générale. Le signal est donc présent, mais fortement intriqué avec des composantes nécessaires au calcul correct.

L’échec du steering n’indique pas ici que la représentation soit causalement inerte. Il indique que la direction identifiée ne fournit pas un axe de contrôle isolable par une translation linéaire fixe. La distinction est méthodologiquement importante : un résultat nul sous intervention peut provenir de la géométrie de l’intervention elle-même, et non de l’absence de rôle du mécanisme ciblé.

<figure class="cm-figure cm-plate" id="figure-06">
  <div class="cm-plate-scroll" tabindex="0" aria-label="Planche 06 — faire défiler horizontalement si nécessaire">
    {% include figures/notes/decodable-nest-pas-causal/pl06-specificite.svg %}
  </div>
  <figcaption>06 — Une direction peut classer les états avec précision tout en étant trop intriquée au calcul de la tâche pour fournir un levier de contrôle sélectif. L’effet nul du steering de Liu est donc compatible avec une information présente mais non spécifique.</figcaption>
</figure>

La dissociation inverse existe également. **Nadaf** étudie des *function vectors* dérivés de démonstrations *in-context* et observe qu’ils peuvent contrôler correctement le comportement alors que la bonne réponse n’est décodable par le *logit lens* à aucune couche. L’analyse porte sur **4 032** paires de transferts entre templates, **12 tâches** et **6 modèles** appartenant à trois familles ; le steering dépasse la précision du *logit lens* pour chaque tâche et chaque modèle étudié.

Ce résultat ne doit pas être identifié directement au cas « causal mais non décodable » de Huang. Le *logit lens*, le probe entraîné et l’activation patching ne mesurent pas la même propriété. Il fournit néanmoins une seconde forme de dissociation : l’échec d’une lecture donnée ne permet pas de conclure qu’un état ne contient aucune structure exploitable pour le contrôle. Nadaf interprète notamment certains *function vectors* comme des instructions computationnelles plutôt que comme des directions codant directement la réponse.

La conséquence méthodologique n’est donc pas qu’une intervention causale doive simplement remplacer un probe. Chaque instrument possède son propre problème d’identification.

Un probe positif peut exploiter une information redondante ou épiphénoménale. Un *activation patching* ou une ablation nulle peut dépendre du contraste choisi, de mécanismes compensatoires ou de la granularité de l’intervention. Un steering nul peut résulter d’une direction mal alignée avec les degrés de liberté effectivement contrôlables, comme chez Liu ; symétriquement, un steering réussi établit qu’une intervention peut provoquer un comportement, mais pas que le modèle utilise spontanément cette même direction pour le produire.

Une interprétation mécanistique exige donc une **triangulation** entre mesures dont les hypothèses et les modes d’échec sont différents : probes et métriques géométriques pour la lecture, patching et ablations pour l’influence dans des contrastes spécifiés, steering pour la contrôlabilité, puis analyses de chemins ou de médiation lorsque l’hypothèse porte sur un circuit.

L’objectif n’est pas d’accumuler des tests jusqu’à obtenir une étiquette de causalité. Il est d’identifier une chaîne computationnelle pour laquelle les résultats de lecture, d’intervention et de contrôle sont conjointement compatibles avec le mécanisme proposé, tout en conservant explicites les alternatives que chaque test laisse ouvertes.

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
