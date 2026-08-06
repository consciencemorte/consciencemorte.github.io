---
title: "Décodable n’est pas causal"
date: 2026-08-06
permalink: /notes/decodable-nest-pas-causal/
reading_minutes: 5
topics: [interprétabilité mécanistique, causalité, représentations]
status: revue de littérature
description: "Une représentation lisible dans les activations d’un modèle ne participe pas nécessairement au calcul qui produit sa sortie."
hero_image: "/assets/img/notes/decodable-nest-pas-causal/hero.jpg"
hero_position: "center 43%"
hero_credit: "Tijdschrift voor entomologie — image courtesy of BHL"
hero_source: "https://biodiversitylibrary.org/page/10847891"
---

La décodabilité est d’abord une propriété géométrique. Une variable est dite décodable lorsqu’un probe peut la reconstruire à partir d’un état interne, avec une performance évaluée hors des données qui ont servi à l’entraîner. Dans le cas d’un probe linéaire, cela indique qu’une direction, un hyperplan ou un sous-espace permet de séparer les classes considérées. Cette observation ne démontre pas que le modèle utilise cette structure pendant l’inférence.

Le flux résiduel constitue l’espace de communication partagé du *Transformer*. À chaque couche, les têtes d’attention et les MLP y lisent certaines composantes par leurs projections d’entrée, puis y écrivent de nouveaux vecteurs par leurs projections de sortie. Une information peut donc être présente dans le flux résiduel tout en restant hors des directions lues par les composants aval. Elle peut également être redondante, apparaître après l’étape décisionnelle pertinente ou être compensée par des mises à jour ultérieures.

<figure class="cm-figure cm-plate" id="figure-01">
  <div class="cm-plate-scroll" tabindex="0" aria-label="Planche 01 — faire défiler horizontalement si nécessaire">
    {% include figures/notes/decodable-nest-pas-causal/pl01-flux-residuel.svg %}
  </div>
  <figcaption>01 — Une direction accessible à un probe externe peut rester hors des projections d’entrée des composants aval. Le schéma illustre un cas possible, non un diagnostic automatique.</figcaption>
</figure>

**Huang** et **Chang** mettent en évidence cette dissociation dans des *Vision Transformers* consacrés au comptage. Dans leurs contrastes, le remplacement de *tokens objets* précoces ou intermédiaires modifie la prédiction alors que le compte y reste faiblement décodable. À l’inverse, les tokens objets finaux permettent un décodage précis sans que leur remplacement modifie la sortie. Le probe teste donc l’information accessible dans l’état ; l’activation patching teste l’effet de remplacer cet état dans un contraste propre et corrompu défini.

<figure class="cm-figure cm-plate" id="figure-02">
  <div class="cm-plate-scroll" tabindex="0" aria-label="Planche 02 — faire défiler horizontalement si nécessaire">
    {% include figures/notes/decodable-nest-pas-causal/pl02-profil-profondeur.svg %}
  </div>
  <figcaption>02 — Lecture qualitative des résultats de Huang et Chang : les tokens objets influencent tôt la sortie puis deviennent décodables tardivement ; le token CLS devient décodable avant que son remplacement n’ait un effet dans les dernières couches. Les courbes indiquent cette séquence, pas des valeurs mesurées.</figcaption>
</figure>

Cette distinction devient plus stricte au niveau des circuits d’attention. Pour passer d’une représentation à une description de circuit, il faut montrer comment elle est lue, transportée ou transformée le long d’un chemin fonctionnel. Ce chemin peut inclure une tête qui extrait une composante du flux résiduel, la déplace entre positions de tokens, puis écrit un vecteur exploité par une tête ou un MLP ultérieur. Une représentation isolée ne constitue donc pas, à elle seule, un circuit.

Le même problème concerne les *sparse autoencoders*. Dans les audits contrôlés de Bal, un atome du décodeur peut être fortement aligné avec une direction cible alors que la feature SAE correspondante ne s’active jamais lorsque cette cible est présente. Une similarité cosinus élevée établit donc une correspondance entre deux directions du décodeur ; elle ne garantit ni l’activation de l’encodeur dans les contextes pertinents, ni un effet sous ablation. L’effet d’une injection constitue encore un troisième test, qui peut se dissocier des deux premiers.

<figure class="cm-figure cm-plate" id="figure-03">
  <div class="cm-plate-scroll" tabindex="0" aria-label="Planche 03 — faire défiler horizontalement si nécessaire">
    {% include figures/notes/decodable-nest-pas-causal/pl03-sae-geometrie-causalite.svg %}
  </div>
  <figcaption>03 — Le mécanisme minimal de l’inertie observée par Bal : l’atome du décodeur est bien orienté, mais la compétition dans l’encodeur peut laisser son activation à zéro lorsque la cible est présente.</figcaption>
</figure>

<figure class="cm-figure cm-plate" id="figure-04">
  <div class="cm-plate-scroll" tabindex="0" aria-label="Planche 04 — faire défiler horizontalement si nécessaire">
    {% include figures/notes/decodable-nest-pas-causal/pl04-quatre-regimes.svg %}
  </div>
  <figcaption>04 — Croiser la performance d’un probe et l’effet d’un patching produit quatre observations possibles sur un même état. Même lorsqu’elles sont toutes deux fortes, ces mesures ne prouvent pas que la direction décodée est celle qu’utilise le modèle.</figcaption>
</figure>

**Liu** observe un écart analogue dans les régimes d’échec de LLM médicaux. L’état d’*overthinking* est modestement mais significativement décodable dans le flux résiduel, tandis que les familles testées de steering linéaire fixe ne produisent pas d’amélioration robuste. Cette conclusion porte sur ces interventions, non sur l’impossibilité de toute correction : l’étude rapporte d’ailleurs un gain préliminaire avec un adaptateur non linéaire appris. La normale du probe est discriminante sans être, pour autant, une direction de contrôle garantie.

<figure class="cm-figure cm-plate" id="figure-05">
  <div class="cm-plate-scroll" tabindex="0" aria-label="Planche 05 — faire défiler horizontalement si nécessaire">
    {% include figures/notes/decodable-nest-pas-causal/pl05-lecture-ecriture.svg %}
  </div>
  <figcaption>05 — Probe, remplacement ou ablation, et steering répondent à trois questions distinctes : accessibilité de l’information, influence dans un contraste et contrôle sous injection. Aucun résultat ne se convertit automatiquement en un autre.</figcaption>
</figure>

La **conclusion méthodologique** est restrictive. Les probes, les similarités cosinus et les métriques de reconstruction documentent la structure du flux résiduel. Une interprétation mécanistique demande de les confronter à des interventions localisées — *activation patching*, *path patching*, ablation de têtes ou d’arêtes, *steering* contrôlé et tests de médiation — avec des contrastes, des baselines et des métriques explicites. L’objectif n’est pas seulement d’identifier où l’information est accessible, mais de déterminer par quels composants elle est lue, transportée et convertie en effet comportemental.

## Sources

Huang, L. et Chang, Y. (2025). *Causality ≠ Decodability, and Vice Versa: Lessons from Interpreting Counting ViTs*. arXiv:2510.09794. ([arXiv][1])

Liu, M. (2026). *Decodable but Not Corrected by Fixed Residual-Stream Linear Steering: Evidence from Medical LLM Failure Regimes*. arXiv:2605.05715. ([arXiv][2])

Bal, M. A. (2026). *From Geometric Recovery to Causal Validation: A Reproducible Audit of Sparse Autoencoder Features, from Superposition Geometry to Causal Inertness*. arXiv:2607.12166. ([arXiv][3])

Elhage, N. et al. (2021). *A Mathematical Framework for Transformer Circuits*. Transformer Circuits Thread. ([Transformer Circuits][4])

Heimersheim, S. et Nanda, N. (2024). *How to Use and Interpret Activation Patching*. arXiv:2404.15255. ([arXiv][5])

[1]: https://arxiv.org/abs/2510.09794 "Causality ≠ Decodability, and Vice Versa: Lessons from Interpreting Counting ViTs"
[2]: https://arxiv.org/abs/2605.05715 "Decodable but Not Corrected by Fixed Residual-Stream Linear Steering"
[3]: https://arxiv.org/abs/2607.12166 "From Geometric Recovery to Causal Validation"
[4]: https://transformer-circuits.pub/2021/framework/index.html "A Mathematical Framework for Transformer Circuits"
[5]: https://arxiv.org/abs/2404.15255 "How to Use and Interpret Activation Patching"
