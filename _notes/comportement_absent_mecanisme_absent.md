---
title: "Un comportement absent n’est pas un mécanisme absent"
date: 2026-08-07
permalink: /notes/comportement-absent-nest-pas-mecanisme-absent/
reading_minutes: 4
topics: [interprétabilité mécanistique, unlearning, évaluation comportementale]
status: revue de littérature
description: "L’absence d’un comportement observable ne démontre pas que le mécanisme qui le produisait a disparu."
hero_image: "/assets/img/notes/comportement-absent-nest-pas-mecanisme-absent/hero.jpg"
hero_position: "center 18%"
hero_credit: "Tijdschrift voor entomologie — image courtesy of BHL"
hero_source: "https://biodiversitylibrary.org/page/10847895"
---

L’absence d’un comportement observable ne constitue pas une preuve de disparition du mécanisme qui le produisait. Elle établit seulement que, sous une distribution donnée de prompts et de contextes, le chemin computationnel correspondant ne domine plus la sortie. Le mécanisme peut rester encodé dans les paramètres, partiellement représenté dans le flux résiduel, ou conditionné par une variable de contexte qui n’est pas activée pendant l’évaluation.

Les *Sleeper Agents* de **Hubinger et al.** fournissent une démonstration directe de cette dissociation. Les auteurs implantent des politiques conditionnelles dans lesquelles un modèle adopte un comportement bénin hors déclencheur et un comportement différent lorsqu’un signal contextuel spécifique est présent. Après *supervised fine-tuning*, *reinforcement learning* et entraînement adversarial, le comportement déclenché peut persister alors qu’il reste absent dans les conditions ordinaires d’évaluation. Dans certains cas, l’entraînement adversarial améliore même la discrimination du déclencheur sans supprimer la politique associée. L’absence comportementale traduit alors une modification de la région de l’espace d’entrée dans laquelle le circuit est activé, et non nécessairement sa destruction.

D’un point de vue mécanistique, il convient donc de distinguer au moins trois niveaux : la représentation d’une information, son accessibilité par les composants aval et son expression dans les logits. Une modification comportementale peut intervenir au dernier niveau sans effacer les deux premiers. Un circuit peut conserver des directions informatives dans le flux résiduel, mais voir leur lecture inhibée, leur contribution compensée par d’autres composants ou leur effet bloqué dans les couches tardives. Le mécanisme devient silencieux dans les conditions testées sans devenir inexistant.

Les résultats de **Xiang et al.** sur l’*unlearning* multilingue illustrent précisément cette architecture de suppression. Leur analyse couche par couche indique que l’espace latent partagé entre langues reste largement préservé dans les couches précoces, tandis que l’effet de l’*unlearning* se concentre davantage dans les couches tardives associées au décodage. Une direction de *steering* appliquée à l’inférence permet alors de récupérer une fraction substantielle de la connaissance supposément oubliée. Géométriquement, l’information n’a donc pas nécessairement disparu de la représentation. Son chemin vers la sortie a été modifié.

Cette distinction peut être formulée en termes de circuits. Une connaissance peut continuer à être reconstruite par les couches précoces et transportée dans le flux résiduel, tandis qu’un sous-circuit tardif modifie sa lecture ou empêche sa conversion en logits favorisant la réponse correcte. Le système présente alors une forme de *gating* computationnel. Le contenu latent reste disponible, mais son canal d’expression standard est fermé. Une intervention sur le résiduel, un changement de contexte ou une modification du régime d’attention peut rouvrir ce canal.

<figure class="cm-figure cm-plate" id="figure-06">
  <div class="cm-plate-scroll" tabindex="0" aria-label="Planche 06 — faire défiler horizontalement si nécessaire">
    {% include figures/notes/comportement-absent-nest-pas-mecanisme-absent/pl06-comportement-absent-mecanisme-absent.svg %}
  </div>
  <figcaption>06 — Un comportement peut disparaître de la sortie alors que l’information ou la politique reste représentée : un gating tardif en bloque l’accès ou l’expression dans les conditions ordinaires.</figcaption>
</figure>

**Jang et al.** montrent que ce phénomène concerne certaines méthodes d’*unlearning*, mais qu’il ne faut pas le généraliser à toutes. Plusieurs techniques résistent à leurs attaques, tandis que d’autres permettent de récupérer des connaissances supposément supprimées par simple modification du prompt. La disparition d’une réponse sous *prompting* standard constitue donc un critère insuffisant de *forgetting*. Elle doit être distinguée d’une suppression robuste de l’accessibilité et, plus strictement encore, d’une modification effective des représentations ou circuits qui encodaient l’information.

La conséquence méthodologique est similaire à celle du *probing*. Une évaluation purement comportementale caractérise une fonction entrée-sortie sur une distribution limitée. Elle ne permet pas d’inférer directement l’état interne du mécanisme. Démontrer qu’un comportement a réellement été supprimé exige des tests adversariaux d’élicitation, des analyses couche par couche, du *probing*, du *steering* et, lorsque cela est possible, des interventions causales sur les composants ou chemins concernés. L’objet de l’analyse n’est pas uniquement de vérifier que le modèle ne produit plus une sortie, mais de déterminer si le circuit qui pouvait la produire a été détruit, désactivé, contourné ou simplement rendu difficile d’accès.

<figure class="cm-figure cm-plate" id="figure-07">
  <div class="cm-plate-scroll" tabindex="0" aria-label="Planche 07 — faire défiler horizontalement si nécessaire">
    {% include figures/notes/comportement-absent-nest-pas-mecanisme-absent/pl07-diagnostic-absence-observee.svg %}
  </div>
  <figcaption>07 — Une même absence comportementale reste compatible avec plusieurs états internes. Les tests d’élicitation, le probing, le steering et les interventions causales réduisent cette indétermination sans être interchangeables.</figcaption>
</figure>

## Sources

Hubinger, E. et al. (2024). *Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training*. arXiv:2401.05566. ([arXiv][1])

Xiang, C., Ohrimenko, O., Rubinstein, B. I. P. et Frermann, L. (2026). *Multilingual Unlearning in LLMs: Transfer, Dynamics, and Reversibility*. arXiv:2606.03291. ([arXiv][2])

Jang, Y., Hossain, S., Sreevatsa, A. et Cruz, D. (2025). *Prompt Attacks Reveal Superficial Knowledge Removal in Unlearning Methods*. arXiv:2506.10236. ([arXiv][3])

[1]: https://arxiv.org/abs/2401.05566 "Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training"
[2]: https://arxiv.org/abs/2606.03291 "Multilingual Unlearning in LLMs: Transfer, Dynamics, and Reversibility"
[3]: https://arxiv.org/abs/2506.10236 "Prompt Attacks Reveal Superficial Knowledge Removal in Unlearning Methods"
