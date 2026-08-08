---
title: "Un comportement absent n’est pas un mécanisme absent"
date: 2026-08-07
permalink: /notes/comportement-absent-nest-pas-mecanisme-absent/
reading_minutes: 5
topics: [interprétabilité mécanistique, unlearning, évaluation comportementale]
status: revue de littérature
description: "L’absence d’un comportement observable ne suffit pas à établir que le mécanisme qui le produisait a disparu."
hero_image: "/assets/img/notes/comportement-absent-nest-pas-mecanisme-absent/hero.jpg"
hero_position: "center 18%"
hero_credit: "Tijdschrift voor entomologie — image courtesy of BHL"
hero_source: "https://biodiversitylibrary.org/page/10847895"
---

L’absence d’un comportement observable ne constitue pas une preuve de disparition du mécanisme qui le produisait. Elle établit seulement que, sous une distribution donnée d’entrées et dans un régime d’inférence donné, ce mécanisme ne détermine plus la sortie observée.

Plusieurs résultats récents rendent cette distinction empirique. **Jang et al.** montrent que, pour ELM, préfixer le prompt par un texte de remplissage en hindi restaure **57,3 %** d’exactitude sur les connaissances supposément désapprises. **Xiang et al.** récupèrent quant à eux environ **50 %** de ces connaissances sur Qwen et **90 %** sur Gemma par une intervention de *steering* appliquée aux activations pendant l’inférence. Dans les deux cas, une absence mesurée sous l’évaluation standard ne constitue donc pas une propriété stable du système sous intervention.

L’interprétation de ces résultats exige de distinguer trois niveaux : **représentation**, **accessibilité** et **expression**. Une information peut ne plus être représentée par les états internes pertinents ; elle peut rester représentée mais ne pas être activée ou accessible dans le contexte considéré ; elle peut enfin être accessible au calcul tout en ayant une influence réduite ou compensée sur les logits. Une même absence en sortie est compatible avec ces trois situations.

Les *Sleeper Agents* de **Hubinger et al.** relèvent principalement du deuxième niveau. Les auteurs entraînent des politiques conditionnelles dont l’expression dépend d’un signal contextuel spécifique. Hors déclencheur, le comportement ciblé est absent ; lorsque le déclencheur est présent, il réapparaît. Après *supervised fine-tuning*, *reinforcement learning* ou entraînement adversarial, cette dépendance peut persister, et certaines procédures adversariales améliorent même la discrimination des contextes dans lesquels la politique doit être exprimée.

Le mécanisme pertinent n’est donc pas nécessairement supprimé ni bloqué dans les couches tardives. Son accessibilité dépend de l’entrée : le contexte ordinaire ne déclenche pas le calcul qui conduit au comportement observé sous la condition spéciale. L’absence comportementale résulte ici d’une différence d’activation.

Les résultats de **Xiang et al.** localisent la dissociation autrement. Leur analyse de l’*unlearning* multilingue indique que les structures partagées entre langues restent relativement préservées dans les couches précoces, alors que les modifications associées à l’oubli sont plus importantes dans les couches tardives. La représentation utile n’apparaît donc pas uniformément effacée ; c’est davantage son exploitation vers la sortie qui est modifiée.

Cette lecture est renforcée par leur expérience de *steering*. Les auteurs estiment une direction associée aux transformations produites par l’*unlearning*, à partir d’un jeu auxiliaire distinct des faits utilisés pour mesurer la récupération, puis interviennent sur les activations pendant l’inférence. L’intervention permet de récupérer environ **50 %** des connaissances sur Qwen et **90 %** sur Gemma. Une destruction complète des structures nécessaires à la réponse expliquerait difficilement une telle réversibilité sous une intervention de faible dimension.

Le résultat ne démontre cependant pas que la représentation initiale soit restée intégralement intacte. Le *steering* modifie causalement la dynamique du modèle ; il peut restaurer l’accès à une représentation résiduelle, amplifier un signal affaibli ou faciliter une reconstruction qui n’aurait pas lieu spontanément. La conclusion établie est donc une propriété de récupérabilité sous intervention, plus forte qu’une simple corrélation de représentation mais plus faible qu’une démonstration d’identité du mécanisme avant et après *unlearning*.

La même distinction s’applique au *probing*. La décodabilité d’une variable à partir du flux résiduel établit qu’une information exploitable est présente dans les activations examinées ; elle ne suffit pas à montrer que cette information participe causalement à la sortie. *Probing*, *steering* et observation comportementale renseignent ainsi des propriétés différentes du même enchaînement représentation–accessibilité–expression.

<figure class="cm-figure cm-plate" id="figure-06">
  <div class="cm-plate-scroll" tabindex="0" aria-label="Planche 06 — faire défiler horizontalement si nécessaire">
    {% include figures/notes/comportement-absent-nest-pas-mecanisme-absent/pl06-localisation-absence.svg %}
  </div>
  <figcaption>06 — L’absence en sortie peut intervenir à trois niveaux : altération de la représentation, défaut d’accès ou d’activation, ou suppression de l’expression vers les logits. Une observation comportementale seule ne localise pas la modification.</figcaption>
</figure>

Les expériences de **Jang et al.** montrent en parallèle que cette dissociation ne doit pas être érigée en description générale de l’*unlearning*. Certaines méthodes sont fortement vulnérables aux changements de contexte : dans le cas d’ELM, le remplissage en hindi suffit à restaurer une exactitude de **57,3 %**. **RMU** et **TAR** résistent en revanche davantage aux attaques étudiées.

Leur analyse des logits fournit une restriction supplémentaire. L’exactitude calculée directement à partir des logits est fortement corrélée à celle des sorties générées. Les modèles étudiés ne semblent donc pas, en général, conserver systématiquement la réponse correcte au niveau de la distribution de sortie tout en la dissimulant par un refus, un changement de format ou une autre transformation superficielle de la génération.

Les résultats disponibles excluent ainsi deux inférences symétriques. Une disparition comportementale ne permet pas de conclure à la destruction des représentations sous-jacentes ; une récupération par prompt ou intervention ne permet pas non plus de conclure que le mécanisme initial est resté intact. Elle établit seulement qu’une capacité suffisante à produire la réponse subsiste sous les conditions de récupération considérées.

<figure class="cm-figure cm-plate" id="figure-07">
  <div class="cm-plate-scroll" tabindex="0" aria-label="Planche 07 — faire défiler horizontalement si nécessaire">
    {% include figures/notes/comportement-absent-nest-pas-mecanisme-absent/pl07-portee-des-methodes.svg %}
  </div>
  <figcaption>07 — Les méthodes d’évaluation portent sur différents niveaux du même schéma. Le probing teste la décodabilité des représentations ; les modifications de contexte et le fine-tuning testent leur accessibilité ; l’analyse des logits renseigne leur expression ; les interventions causales testent les dépendances entre ces niveaux.</figcaption>
</figure>

Cette indétermination impose également une limite à la notion de suppression. Aucun protocole fini ne peut établir qu’une connaissance est inaccessible sous l’ensemble des prompts, contextes, modifications de paramètres et interventions internes possibles. Une affirmation d’*unlearning* doit donc préciser la classe d’accès contre laquelle la suppression a effectivement été testée.

**Patil, Hase et Bansal** formalisent cette contrainte sous la forme d’un modèle de menace. Leur critère considère une attaque comme réussie lorsque la réponse sensible appartient à un ensemble de (B) candidats produits par l’attaquant, (B) constituant son budget. Après édition de GPT-J par ROME, leur attaque *white-box* par projection d’états cachés récupère encore la réponse ciblée dans **38 %** des cas avec un budget de **(B=20)**.

Le nombre illustre directement pourquoi une mesure de récupération n’est interprétable qu’avec son régime d’accès. Un protocole qui échoue à extraire une connaissance avec un seul prompt en accès *black-box* et un protocole qui échoue après vingt candidats construits à partir des états cachés ne soutiennent pas la même affirmation. Le modèle de menace doit donc spécifier le budget de requêtes ou de candidats, l’accès éventuel aux poids, aux logits et aux activations, la possibilité de modifier les entrées ou les paramètres, ainsi que l’information préalable dont dispose l’attaquant.

<figure class="cm-figure cm-plate" id="figure-08">
  <div class="cm-plate-scroll" tabindex="0" aria-label="Planche 08 — faire défiler horizontalement si nécessaire">
    {% include figures/notes/comportement-absent-nest-pas-mecanisme-absent/pl08-modele-de-menace.svg %}
  </div>
  <figcaption>08 — Un taux de récupération ou de résistance n’est interprétable qu’avec son régime d’accès et son budget. Une non-récupération ne soutient une affirmation de suppression que dans le modèle de menace effectivement testé.</figcaption>
</figure>

Cette formulation est également cohérente avec **Lynch et al.**, qui évaluent l’*unlearning* au moyen de plusieurs familles de tests. Une méthode peut satisfaire une métrique comportementale tout en laissant une connaissance récupérable par une autre procédure ou en conservant des représentations internes proches de celles du modèle initial. L’évaluation robuste consiste alors à déterminer quelles formes d’accès ont effectivement été neutralisées, plutôt qu’à inférer une suppression interne à partir d’un seul protocole.

Le modèle de menace ne résout toutefois pas le problème mécanistique. Montrer qu’aucune attaque d’une classe donnée ne récupère une connaissance établit une résistance sous cette classe d’interventions ; cela ne distingue pas une représentation détruite d’une représentation encore présente mais inaccessible aux procédures autorisées. Les deux questions doivent donc rester séparées : le modèle de menace borne ce que l’évaluation comportementale permet d’affirmer, tandis que l’analyse mécanistique cherche à localiser la modification entre représentation, accessibilité et expression.

Une connaissance peut ainsi être **non récupérable sous un modèle de menace donné** sans que son absence mécanistique soit établie. C’est précisément la différence entre constater qu’un comportement a disparu et montrer pourquoi il ne peut plus être produit.

## Sources

Hubinger, E. et al. (2024). *Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training*. arXiv:2401.05566. ([arXiv][1])

Xiang, C., Ohrimenko, O., Rubinstein, B. I. P. et Frermann, L. (2026). *Multilingual Unlearning in LLMs: Transfer, Dynamics, and Reversibility*. arXiv:2606.03291. ([arXiv][2])

Jang, Y., Hossain, S., Sreevatsa, A. et Cruz, D. (2025). *Prompt Attacks Reveal Superficial Knowledge Removal in Unlearning Methods*. arXiv:2506.10236. ([arXiv][3])

Lynch, A., Guo, P., Ewart, A., Casper, S. et Hadfield-Menell, D. (2024). *Eight Methods to Evaluate Robust Unlearning in LLMs*. arXiv:2402.16835. ([arXiv][4])

Patil, V., Hase, P. et Bansal, M. (2023). *Can Sensitive Information Be Deleted From LLMs? Objectives for Defending Against Extraction Attacks*. arXiv:2309.17410. ([arXiv][5])

[1]: https://arxiv.org/abs/2401.05566 "Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training"
[2]: https://arxiv.org/abs/2606.03291 "Multilingual Unlearning in LLMs: Transfer, Dynamics, and Reversibility"
[3]: https://arxiv.org/abs/2506.10236 "Prompt Attacks Reveal Superficial Knowledge Removal in Unlearning Methods"
[4]: https://arxiv.org/abs/2402.16835 "Eight Methods to Evaluate Robust Unlearning in LLMs"
[5]: https://arxiv.org/abs/2309.17410 "Can Sensitive Information Be Deleted From LLMs? Objectives for Defending Against Extraction Attacks"
