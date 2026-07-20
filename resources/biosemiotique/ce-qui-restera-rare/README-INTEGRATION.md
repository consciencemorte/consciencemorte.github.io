# Planches gravées — assurance des modèles (révision 2)

Six planches SVG pour la charte de **Conscience Morte**. Elles n'utilisent que les
variables déjà définies dans `assets/main.scss` :

`--paper`, `--paper-deep`, `--ink`, `--graphite`, `--line`, `--oxide`, `--moss`, `--blue`
et les trois familles `--font-editorial`, `--font-interface`, `--font-code`.

Aucune couleur en dur : elles suivent le sélecteur `[data-theme="dark"]` sans retouche.

---

## Ce qui a changé dans cette révision

Le principe précédent tenait — la trame porte l'argument, le texte est réservé dans
la trame, le cadre est celui de la taille-douce. Ce qui manquait, c'était que **la
composition elle-même dise quelque chose**. Cinq planches sur six étaient des
diagrammes de boîtes et de flèches sur lesquels une trame avait été posée après
coup ; la trame décorait un argument qu'elle ne portait pas.

| Planche | Avant | Maintenant |
|---|---|---|
| **01** | L'écart n'existait que comme accolade dans la marge droite | L'écart **est** la figure : un coin hachuré qui s'ouvre, peuplé d'un semis d'anneaux dont la densité croît. Le lecteur voit l'espace avant de lire le chiffre. |
| **02** | Organigramme de cercles ; aucune grandeur, aucune date | Deux registres sur un même axe de temps. En haut la **vague** mesurée (0,5 % en août 2024, 3,3 % en mars 2026) ; en bas le **sillage** projeté (100–300 modèles fin 2028). Une verticale sépare le mesuré du projeté. |
| **03** | Cinq cartes en haut, une stratigraphie décorative en bas, sans lien entre elles | Sept maillons et sept colonnes ; **la coupe est alignée sur les mêmes colonnes**. Les strates se pincent : celle des garde-fous s'effondre exactement à la colonne de la distillation. |
| **04** | Matrice correcte | Ajout de ce que chaque couche **observe** (c'est ce qui les distingue, pas la question) et de la maturité de la pratique — la claim centrale de la section. |
| **05** | Le pipeline de l'expérience | Le pipeline **et son témoin**. Deux étudiants, deux initialisations : le ruban oxyde entre dans l'un et s'arrête net devant l'autre. La borne du résultat est dessinée, plus seulement écrite en petites capitales. |
| **06** | Quatre lignes identiques dans un tableau | Une **échelle d'accès** : trois piles de hauteur croissante. Le seuil tombe entre la boîte noire et l'artefact, pas entre les méthodes. La forme dit ce que le pied de planche disait tout seul. |

Trois corrections de fond, aussi :

- **PL. 01** — le « × 1500 » est le rapport de deux indices ramenés à la même base.
  Il ne figure pas dans l'article et n'a pas de source. Il est conservé, mais démoté :
  les deux **taux** (× 2,4 / an ; ÷ 280 en 23 mois) sont désormais les annotations
  principales, et la mise en garde a quitté le pied de planche pour rejoindre le
  chiffre. Une planche qui a besoin de trois lignes de disclaimer pour ne pas mentir
  n'a pas gagné sa place.
- **PL. 02** — deux valeurs sont relevées, la vague qui les relie ne l'est pas. C'est
  écrit sur la planche, pas dans une note.
- **PL. 03** — l'ancienne légende annonçait « angle de trame = origine présumée »
  alors que chaque strate portait une trame uniforme : l'angle n'encodait rien. Il
  encode maintenant réellement le maillon d'origine, et la clé le déclare.

### Lisibilité

- Plancher typographique relevé : plus rien sous 9,5 px, et 9,5 px réservé aux
  libellés internes ; le corps courant des planches est à 10,5–11,5 px. À 980 px de
  large (largeur réelle de `.prose figure`, marges négatives comprises) le viewBox de
  1000 unités rend donc à ~1:1.
- Le halo `.h` passe de 4 px à 3 px : sur du 10 px, 2 px de contour de chaque côté
  mangeaient les contrepoinçons.
- Le vide est annoté. Sur les planches 03 et 06, la zone blanche est un argument
  (« l'artefact n'existe pas encore », « rien à examiner ») : elle est légendée comme
  telle plutôt que laissée pour un défaut de composition.

### Robustesse

- **`var()` sorti des attributs de présentation.** Les trames faisaient
  `<line stroke="var(--ink,…)">`. C'est théoriquement correct — un attribut de
  présentation est une déclaration CSS — mais le support a été longtemps inégal, et
  un repli silencieux en noir dans une planche entière est un mode d'échec coûteux.
  Toutes les couleurs de trame passent désormais par des classes du bloc `<style>`,
  déjà scopées sous `.cm-01` … `.cm-06`.
- **Trames mortes supprimées.** Chaque fichier définissait quatorze `<pattern>` et en
  utilisait trois ou quatre. Chaque planche ne déclare plus que ce qu'elle emploie.
- **Polices par variable.** Les blocs `<style>` codaient `"Newsreader"`, `"Inter"`,
  `"IBM Plex Mono"` en dur au lieu de `--font-editorial` &c. Corrigé, en propriétés
  séparées plutôt qu'en raccourci `font:` (le raccourci avec `var()` marche, mais
  autant ne pas en dépendre).
- IDs préfixés et classes scopées : conservés tels quels, c'était déjà juste.

---

## Intégration

Copier les six SVG canoniques dans
`_includes/figures/biosemiotique/ce-qui-restera-rare/`, puis les inclure **en
ligne** (c'est ce qui leur fait hériter des variables CSS et suivre le thème
clair/sombre). Le dossier source est exclu du build afin de ne pas publier les
références, archives et aperçus de travail :

```html
<figure class="cm-figure cm-plate">
  <div class="cm-plate-scroll">
    {% include figures/biosemiotique/ce-qui-restera-rare/03-chaine-cognitive.svg %}
  </div>
  <figcaption>03 — Sept transformations, un seul artefact : la filière déclarée et la coupe du dépôt effectif, aux mêmes colonnes. Planche originale.</figcaption>
</figure>
```

Le SCSS ajoute déjà `FIG. ` devant chaque légende : la légende commence donc par
son numéro (`03 — …`), jamais par `Fig. 03`.

## SCSS complémentaire

Inchangé par rapport à la version précédente :

```scss
.cm-plate svg {
  display: block;
  width: 100%;
  height: auto;
  overflow: visible;
}

/* Sur mobile, .prose figure perd ses marges négatives : la planche tomberait à
   ~345px de large. On garde ses proportions et on laisse le lecteur la balayer,
   comme une planche dépliante. */
@media (max-width: 740px) {
  .cm-plate-scroll {
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
    scrollbar-width: thin;
    scrollbar-color: var(--line) transparent;
  }
  .cm-plate-scroll svg { min-width: 720px; }
  .cm-plate figcaption::after {
    content: " — faire glisser pour lire la planche entière";
    color: var(--oxide);
  }
}

@media print {
  .cm-plate-scroll { overflow: visible; }
  .cm-plate-scroll svg { min-width: 0; }
}
```

Pas besoin de `vector-effect` en SCSS : il est déjà porté par les classes de
trait à l'intérieur de chaque planche.

---

## Vocabulaire graphique

| Trame | Sens |
|---|---|
| Contre-hachure serrée | facteur rare, concentré (la frontière) |
| Hachure | dépôt, écart, corpus — la nature est portée par la couleur |
| Pointillé | incertitude, projection, origine indéterminée, pratique non instituée |
| Angle de la hachure | maillon d'origine d'une strate (PL. 03, réellement encodé) |
| Superposition de deux trames | héritage combiné (PL. 05, l'étudiant de gauche) |
| Semis d'anneaux | population de modèles (PL. 01, PL. 02) — plein : observé ; pointillé : projeté |
| Hauteur d'une pile | degré d'accès (PL. 06) |

---

## Ordre et emplacement

Inchangé, sauf **02**, dont la place naturelle est maintenant juste avant la phrase
qu'elle illustre plutôt qu'après :

1. `01-economie-frontiere.svg` — après le deuxième paragraphe de l'introduction.
2. `02-concentration-sillage.svg` — après « La frontière se concentre ; son sillage prolifère. »
3. `03-chaine-cognitive.svg` — après l'énumération des sept maillons.
4. `04-regime-assurance.svg` — après le premier paragraphe de la section sur le régime d'assurance.
   *(`04-regime-assurance-v2.svg` reste un fichier de travail non publié ; seule la version sans suffixe est canonique.)*
5. `05-apprentissage-subliminal.svg` — après la description de l'expérience canonique.
6. `06-audit-aveugle.svg` — après la présentation de l'exercice d'audit à l'aveugle.

## Sources à créditer dans les légendes

- **01** : graphique de l'auteur d'après Cottier et al., *The Rising Costs of Training
  Frontier AI Models*, et Stanford HAI, *AI Index Report 2025* (chap. 1). Le faisceau
  reprend l'intervalle × 2,0 – × 2,9 par an ; il s'ouvre avec le temps, comme il se
  doit d'un taux composé. La série basse reprend les points de la courbe
  « GPT-3.5 level+ on MMLU » de l'AI Index. Le semis d'anneaux est schématique : il
  figure une population, il ne rejoue pas modèle par modèle.
- **02** : d'après Stanford HAI, *AI Index Report 2026* (chap. 2) pour les deux valeurs
  d'écart, et Kumar et Manning, *Trends in Frontier AI Model Count*, arXiv:2504.16138,
  pour la fourchette 100–300. Composition originale.
- **05** : adapté de Cloud et al., *Subliminal Learning*, arXiv:2507.14805 — y compris
  le témoin inter-modèles, qui est l'apport de la révision.
- **06** : adapté de Marks, Treutlein et al., *Auditing Language Models for Hidden
  Objectives*, arXiv:2503.10965.
- **03**, **04** : planches originales.

Les planches 05 et 06 restent des redessins complets dans ta charte : composition,
vocabulaire graphique et légendes sont à toi. Elles restent des adaptations sur le
plan intellectuel — le crédit ci-dessus est nécessaire.

---

## Contrôle avant publication

- `apercu.html` ouvre les six planches à la suite, dans la largeur réelle de
  `.prose`, avec bascule clair/sombre et bascule 980 / 720 px.
- Vérifier la bascule sombre : les trames sont en `--ink` / `--oxide` / `--blue` /
  `--moss` à opacité fixe, elles s'inversent avec le thème ; le halo `.h` est en
  `--paper`, il s'inverse aussi.
- Vérifier sous 740 px que le défilement horizontal s'arme et que la mention
  « faire glisser » apparaît.
