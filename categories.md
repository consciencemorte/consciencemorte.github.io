---
layout: default
title: "Champs d'étude"
permalink: /categories/
---

<section class="cm-shell cm-page">
  <header class="cm-page-header">
    <h1 class="cm-hero-title">Champs d'étude</h1>
    <p class="cm-hero-subtitle">Tous les fragments par taxonomie</p>
  </header>

  {% if site.categories.size > 0 %}
    <div class="cm-category-grid">
      {% for cat in site.categories %}
        {% assign cat_name = cat[0] %}
        {% assign posts_in_cat = cat[1] %}
        <article class="cm-card" id="{{ cat_name | slugify }}">
          <h2 class="cm-card-title">{{ cat_name }}</h2>
          <p class="cm-cat-count">{{ posts_in_cat | size }} fragment{% if posts_in_cat.size > 1 %}s{% endif %}</p>
          <ul class="cm-list">
            {% for post in posts_in_cat %}
              <li>
                <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
                <span class="cm-post-meta">{{ post.date | date: "%-d %b %Y" }}</span>
              </li>
            {% endfor %}
          </ul>
        </article>
      {% endfor %}
    </div>
  {% else %}
    <p>Aucune catégorie n'est encore cristallisée.</p>
  {% endif %}
</section>
