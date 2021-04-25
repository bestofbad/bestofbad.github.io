---
layout: single
title: "Helpful with diverse matters"
permalink: /bSet/
post_categories : bSet
author_profile: true
sidebar:
  nav: "Blogroll"
---

Setting front page


{% for category in site.categories %}
  {% if category[0] == page.post_categories %}
  <ul>
    {% for post in category[1] %}
      <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
  {% endif %}
{% endfor %}
