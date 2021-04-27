---
layout: single
title: "Helpful with diverse matters"
permalink: /bPy/
post_categories : bPy
author_profile: true
sidebar:
  nav: "Blogroll"
---

front page for Blogs about Python


{% for category in site.categories %}
  {% if category[0] == page.post_categories %}
  <ul>
    {% for post in category[1] %}
      <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
  {% endif %}
{% endfor %}
