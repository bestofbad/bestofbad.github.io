---
layout: single
title: "Helpful with diverse matters"
permalink: /bDL/
post_categories : bDL
author_profile: true
sidebar:
  nav: "Blogroll"
---

front page for Blogs of DL


{% for category in site.categories %}
  {% if category[0] == page.post_categories %}
  <ul>
    {% for post in category[1] %}
      <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
  {% endif %}
{% endfor %}
