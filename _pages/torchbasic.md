---
layout: single
title: "Basic Practice in Pytorch"
permalink: /torchbasic/
post_categories: torchbasic
author_profile: true
sidebar:
  nav: "DL"

---


{% for category in site.categories %}
  {% if category[0] == page.post_categories %}
  <ul>
    {% for post in category[1] %}
      <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
  {% endif %}
{% endfor %}
