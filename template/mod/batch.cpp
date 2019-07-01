{% macro batch() -%}
{{ caller("vector", numpy.random.randint(2, 6, 1)) }}
{{ caller("matrix", numpy.random.randint(2, 6, 2)) }}
{{ caller("tensor", numpy.random.randint(2, 6, numpy.random.randint(3, 6))) }}
{{ caller("scalar", ()) -}}
{% endmacro -%}
{# vim: set ft=liquid: #}
