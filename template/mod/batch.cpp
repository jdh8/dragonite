{% macro batch(ndim=0) -%}
{{ caller("vector", numpy.random.randint(2, 6, ndim + 1)) }}
{{ caller("matrix", numpy.random.randint(2, 6, ndim + 2)) }}
{{ caller("tensor", numpy.random.randint(2, 6, numpy.random.randint(ndim + 3, 6))) }}
{{ caller("scalar", numpy.random.randint(2, 6, ndim)) -}}
{% endmacro -%}
{# vim: set ft=liquid: #}
