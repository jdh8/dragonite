{% macro channel() -%}
{% set channels = numpy.random.randint(2, 6, 2) -%}
{{ caller("vector", channels, numpy.random.randint(6, 24, 1), numpy.random.randint(2, 6, 1)) }}
{{ caller("matrix", channels, numpy.random.randint(6, 12, 2), numpy.random.randint(2, 6, 2)) }}
{{ caller("tensor", channels, numpy.random.randint(6, 10, 3), numpy.random.randint(2, 6, 3)) }}
{% endmacro -%}
{# vim: set ft=liquid: #}
