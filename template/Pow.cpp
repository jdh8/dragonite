#include "common.hpp"

{% from "mod/batch.cpp" import batch -%}
{% from "mod/binary.cpp" import inner, outer -%}

{% call(name, shape) batch() -%}
    {% call(x, y) outer("Pow", name, shape) -%}
        {% set r = numpy.random.rayleigh(size=x).astype(numpy.float32) -%}
        {% set a = numpy.random.randn(*y).astype(numpy.float32) -%}
        {{ inner(r**a, r, a) }}

        {% set b = numpy.random.randint(0, 16, y, numpy.int8) -%}
        {{ inner((-r)**b, -r, b) }}
    {% endcall -%}
{% endcall -%}
{# vim: set ft=liquid: #}
