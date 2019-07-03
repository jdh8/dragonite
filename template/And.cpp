#include "common.hpp"

{% from "mod/batch.cpp" import batch -%}
{% from "mod/binary.cpp" import inner, outer -%}

{% call(name, shape) batch() -%}
    {% call(x, y) outer("And", name, shape) -%}
        {% set a = numpy.random.randint(0, 2, x, numpy.int8) -%}
        {% set b = numpy.random.randint(0, 2, y, numpy.int8) -%}
        {{ inner(numpy.bitwise_and(a, b), a, b) }}
    {% endcall -%}
{% endcall -%}
{# vim: set ft=liquid: #}
