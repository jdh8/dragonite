#include "dragonite/common.hpp"

{% from "mod/batch.cpp" import batch -%}
{% from "mod/binary.cpp" import inner, outer -%}

{% call(name, shape) batch() -%}
    {% call(x, y) outer("Greater", name, shape) -%}
        {% set a = numpy.random.standard_cauchy(x).astype(numpy.float32) -%}
        {% set b = numpy.random.standard_cauchy(y).astype(numpy.float32) -%}
        {{ inner((a > b).astype(numpy.int8), a, b) }}
    {% endcall -%}
{% endcall -%}
{# vim: set ft=liquid: #}
