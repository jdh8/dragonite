#include "common.hpp"

{% from "mod/batch.cpp" import batch -%}
{% from "mod/reduce.cpp" import inner, outer -%}

{% call(name, shape) batch() -%}
    {% call(x, axes) outer("ReduceL2", name, shape) -%}
        {{ inner(numpy.hypot.reduce(numpy.abs(x), axis=axes, keepdims=true), axes) }}
    {% endcall -%}
{% endcall -%}
{# vim: set ft=liquid: #}
