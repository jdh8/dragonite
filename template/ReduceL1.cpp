#include "dragonite.hpp"

{% from "mod/batch.cpp" import batch -%}
{% from "mod/reduce.cpp" import inner, outer -%}

{% call(name, shape) batch() -%}
    {% call(x, axes) outer("ReduceL1", name, shape) -%}
        {{ inner(numpy.sum(numpy.abs(x), axis=axes, keepdims=true), axes) }}
    {% endcall -%}
{% endcall -%}
{# vim: set ft=liquid: #}

