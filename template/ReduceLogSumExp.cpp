#include "common.hpp"

{% from "mod/batch.cpp" import batch -%}
{% from "mod/reduce.cpp" import inner, outer -%}

{% call(name, shape) batch() -%}
    {% call(x, axes) outer("ReduceLogSumExp", name, shape) -%}
        {{ inner(numpy.logaddexp.reduce(x, axis=axes, keepdims=true), axes) }}
    {% endcall -%}
{% endcall -%}
{# vim: set ft=liquid: #}
