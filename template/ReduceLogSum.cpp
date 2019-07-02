#include "dragonite/common.hpp"

{% from "mod/batch.cpp" import batch -%}
{% from "mod/reduce.cpp" import inner, outer -%}

{% call(name, shape) batch() -%}
    {% call(x, axes) outer("ReduceLogSum", name, shape, numpy.random.exponential) -%}
        {{ inner(numpy.log(numpy.sum(x, axis=axes, keepdims=true)), axes) }}
    {% endcall -%}
{% endcall -%}
{# vim: set ft=liquid: #}
