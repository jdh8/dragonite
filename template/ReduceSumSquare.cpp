#include "dragonite/common.hpp"

{% from "mod/batch.cpp" import batch -%}
{% from "mod/reduce.cpp" import inner, outer -%}

{% call(name, shape) batch() -%}
    {% call(x, axes) outer("ReduceSumSquare", name, shape) -%}
        {{ inner(numpy.sum(numpy.square(x), axis=axes, keepdims=true), axes) }}
    {% endcall -%}
{% endcall -%}
{# vim: set ft=liquid: #}
