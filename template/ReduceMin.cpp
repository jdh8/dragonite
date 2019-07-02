#include "dragonite.hpp"

{% from "mod/batch.cpp" import batch -%}
{% from "mod/reduce.cpp" import inner, outer -%}

{% macro verify(message) -%}
    ASSERT_FALSE(std::memcmp(buffer, y, sizeof(y)));
{% endmacro -%}

{% call(name, shape) batch() -%}
    {% call(x, axes) outer("ReduceMin", name, shape) -%}
        {{ inner(numpy.fmin.reduce(x, axis=axes, keepdims=true), axes, verify) }}
    {% endcall -%}
{% endcall -%}
{# vim: set ft=liquid: #}
