#include "common.hpp"

{% from "mod/batch.cpp" import batch -%}
{% from "mod/argmax.cpp" import testcase -%}

{% call(name, shape) batch() -%}
    {{ testcase("ArgMin", name, shape) -}}
{% endcall -%}
{# vim: set ft=liquid: #}
