#include "dragonite.hpp"

{% from "mod/batch.cpp" import batch -%}

{% macro subtest(x, axis, p) %}
{
    {% set table = numpy.fmax, numpy.add, numpy.hypot -%}
    {% set y = x / table[p].reduce(numpy.abs(x), axis, keepdims=true) -%}

    const float y[] = {{ y.flatten() | array }};

    ONNC_RUNTIME_lpnormalization_float(nullptr, x, ndim, shape, buffer, ndim, shape, {{ axis }}, {{ p }});
    dragonite::verify(buffer, y, size, "axis={{ axis }}, p={{ p }}");
}
{% endmacro -%}

{% call(name, shape) batch() -%}
SKYPAT_F(LpNormalization, {{ name }})
{
    {% set x = numpy.random.standard_cauchy(shape).astype(numpy.float32) -%}
    {% set ndim = shape | length -%}

    const float x[] = {{ x.flatten() | array }};
    const std::int32_t shape[] = {{ shape | array }};
    const std::size_t ndim = {{ shape | length }};
    const std::size_t size = {{ x.size }};

    float buffer[size];

    {% for axis in range(-1, ndim) -%}
        {% for p in (1, 2) -%}
            {{ subtest(x, axis, p) | indent -}}
        {% endfor -%}
    {% endfor %}
}
{% endcall -%}
{# vim: set ft=liquid: #}
