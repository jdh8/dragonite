#include "dragonite.hpp"

{% from "mod/batch.cpp" import batch -%}

{% call(name, shape) batch() -%}
SKYPAT_F(Hardmax, {{ name }})
{
    {% set x = numpy.random.standard_cauchy(shape).astype(numpy.float32) + 96 -%}
    {% set ndim = shape | length -%}

    const float x[] = {{ x.flatten() | array }};

    const std::int32_t shape[] = {{ shape | array }};
    const std::int32_t ndim = {{ ndim }};

    float buffer[{{ x.size }}];

    {% for axis in range(ndim) -%}
    {
        {% set axes = range(axis, ndim) | tuple -%}
        {% set y = (x == numpy.fmax.reduce(x, axis=axes, keepdims=true)).astype(numpy.int8) -%}

        const float y[] = {{ y.flatten() | array }};
        ONNC_RUNTIME_hardmax_float(nullptr, x, ndim, shape, buffer, ndim, shape, {{ axis }});
        ASSERT_FALSE(std::memcmp(buffer, y, sizeof(y)));
    }
    {% else -%}
    {
        ONNC_RUNTIME_hardmax_float(nullptr, x, ndim, shape, buffer, ndim, shape, 0);
        ASSERT_EQ(*buffer, 1);
    }
    {% endfor %}
}
{% endcall -%}
{# vim: set ft=liquid: #}
