#include "dragonite/common.hpp"

{% from "mod/batch.cpp" import batch -%}

{% call(name, shape) batch() -%}
SKYPAT_F(LogSoftmax, {{ name }})
{
    {% set x = numpy.random.standard_cauchy(shape).astype(numpy.float32) + 96 -%}
    {% set ndim = shape | length -%}

    const float x[] = {{ x.flatten() | array }};

    const std::int32_t shape[] = {{ shape | array }};
    const std::int32_t ndim = {{ ndim }};
    const std::size_t size = {{ x.size }};

    float buffer[size];

    {% for axis in range(ndim) -%}
    {
        {% set axes = range(axis, ndim) | tuple -%}
        {% set y = x - numpy.logaddexp.reduce(x, axis=axes, keepdims=true) -%}

        const float y[] = {{ y.flatten() | array }};

        ONNC_RUNTIME_logsoftmax_float(nullptr, x, ndim, shape, buffer, ndim, shape, {{ axis }});
        ASSERT_LE(dragonite::norm(y, buffer, size), 1e-5 * dragonite::norm(y, size));
    }
    {% else -%}
    {
        ONNC_RUNTIME_logsoftmax_float(nullptr, x, ndim, shape, buffer, ndim, shape, 0);
        ASSERT_EQ(*buffer, 0);
    }
    {% endfor %}
}
{% endcall -%}
{# vim: set ft=liquid: #}
