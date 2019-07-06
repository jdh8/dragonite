#include "common.hpp"

{% from "mod/batch.cpp" import batch -%}

{% call(name, shape) batch(2) -%}
SKYPAT_F(GlobalAveragePool, {{ name }})
{
    {% set ndim = shape | length -%}
    {% set axes = range(2, ndim) | tuple -%}
    {% set x = numpy.random.standard_cauchy(shape).astype(numpy.float32) -%}
    {% set y = numpy.mean(x, axes, keepdims=true) -%}

    const float x[] = {{ x.flatten() | array }};
    const float y[] = {{ y.flatten() | array }};

    const std::int32_t shape[] = {{ shape | array }};
    const std::int32_t reduced[] = {{ y.shape | array }};
    const std::size_t size = {{ y.size }};
    const std::size_t ndim = {{ ndim }};

    float buffer[size];

    ONNC_RUNTIME_globalaveragepool_float(nullptr, x, ndim, shape, buffer, ndim, reduced);

    for (std::size_t i = 0; i < size; ++i) {
        using std::abs;
        ASSERT_LE_MSG(abs(buffer[i] - y[i]), abs(1e-5 * y[i]), i);
    }
}
{% endcall -%}
{# vim: set ft=liquid: #}
