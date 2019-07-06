#include "common.hpp"

{% from "mod/batch.cpp" import batch -%}

{% call(name, shape) batch(2) -%}
SKYPAT_F(GlobalMaxPool, {{ name }})
{
    {% set ndim = shape | length -%}
    {% set axes = range(2, ndim) | tuple -%}
    {% set x = numpy.random.standard_cauchy(shape).astype(numpy.float32) -%}
    {% set y = numpy.fmax.reduce(x, axes, keepdims=true) -%}

    const float x[] = {{ x.flatten() | array }};
    const float y[] = {{ y.flatten() | array }};

    const std::int32_t shape[] = {{ shape | array }};
    const std::int32_t reduced[] = {{ y.shape | array }};
    const std::size_t ndim = {{ ndim }};

    float buffer[{{ y.size }}];

    ONNC_RUNTIME_globalmaxpool_float(nullptr, x, ndim, shape, buffer, ndim, reduced);
    ASSERT_FALSE(std::memcmp(buffer, y, sizeof(y)));
}
{% endcall -%}
{# vim: set ft=liquid: #}
