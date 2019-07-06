#include "common.hpp"

#define VERIFY(buffer, y, size) do {                               \
    for (std::size_t i = 0; i < size; ++i) {                       \
        using std::abs;                                            \
        ASSERT_LE_MSG(abs(buffer[i] - y[i]), abs(1e-5 * y[i]), i); \
    }                                                              \
} while (0)

{% from "mod/batch.cpp" import batch -%}

{% call(name, shape) batch(2) -%}
SKYPAT_F(GlobalLpPool, {{ name }})
{
    {% set x = numpy.random.standard_cauchy(shape).astype(numpy.float32) -%}
    {% set ndim = shape | length -%}
    {% set axes = range(2, ndim) | tuple -%}

    {% set y1 = numpy.sum(numpy.abs(x), axes, keepdims=true) -%}
    {% set y2 = numpy.hypot.reduce(x, axes, keepdims=true) -%}
    {% set y3 = numpy.linalg.norm(x.reshape((shape[0], shape[1], -1)), 3, axis=2) -%}

    const float x[] = {{ x.flatten() | array }};
    const float y1[] = {{ y1.flatten() | array }};
    const float y2[] = {{ y2.flatten() | array }};
    const float y3[] = {{ y3.flatten() | array }};

    const std::int32_t shape[] = {{ shape | array }};
    const std::int32_t reduced[] = {{ y1.shape | array }};
    const std::size_t size = {{ y1.size }};
    const std::size_t ndim = {{ ndim }};

    const auto f = ONNC_RUNTIME_globallppool_float;
    float buffer[size];

    f(nullptr, x, ndim, shape, buffer, ndim, reduced, 1);
    VERIFY(buffer, y1, size);

    f(nullptr, x, ndim, shape, buffer, ndim, reduced, 2);
    VERIFY(buffer, y2, size);

    f(nullptr, x, ndim, shape, buffer, ndim, reduced, 3);
    VERIFY(buffer, y3, size);
}
{% endcall -%}
{# vim: set ft=liquid: #}
