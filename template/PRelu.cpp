#include "dragonite/common.hpp"

{% from "mod/batch.cpp" import batch -%}

{% macro subtest(x, subshape) -%}
{
    {% set slope = numpy.random.gumbel(size=subshape).astype(numpy.float32) -%}

    const float slope[] = {{ slope.flatten() | array }};
    const float y[] = {{ numpy.where(numpy.signbit(x), slope * x, x).flatten() | array }};

    const std::int32_t subshape[] = {{ subshape | array }};
    bool (*predicate)(float, float) = dragonite::within<1>;

    ONNC_RUNTIME_prelu_float(nullptr, x, ndim, shape, slope, ndim, subshape, buffer, ndim, shape);
    ASSERT_TRUE(std::equal(y, y + size, buffer, predicate));

    for (std::int32_t squeezed = 1; squeezed <= ndim && subshape[squeezed - 1] == 1; ++squeezed) {
        ONNC_RUNTIME_prelu_float(nullptr, x, ndim, shape, slope, ndim - squeezed, subshape + squeezed, buffer, ndim, shape);
        ASSERT_TRUE_MSG(std::equal(y, y + size, buffer, predicate), squeezed);
    }
}
{% endmacro -%}

{% call(name, shape) batch() -%}
SKYPAT_F(PRelu, {{ name }})
{
    {% set x = numpy.random.standard_cauchy(shape).astype(numpy.float32) -%}
    {% set ndim = shape | length -%}

    const float x[] = {{ x.flatten() | array }};

    const std::int32_t shape[] = {{ shape | array }};
    const std::int32_t ndim = {{ ndim }};
    const std::size_t size = {{ x.size }};

    float buffer[size];

    {% for bits in (ndim * ((1, 0),)) | star(itertools.product) -%}
        {{ subtest(x, numpy.power(shape, bits)) | indent }}
    {% endfor %}
}
{% endcall -%}
{# vim: set ft=liquid: #}
