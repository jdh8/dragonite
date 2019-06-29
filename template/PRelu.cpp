#include "dragonite.hpp"

{% macro subtest(x, subshape) -%}
{
    {% set slope = numpy.random.sample(size = subshape).astype(numpy.float32) -%}

    const float slope[] = {{ slope.flatten() | array }};
    const float y[] = {{ numpy.where(numpy.signbit(x), slope * x, x).flatten() | array }};

    const std::int32_t subshape[] = {{ subshape | array }};

    ONNC_RUNTIME_prelu_float(nullptr, x, ndim, shape, slope, ndim, subshape, buffer, ndim, shape);
    ASSERT_TRUE(dragonite::approx(buffer, y, size));

    for (std::int32_t squeezed = 1; squeezed <= ndim && subshape[squeezed - 1] == 1; ++squeezed) {
        ONNC_RUNTIME_prelu_float(nullptr, x, ndim, shape, slope, ndim - squeezed, subshape + squeezed, buffer, ndim, shape);
        ASSERT_TRUE_MSG(dragonite::approx(buffer, y, size), squeezed);
    }
}
{% endmacro -%}

{% macro testcase(name, shape) -%}
SKYPAT_F(PRelu, {{ name }})
{
    {% set x = numpy.random.normal(size = shape).astype(numpy.float32) -%}
    {% set ndim = shape | length %}

    const float x[] = {{ x.flatten() | array }};

    const std::int32_t shape[] = {{ shape | array }};
    const std::int32_t ndim = {{ ndim }};
    const std::size_t size = {{ x.size }};

    float buffer[size];

    {% for bits in (ndim * ((1, 0),)) | star(itertools.product) -%}
        {{ subtest(x, numpy.power(shape, bits)) | indent }}
    {% endfor %}
}
{% endmacro -%}

{{ testcase("scalar", ()) }}
{{ testcase("vector", numpy.random.randint(2, 6, 1)) }}
{{ testcase("matrix", numpy.random.randint(2, 6, 2)) }}
{{ testcase("tensor", numpy.random.randint(2, 6, numpy.random.randint(3, 6))) -}}
{# vim: set ft=liquid: #}
