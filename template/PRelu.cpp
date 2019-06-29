#include "dragonite.hpp"

{% macro testcase(name, x, slope) -%}
SKYPAT_F(PRelu, {{ name }})
{
    const float x[] = {{ x.flatten() | array }};
    const float slope[] = {{ slope.flatten() | array }};
    const float y[] = {{ numpy.where(numpy.signbit(x), numpy.broadcast_to(slope, x.shape) * x, x).flatten() | array }};

    const std::int32_t shape[] = {{ x.shape | array }};
    const std::int32_t slope_shape[] = {{ slope.shape | array }};

    const std::int32_t ndim = {{ x.ndim }};
    const std::size_t size = {{ x.size }};

    float buffer[size];

    ONNC_RUNTIME_prelu_float(nullptr, x, ndim, shape, slope, {{ slope.ndim }}, slope_shape, buffer, ndim, shape);

    ASSERT_TRUE(dragonite::approx(buffer, y, size));
}
{% endmacro -%}

{% set shape = numpy.random.randint(2, 6, numpy.random.randint(3, 6)) %}

{{ testcase("ScalarScalar",
    numpy.array(numpy.float32(numpy.random.randn())),
    numpy.float32(numpy.random.rand())) }}
{{ testcase("VectorScalar",
    numpy.random.normal(size = shape[:1]).astype(numpy.float32),
    numpy.float32(numpy.random.rand())) }}
{{ testcase("MatrixScalar",
    numpy.random.normal(size = shape[:2]).astype(numpy.float32),
    numpy.float32(numpy.random.rand())) }}
{{ testcase("TensorScalar",
    numpy.random.normal(size = shape).astype(numpy.float32),
    numpy.float32(numpy.random.rand())) }}

{{ testcase("VectorVector",
    numpy.random.normal(size = shape[3]).astype(numpy.float32),
    numpy.random.sample(size = shape[3]).astype(numpy.float32)) }}
{{ testcase("MatrixVector",
    numpy.random.normal(size = shape[:2]).astype(numpy.float32),
    numpy.random.sample(size = shape[1]).astype(numpy.float32)) }}
{{ testcase("TensorVector",
    numpy.random.normal(size = shape).astype(numpy.float32),
    numpy.random.sample(size = shape[-1:]).astype(numpy.float32)) }}

{{ testcase("MatrixMatrix",
    numpy.random.normal(size = shape[:2]).astype(numpy.float32),
    numpy.random.sample(size = shape[:2]).astype(numpy.float32)) }}
{{ testcase("TensorMatrix",
    numpy.random.normal(size = shape).astype(numpy.float32),
    numpy.random.sample(size = shape[-2:]).astype(numpy.float32)) }}

{{ testcase("TensorTensor",
    numpy.random.normal(size = shape).astype(numpy.float32),
    numpy.random.sample(size = shape).astype(numpy.float32)) -}}
{# vim: set ft=liquid: #}
