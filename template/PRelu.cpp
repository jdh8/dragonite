#include "dragonite.hpp"

{% macro testcase(name, x, slope) -%}
SKYPAT_F(PRelu, {{ name }})
{
    const float x[] = {{ x | flatten }};
    const float slope[] = {{ slope | flatten }};
    const float y[] = {{ numpy.where(numpy.signbit(x), numpy.broadcast_to(slope, x.shape) * x, x) | flatten }};

    const std::int32_t shape[] = { {{ x.shape | join(", ") or -1 }} };
    const std::int32_t slope_shape[] = { {{ slope.shape | join(", ") or -1 }} };

    float buffer[{{ x.size }}];

    ONNC_RUNTIME_prelu_float(nullptr,
        x, {{ x.ndim }}, shape,
        slope, {{ slope.ndim }}, slope_shape,
        buffer, {{ x.ndim }}, shape);

    dragonite::verify(buffer, y, {{ x.size }});
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
