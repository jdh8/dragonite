#include "dragonite.hpp"

{% macro testcase(name, x, axis=1) -%}
SKYPAT_F(Softmax, {{ name }})
{
    {% set axes = range(axis, x.ndim) | tuple -%}
    {% set fmax = numpy.fmax.reduce(x, axis=axes, keepdims=true) -%}
    {% set exp = numpy.exp(x - fmax) -%}
    {% set y = exp / numpy.sum(exp, axis=axes, keepdims=true) -%}

    const float x[] = {{ x | flatten }};
    const float y[] = {{ y | flatten }};

    const std::int32_t shape[] = { {{ x.shape | join(", ") or -1 }} };
    const std::int32_t ndim = {{ x.ndim }};
    const std::size_t size = {{ x.size }};

    float buffer[size];

    ONNC_RUNTIME_softmax_float(nullptr, x, ndim, shape, buffer, ndim, shape, {{ axis }});

    dragonite::verify(buffer, y, size);
}
{% endmacro -%}

{% set scalar = numpy.float32(numpy.random.randn()) -%}
{% set vector = numpy.random.randn(numpy.random.randint(2, 6)).astype(numpy.float32) -%}
{% set matrix = numpy.random.normal(size = numpy.random.randint(2, 6, 2)).astype(numpy.float32) -%}
{% set tensor = numpy.random.normal(size = numpy.random.randint(2, 6, numpy.random.randint(3, 6))).astype(numpy.float32) -%}

{{ testcase("scalar", scalar) }}
{{ testcase("vector", vector, axis=0) }}
{{ testcase("matrix", matrix) }}
{{ testcase("matrix0", matrix, axis=0) }}
{{ testcase("tensor", tensor) }}
{{ testcase("tensor0", tensor, axis=0) }}
{{ testcase("tensor2", tensor, axis=2) -}}
{{ testcase("tensorlast", tensor, axis= tensor.ndim - 1) -}}
{# vim: set ft=liquid: #}
