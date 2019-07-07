#include "common.hpp"

{% from "mod/channel.cpp" import channel -%}

{% call(name, channels, image, kernel) channel() -%}
SKYPAT_F(LpPool, {{ name }})
{
    {% set ndim = kernel | length -%}
    {% set pads = numpy.minimum(numpy.tile(kernel - 1, 2), numpy.random.randint(0, 8, 2 * ndim)) -%}
    {% set strides = numpy.random.randint(1, 4, ndim) -%}
    {% set x = numpy.random.standard_cauchy(numpy.concatenate((channels, image))).astype(numpy.float32) -%}
    {% set (y,) = "LpPool" | infer(x, kernel_shape=kernel, pads=pads, strides=strides) -%}

    const char autopad[] = "NOTSET";
    const std::int32_t kernel[] = {{ kernel | array }};
    const std::int32_t pads[] = {{ pads | array }};
    const std::int32_t strides[] = {{ strides | array }};
    const float x[] = {{ x.flatten() | array }};
    const float y[] = {{ y.flatten() | array }};
}
{% endcall -%}
{# vim: set ft=liquid: #}
