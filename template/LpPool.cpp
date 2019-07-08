#include "common.hpp"

{% from "mod/channel.cpp" import channel -%}

{% call(name, channels, image, kernel) channel() -%}
SKYPAT_F(LpPool, {{ name }})
{
    {% set ndim = kernel | length -%}
    {% set pads = numpy.minimum(numpy.tile(kernel - 1, 2), numpy.random.randint(0, 8, 2 * ndim)) -%}
    {% set strides = numpy.random.randint(1, 4, ndim) -%}
    {% set x = numpy.random.standard_cauchy(numpy.concatenate((channels, image))).astype(numpy.float32) -%}

    const char autopad[] = "NOTSET";

    std::int32_t kernel[] = {{ kernel | array }};
    std::int32_t pads[] = {{ pads | array }};
    std::int32_t strides[] = {{ strides | array }};
    const std::int32_t shape[] = {{ x.shape | array }};

    const std::size_t rows = {{ numpy.prod(channels) }};
    const std::size_t ndim = {{ ndim }};

    const float x[] = {{ x.flatten() | array }};

    {% for p in (1, 2, 3) -%}
    {
        {% set (y,) = "LpPool" | infer(x, kernel_shape=kernel, pads=pads, strides=strides) -%}

        const float y[] = {{ y.flatten() | array }};
        const std::int32_t reduced[] = {{ y.shape | array }};
        const std::size_t cols = {{ numpy.prod(y.shape[2:]) }};
        const auto matrix = reinterpret_cast<const float (*)[cols]>(y);
        float buffer[rows][cols];

        ONNC_RUNTIME_lppool_float(nullptr, x, ndim + 2, shape, *buffer, ndim + 2, reduced,
            autopad, kernel, ndim, {{ p }}, pads, 2 * ndim, strides, ndim);

        ASSERT_TRUE(std::equal(matrix, matrix + rows, buffer, [cols](const float* actual, const float* estimate)
        {
            return dragonite::norm(actual, estimate, cols) <= 1e-5 * dragonite::norm(actual, cols);
        }));
    }
    {% endfor -%}
}
{% endcall -%}
{# vim: set ft=liquid: #}
