#include "common.hpp"

{% set blocksize = numpy.random.randint(2, 6) -%}
{% set wide = numpy.random.randint(2, 6, 4) * (1, 1, blocksize, blocksize) -%}
{% set size = numpy.prod(wide) -%}
{% set x = numpy.arange(size, dtype=numpy.float32).reshape(wide) -%}
{% set (y,) = "SpaceToDepth" | infer(x, blocksize=blocksize) -%}

SKYPAT_F(SpaceToDepth, image)
{
    const float x[] = {{ x.flatten() | array }};
    const float y[] = {{ y.flatten() | array }};

    const std::int32_t wide[] = {{ wide | array }};
    const std::int32_t deep[] = {{ y.shape | array }};
    
    float buffer[{{ size }}];

    ONNC_RUNTIME_spacetodepth_float(nullptr, x, 4, wide, buffer, 4, deep, {{ blocksize }});
    ASSERT_FALSE(std::memcmp(buffer, y, sizeof(y)));
}
{# vim: set ft=liquid: #}

