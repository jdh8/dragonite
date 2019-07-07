#include "common.hpp"

{% set blocksize = numpy.random.randint(2, 6) -%}
{% set deep = numpy.random.randint(2, 6, 4) * (1, blocksize * blocksize, 1, 1) -%}
{% set size = numpy.prod(deep) -%}
{% set x = numpy.arange(size, dtype=numpy.float32).reshape(deep) -%}
{% set (y,) = "DepthToSpace" | infer(x, blocksize=blocksize) -%}

SKYPAT_F(DepthToSpace, image)
{
    const float x[] = {{ x.flatten() | array }};
    const float y[] = {{ y.flatten() | array }};

    const std::int32_t deep[] = {{ deep | array }};
    const std::int32_t wide[] = {{ y.shape | array }};
    
    float buffer[{{ size }}];

    ONNC_RUNTIME_depthtospace_float(nullptr, x, 4, deep, buffer, 4, wide, {{ blocksize }});
    ASSERT_FALSE(std::memcmp(buffer, y, sizeof(y)));
}
{# vim: set ft=liquid: #}

