#include "dragonite.hpp"

{% macro view(rows, cols, depth, A, B, C, orderA, orderB) -%}
{
    {% set AB = A.reshape((rows, depth), order=orderA) | gemm(B.reshape((depth, cols), order=orderB)) -%}
    {% set alpha = numpy.float32(numpy.random.randn()) -%}
    {% set beta = numpy.float32(numpy.random.randn()) -%}

    float alpha = {{ alpha }};
    float beta = {{ beta }};
    const bool transA = {{ (orderA == 'F') | lower }};
    const bool transB = {{ (orderB == 'F') | lower }};

    const float AB[] = {{ AB.flatten() | array }};
    const float Y[] = {{ (alpha * AB + beta * C).flatten() | array }};
    const float Yrow[] = {{ (alpha * AB + beta * C[0]).flatten() | array }};
    const float Ycol[] = {{ (alpha * AB + beta * C.flatten()[range(rows), numpy.newaxis]).flatten() | array }};
    const float Ysca[] = {{ (alpha * AB + beta * C[0, 0]).flatten() | array }};

    const std::int32_t* Lshape = transA ? ATshape : Ashape;
    const std::int32_t* Rshape = transB ? BTshape : Bshape;
    const std::int32_t column[] = { rows, 1 };

    const char message[] = "orderA='{{ orderA }}', orderB='{{ orderB }}'";
    auto f = ONNC_RUNTIME_gemm_float;

    f(nullptr, A, 2, Lshape, B, 2, Rshape, O, 0, nullptr, buffer, 2, Cshape, 1, -0.0, transA, transB);
    dragonite::verify(buffer, AB, size, message);

    f(nullptr, A, 2, Lshape, B, 2, Rshape, C, 2, Cshape, buffer, 2, Cshape, alpha, beta, transA, transB);
    dragonite::verify(buffer, Y, size, message);

    f(nullptr, A, 2, Lshape, B, 2, Rshape, C, 1, Cshape + 1, buffer, 2, Cshape, alpha, beta, transA, transB);
    dragonite::verify(buffer, Yrow, size, message);

    f(nullptr, A, 2, Lshape, B, 2, Rshape, C, 2, column, buffer, 2, Cshape, alpha, beta, transA, transB);
    dragonite::verify(buffer, Ycol, size, message);

    f(nullptr, A, 2, Lshape, B, 2, Rshape, C, 0, nullptr, buffer, 2, Cshape, alpha, beta, transA, transB);
    dragonite::verify(buffer, Ysca, size, message);
}
{% endmacro -%}

{% macro testcase(name, rows, cols, depth) -%}
SKYPAT_F(Gemm, {{ name }})
{
    const std::int32_t rows = {{ rows }};
    const std::int32_t cols = {{ cols }};
    const std::int32_t depth = {{ depth }};
    const std::int32_t size = rows * cols;

    const std::int32_t Ashape[] = { rows, depth };
    const std::int32_t Bshape[] = { depth, cols };
    const std::int32_t Cshape[] = { rows, cols };

    const std::int32_t ATshape[] = { depth, rows };
    const std::int32_t BTshape[] = { cols, depth };

    {% set A = numpy.random.randn(rows * depth).astype(numpy.float32) -%}
    {% set B = numpy.random.randn(depth * cols).astype(numpy.float32) -%}
    {% set C = numpy.random.randn(rows, cols).astype(numpy.float32) -%}

    const float A[] = {{ A.flatten() | array }};
    const float B[] = {{ B.flatten() | array }};
    const float C[] = {{ C.flatten() | array }};
    const float O[size] = { 0 };

    float buffer[size];

    {{ view(rows, cols, depth, A, B, C, 'C', 'C') | indent }}
    {{ view(rows, cols, depth, A, B, C, 'F', 'C') | indent }}
    {{ view(rows, cols, depth, A, B, C, 'C', 'F') | indent }}
    {{ view(rows, cols, depth, A, B, C, 'F', 'F') | indent }}
}
{% endmacro -%}

{{ testcase("basic", 2, 2, 2) }}
{{ testcase("hetero", 5, 4, 3) }}
{{ testcase("shallow", 8, 3, 1) }}
{{ testcase("vector", 3, 1, 9) -}}
{# vim: set ft=liquid: #}
