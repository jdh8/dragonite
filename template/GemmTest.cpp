#include "dragonite.hpp"

{% macro view(rows, cols, depth, A, B, C, orderA, orderB) -%}
{
    {% set AB = gemm(A.reshape((rows, depth), order=orderA), B.reshape((depth, cols), order=orderB)) -%}
    {% set alpha = numpy.random.randn() -%}
    {% set beta = numpy.random.randn() -%}

    float alpha = {{ alpha }};
    float beta = {{ beta }};
    bool transA = {{ (orderA == 'F') | lower }};
    bool transB = {{ (orderB == 'F') | lower }};

    const float AB[] = {{ AB | flatten }};
    const float Y[] = {{ (alpha * AB + beta * C) | flatten }};
    const float Yr[] = {{ (alpha * AB + beta * C[0]) | flatten }};
    const float Yv[] = {{ (alpha * AB + beta * C.flatten()[numpy.newaxis, range(rows)].T) | flatten }};
    const float Ys[] = {{ (alpha * AB + beta * C[0, 0]) | flatten }};

    const std::int32_t vector[] = { rows, 1 };

    ONNC_RUNTIME_gemm_float(nullptr, A, 2, Ashape, B, 2, Bshape, O, 0, nullptr, buffer, 2, Cshape, 1, -0.0, transA, transB);
    ASSERT_TRUE((dragonite::approx)(buffer, AB, rows * cols));

    ONNC_RUNTIME_gemm_float(nullptr, A, 2, Ashape, B, 2, Bshape, C, 2, Cshape, buffer, 2, Cshape, alpha, beta, transA, transB);
    ASSERT_TRUE((dragonite::approx)(buffer, Y, rows * cols));

    ONNC_RUNTIME_gemm_float(nullptr, A, 2, Ashape, B, 2, Bshape, C, 1, Cshape + 1, buffer, 2, Cshape, alpha, beta, transA, transB);
    ASSERT_TRUE((dragonite::approx)(buffer, Yr, rows * cols));

    ONNC_RUNTIME_gemm_float(nullptr, A, 2, Ashape, B, 2, Bshape, C, 2, vector, buffer, 2, Cshape, alpha, beta, transA, transB);
    ASSERT_TRUE((dragonite::approx)(buffer, Yv, rows * cols));

    ONNC_RUNTIME_gemm_float(nullptr, A, 2, Ashape, B, 2, Bshape, C, 0, nullptr, buffer, 2, Cshape, alpha, beta, transA, transB);
    ASSERT_TRUE((dragonite::approx)(buffer, Ys, rows * cols));
}
{% endmacro -%}

{% macro testcase(name, rows, cols, depth) -%}
SKYPAT_F(Operator_Gemm, {{ name }})
{
    const std::int32_t rows = {{ rows }};
    const std::int32_t cols = {{ cols }};
    const std::int32_t depth = {{ depth }};

    const std::int32_t Ashape[] = { rows, depth };
    const std::int32_t Bshape[] = { depth, cols };
    const std::int32_t Cshape[] = { rows, cols };

    {% set A = numpy.random.randn(rows * depth) -%}
    {% set B = numpy.random.randn(depth * cols) -%}
    {% set C = numpy.random.randn(rows, cols) -%}

    const float A[] = {{ A | flatten }};
    const float B[] = {{ B | flatten }};
    const float C[] = {{ C | flatten }};
    const float O[rows * cols] = { 0 };

    float buffer[rows * cols];

    {{ view(rows, cols, depth, A, B, C, 'C', 'C') }}
    {{ view(rows, cols, depth, A, B, C, 'F', 'C') }}
    {{ view(rows, cols, depth, A, B, C, 'C', 'F') }}
    {{ view(rows, cols, depth, A, B, C, 'F', 'F') }}
}
{% endmacro -%}

{{ testcase("2x2", 2, 2, 2) }}
{{ testcase("543", 5, 4, 5) }}
