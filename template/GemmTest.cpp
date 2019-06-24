#include "dragonite.hpp"

#define COMPARE(buffer, answer, size, message) do {                                        \
    using dragonite::lpnorm;                                                               \
    dragonite::subtract(buffer, answer, size);                                             \
    ASSERT_TRUE_MSG(lpnorm<1>(buffer, size) < 3e-7f * lpnorm<1>(answer, size), message);   \
    ASSERT_TRUE_MSG(lpnorm<2>(buffer, size) < 5e-7f * lpnorm<2>(answer, size), message);   \
    ASSERT_TRUE_MSG(lpnorm<-1>(buffer, size) < 1e-6f * lpnorm<-1>(answer, size), message); \
} while (0)

{% macro view(rows, cols, depth, A, B, C, orderA, orderB) -%}
{
    {% set AB = gemm(A.reshape((rows, depth), order=orderA), B.reshape((depth, cols), order=orderB)) -%}
    {% set alpha = numpy.random.randn() -%}
    {% set beta = numpy.random.randn() -%}

    float alpha = {{ alpha }};
    float beta = {{ beta }};
    const bool transA = {{ (orderA == 'F') | lower }};
    const bool transB = {{ (orderB == 'F') | lower }};

    const float AB[] = {{ AB | flatten }};
    const float Y[] = {{ (alpha * AB + beta * C) | flatten }};
    const float Yrow[] = {{ (alpha * AB + beta * C[0]) | flatten }};
    const float Ycol[] = {{ (alpha * AB + beta * C.flatten()[numpy.newaxis, range(rows)].T) | flatten }};
    const float Ysca[] = {{ (alpha * AB + beta * C[0, 0]) | flatten }};

    const std::int32_t* Lshape = transA ? ATshape : Ashape;
    const std::int32_t* Rshape = transB ? BTshape : Bshape;
    const std::int32_t column[] = { rows, 1 };

    const char message[] = "orderA='{{ orderA }}', orderB='{{ orderB }}'";
    auto f = ONNC_RUNTIME_gemm_float;

    f(nullptr, A, 2, Lshape, B, 2, Rshape, O, 0, nullptr, buffer, 2, Cshape, 1, -0.0, transA, transB);
    COMPARE(buffer, AB, size, message);

    f(nullptr, A, 2, Lshape, B, 2, Rshape, C, 2, Cshape, buffer, 2, Cshape, alpha, beta, transA, transB);
    COMPARE(buffer, Y, size, message);

    f(nullptr, A, 2, Lshape, B, 2, Rshape, C, 1, Cshape + 1, buffer, 2, Cshape, alpha, beta, transA, transB);
    COMPARE(buffer, Yrow, size, message);

    f(nullptr, A, 2, Lshape, B, 2, Rshape, C, 2, column, buffer, 2, Cshape, alpha, beta, transA, transB);
    COMPARE(buffer, Ycol, size, message);

    f(nullptr, A, 2, Lshape, B, 2, Rshape, C, 0, nullptr, buffer, 2, Cshape, alpha, beta, transA, transB);
    COMPARE(buffer, Ysca, size, message);
}
{% endmacro -%}

{% macro testcase(name, rows, cols, depth) -%}
SKYPAT_F(Operator_Gemm, {{ name }})
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

    {% set A = numpy.random.randn(rows * depth) -%}
    {% set B = numpy.random.randn(depth * cols) -%}
    {% set C = numpy.random.randn(rows, cols) -%}

    const float A[] = {{ A | flatten }};
    const float B[] = {{ B | flatten }};
    const float C[] = {{ C | flatten }};
    const float O[size] = { 0 };

    float buffer[size];

    {{ view(rows, cols, depth, A, B, C, 'C', 'C') }}
    {{ view(rows, cols, depth, A, B, C, 'F', 'C') }}
    {{ view(rows, cols, depth, A, B, C, 'C', 'F') }}
    {{ view(rows, cols, depth, A, B, C, 'F', 'F') }}
}
{% endmacro -%}

{{ testcase("basic", 2, 2, 2) }}
{{ testcase("hetero", 5, 4, 3) }}
{{ testcase("shallow", 8, 3, 1) }}
{{ testcase("vector", 3, 1, 9) }}
