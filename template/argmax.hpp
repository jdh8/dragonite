{% macro testcase(operator, name, x, axis=1, keepdims=true) -%}
SKYPAT_F({{ operator }}, {{ name }})
{
    {% set y = x[operator | lower](axis) -%}

    const float x[] = {{ x | flatten }};
    const float y[] = {{ y | flatten }};

    const std::int32_t xshape[] = { {{ x.shape | join(", ") or -1 }} };
    const std::int32_t yshape[] = { {{ (x.shape[:axis] + (1,) * keepdims + x.shape[axis + 1:] or (-1,)) | join(", ") }} };

    const std::size_t size = {{ y.size }};

    float buffer[size];

    ONNC_RUNTIME_{{ operator | lower }}_float(nullptr,
        x, {{ x.ndim }}, xshape,
        buffer, {{ y.ndim + keepdims }}, yshape,
        {{ axis }}, {{ keepdims | lower }});

    ASSERT_FALSE(std::memcmp(buffer, y, size * sizeof(float)));
}
{% endmacro -%}

{% macro batch(operator) -%}
    {% set scalar = numpy.array(numpy.float32(numpy.random.randn())) -%}
    {% set vector = numpy.random.randn(numpy.random.randint(2, 6)).astype(numpy.float32) -%}
    {% set matrix = numpy.random.normal(size = numpy.random.randint(2, 6, 2)).astype(numpy.float32) -%}
    {% set tensor = numpy.random.normal(size = numpy.random.randint(2, 6, numpy.random.randint(3, 6))).astype(numpy.float32) -%}

    {{ testcase(operator, "scalar", scalar, axis=0) -}}
    {{ testcase(operator, "vector", vector, axis=0) -}}
    {{ testcase(operator, "matrix", matrix) -}}
    {{ testcase(operator, "matrix0", matrix, axis=0) -}}
    {{ testcase(operator, "tensor", tensor) -}}
    {{ testcase(operator, "tensor0", tensor, axis=0) -}}
    {{ testcase(operator, "tensor2", tensor, axis=2) -}}
{% endmacro -%}
