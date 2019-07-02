{% macro outer(operator, name, shape) -%}
SKYPAT_F({{ operator }}, {{ name }})
{
    {% set ndim = shape | length -%}
    {% set powerset = (ndim * ((1, 0),)) | star(itertools.product) -%}

    const auto f = ONNC_RUNTIME_{{ operator | lower }}_float;
    const std::int32_t ndim = {{ ndim }};

    {% for a in powerset -%}
        {% for b in powerset -%}
            {{ caller(numpy.power(shape, a), numpy.power(shape, b)) | indent }}
        {% endfor -%}
    {% endfor %}
}
{% endmacro -%}

{% macro inner(c, a, b, tolerance=1) -%}
{
    const float a[] = {{ a.flatten() | array }};
    const float b[] = {{ b.flatten() | array }};
    const float c[] = {{ c.flatten() | array }};

    const std::int32_t ashape[] = {{ a.shape | array }};
    const std::int32_t bshape[] = {{ b.shape | array }};
    const std::int32_t cshape[] = {{ c.shape | array }};

    bool (*predicate)(float, float) = dragonite::within<{{ tolerance }}>;
    const std::size_t size = {{ c.size }};
    float buffer[size];

    f(nullptr, a, ndim, ashape, b, ndim, bshape, buffer, ndim, cshape);
    ASSERT_TRUE(std::equal(c, c + size, buffer, predicate));

    for (std::int32_t squeezed = 1; squeezed <= ndim && ashape[squeezed - 1] == 1; ++squeezed) {
        f(nullptr, a, ndim - squeezed, ashape + squeezed, b, ndim, bshape, buffer, ndim, cshape);
        ASSERT_TRUE_MSG(std::equal(c, c + size, buffer, predicate), squeezed);
    }

    for (std::int32_t squeezed = 1; squeezed <= ndim && bshape[squeezed - 1] == 1; ++squeezed) {
        f(nullptr, a, ndim, ashape, b, ndim - squeezed, bshape + squeezed, buffer, ndim, cshape);
        ASSERT_TRUE_MSG(std::equal(c, c + size, buffer, predicate), squeezed);
    }
}
{% endmacro -%}
{# vim: set ft=liquid: #}
