{% macro outer(operator, name, shape) -%}
SKYPAT_F({{ operator }}, {{ name }})
{
    {% set x = numpy.random.normal(size = shape).astype(numpy.float32) -%}
    {% set ndim = shape | length -%}

    const float x[] = {{ x.flatten() | array }};
    const std::int32_t shape[] = {{ shape | array }};
    const auto f = ONNC_RUNTIME_{{ operator | lower }}_float;

    {% for r in range(ndim, -1, -1) -%}
        {% for axes in itertools.combinations(range(ndim), r) -%}
            {{ caller(x, axes) -}}
        {% endfor -%}
    {% endfor %}
}
{% endmacro -%}

{% macro inner(y, axes) -%}
{
    const float y[] = {{ y.flatten() | array }};
    const std::int32_t axes[] = {{ axes | array }};
    const std::size_t ndim = {{ y.ndim }};
    const std::size_t reductions = {{ axes | length }};
    const std::size_t size = {{ y.size }};

    float buffer[size];

    const std::int32_t unsqueezed[] = {{ y.shape | array }};
    const std::int32_t squeezed[] = {{ y.squeeze(axes).shape | array }};

    f(nullptr, x, ndim, shape, buffer, ndim, unsqueezed, axes, reductions, true);
    dragonite::verify(buffer, y, size, "axes={{ axes }}, unsqueezed");

    f(nullptr, x, ndim, shape, buffer, ndim - reductions, squeezed, axes, reductions, false);
    dragonite::verify(buffer, y, size, "axes={{ axes }}, squeezed");
}
{% endmacro -%}
{# vim: set ft=liquid: #}
