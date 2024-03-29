{% macro outer(operator, name, shape, rng=numpy.random.standard_cauchy) -%}
SKYPAT_F({{ operator }}, {{ name }})
{
    {% set x = rng(size=shape).astype(numpy.float32) -%}
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

{% macro assert(message) -%}
    ASSERT_LE(dragonite::norm(y, buffer, size), 1e-5 * dragonite::norm(y, size));
{% endmacro -%}

{% macro inner(y, axes, verify=assert) -%}
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
    {{ verify("axes="~ axes ~", unsqueezed") }}

    f(nullptr, x, ndim, shape, buffer, ndim - reductions, squeezed, axes, reductions, false);
    {{ verify("axes="~ axes ~", squeezed") }}
}
{% endmacro -%}
{# vim: set ft=liquid: #}
