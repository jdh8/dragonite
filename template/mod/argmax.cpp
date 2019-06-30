{% macro testcase(operator, name, shape) -%}
SKYPAT_F({{ operator }}, {{ name }})
{
    {% set x = numpy.random.normal(size=shape).astype(numpy.float32) -%}
    {% set ndim = shape | length -%}

    const float x[] = {{ x.flatten() | array }};

    const std::int32_t shape[] = {{ shape | array }};
    const std::int32_t ndim = {{ ndim }};

    const auto f = ONNC_RUNTIME_{{ operator | lower }}_float;

    {% for axis in range(ndim) -%}
    {
        {% set y = x[operator | lower](axis) -%}

        const float y[] = {{ y.flatten() | array }};

        const std::int32_t unsqueezed[] = {{ (y.shape[:axis] + (1,) + y.shape[axis + 1:]) | array }};
        const std::int32_t squeezed[] = {{ y.shape | array }};

        float buffer[{{ y.size }}];

        f(nullptr, x, ndim, shape, buffer, ndim, unsqueezed, {{ axis }}, true);
        ASSERT_FALSE(std::memcmp(buffer, y, sizeof(y)));

        f(nullptr, x, ndim, shape, buffer, ndim - 1, squeezed, {{ axis }}, false);
        ASSERT_FALSE(std::memcmp(buffer, y, sizeof(y)));
    }
    {% else -%}
    {
        float buffer;

        f(nullptr, x, ndim, shape, &buffer, 0, nullptr, 0, true);
        ASSERT_EQ(buffer, 0);

        f(nullptr, x, ndim, shape, &buffer, 0, nullptr, 0, false);
        ASSERT_EQ(buffer, 0);
    }
    {% endfor %}
}
{% endmacro -%}
{# vim: set ft=liquid: #}
