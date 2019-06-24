#ifndef ONNC_DRAGONITE_HPP
#define ONNC_DRAGONITE_HPP

#ifdef __GNUC__
#define restrict __restrict
#else
#define restrict
#endif

extern "C" {
#include <onnc/Runtime/onnc-runtime.h>
}

#undef restrict

#include <skypat/skypat.h>
#include <cstddef>
#include <cstdint>
#include <cmath>

namespace dragonite {

template<int Power>
struct LpNorm;

template<>
struct LpNorm<1>
{
    template<typename T>
    static T compute(const T* x, std::size_t n)
    {
        using std::abs;
        T accumulator = -0.0;

        for (std::size_t i = 0; i < n; ++i)
            accumulator += abs(x[i]);

        return accumulator;
    }
};

template<>
struct LpNorm<2>
{
    template<typename T>
    static T compute(const T* x, std::size_t n)
    {
        using std::sqrt;
        T accumulator = -0.0;

        for (std::size_t i = 0; i < n; ++i)
            accumulator += x[i] * x[i];

        return sqrt(accumulator);
    }

    static float compute(const float* x, std::size_t n)
    {
        double accumulator = -0.0;

        for (std::size_t i = 0; i < n; ++i)
            accumulator += x[i] * x[i];

        return std::sqrt(accumulator);
    }
};

template<>
struct LpNorm<-1>
{
    template<typename T>
    static T compute(const T* x, std::size_t n)
    {
        using std::abs;
        T accumulator = 0;

        for (std::size_t i = 0; i < n; ++i) {
            T candidate = abs(x[i]);

            if (candidate > accumulator)
                accumulator = candidate;
        }
        return accumulator;
    }
};

template<int Power, typename T>
T lpnorm(const T* x, std::size_t n)
{
    return LpNorm<Power>::compute(x, n);
}

inline bool approx(float* buffer, const float* answer, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        buffer[i] -= answer[i];

    return lpnorm<2>(buffer, n) < 1e-6f * lpnorm<2>(answer, n);
}

} // namespace dragonite
#endif
