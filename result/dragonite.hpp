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

#if !(defined(__GNUC__) || defined(_MSC_VER))
#define __restrict
#endif
template<typename T>
inline void subtract(T* __restrict minuend, const T* __restrict subtrahend, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        minuend[i] -= subtrahend[i];
}
#if !(defined(__GNUC__) || defined(_MSC_VER))
#undef __restrict
#endif
} // namespace dragonite
#endif
