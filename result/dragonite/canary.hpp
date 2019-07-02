#ifndef DRAGONITE_CANARY_HPP
#define DRAGONITE_CANARY_HPP

#include <limits>

namespace dragonite {

struct Canary
{
    template<typename T>
    operator T() const
    {
        typedef std::numeric_limits<T> Traits;

        if (Traits::has_signaling_NaN)
            return Traits::signaling_NaN();

        if (Traits::has_quiet_NaN)
            return Traits::quiet_NaN();

        if (Traits::has_infinity)
            return Traits::infinity();

        return Traits::max();
    }
};

const Canary canary;

} // namespace dragonite

#endif
