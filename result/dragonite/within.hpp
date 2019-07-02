#ifndef DRAGONITE_WITHIN_HPP
#define DRAGONITE_WITHIN_HPP

#include <cstdint>
#include <cstring>

namespace dragonite {

template<unsigned tolerance>
inline bool within(float x, float y)
{
    std::uint32_t a, b;

    std::memcpy(&a, &x, sizeof(std::uint32_t));
    std::memcpy(&b, &y, sizeof(std::uint32_t));

    return a - b + tolerance <= 2 * tolerance;
}

} // namespace dragonite

#endif


