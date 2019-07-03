#ifndef DRAGONITE_COMMON_HPP
#define DRAGONITE_COMMON_HPP

#include "canary.hpp"
#include "norm.hpp"
#include "within.hpp"
#include <skypat/skypat.h>

#ifdef __GNUC__
#define restrict __restrict
#else
#define restrict
#endif

extern "C" {
#include <onnc/Runtime/onnc-runtime.h>
}

#undef restrict
#endif
