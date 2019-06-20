import sys

header = '''#define restrict __restrict
extern "C" {
#include <onnc/Runtime/onnc-runtime.h>
}
#undef restrict

#include <skypat/skypat.h>
#include <cmath>
#include <cstdint>

using std::int32_t;
'''

def format(array):
    return str(array.tolist()).replace('[', '{').replace(']', '}').replace("nan", "NAN").replace("inf", "INFINITY")

def declare(constructor, prefix):
    return ""

def define(stuff, prefix):
    return ""

def parametrize(stuff):
    return ""

def body(name, f, *x, position = sys.maxsize):
    y = f(*x)
    parameters = list(map(parametrize, x))
    parameters.insert(position, parametrize(y))

    code = "" + declare(type(y), "output") + define(y, "answer")

    for i in range(len(x)):
        code += define(x[i], "input_" + str(i))

    return code + name + "(nullptr," + ",".join(parameters) + ");"

body("name", declare, 1, 2)
