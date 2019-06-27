import jinja2
import numpy
import sys

with open(sys.argv[2], "w") as stream:
    def softmax(x, *, axis=1):
        axes = *range(axis, x.ndim),
        fmax = numpy.fmax.reduce(x, axis=axes, keepdims=True)
        exp = numpy.exp(x - fmax)
        return exp / numpy.sum(exp, axis=axes, keepdims=True)

    environment = jinja2.Environment(loader = jinja2.FileSystemLoader('.'))

    environment.globals["gemm"] = lambda x, y: x @ y
    environment.globals["numpy"] = numpy

    environment.filters["flatten"] = lambda x: "{ " + ", ".join(map(str, x.flatten())) + " }"
    environment.filters["softmax"] = softmax

    stream.write(environment.get_template(sys.argv[1]).render())
