import jinja2
import numpy
import sys

with open(sys.argv[2], "w") as stream:
    environment = jinja2.Environment(loader = jinja2.FileSystemLoader('.'))

    environment.globals["gemm"] = lambda x, y: x @ y
    environment.globals["numpy"] = numpy

    environment.filters["flatten"] = lambda x: "{" + ",".join(map(str, x.flatten())) + "}"

    stream.write(environment.get_template(sys.argv[1]).render())
