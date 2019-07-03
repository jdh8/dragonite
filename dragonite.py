import jinja2
import numpy

import itertools
import sys

with open(sys.argv[2], "w") as stream:
    environment = jinja2.Environment(loader = jinja2.FileSystemLoader([".", "template"]))
    environment.globals["itertools"] = itertools
    environment.globals["numpy"] = numpy

    environment.filters["array"] = lambda x: "{ " + (", ".join(map(str, x)) or "dragonite::canary") + " }"
    environment.filters["gemm"] = lambda x, y: x @ y
    environment.filters["tuple"] = tuple

    stream.write(environment.get_template(sys.argv[1]).render())
