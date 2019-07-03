import jinja2
import numpy

import itertools
import os

def run(destination, source):
    environment = jinja2.Environment(loader = jinja2.FileSystemLoader(source))
    environment.globals["itertools"] = itertools
    environment.globals["numpy"] = numpy

    environment.filters["array"] = lambda x: "{ " + (", ".join(map(str, x)) or "dragonite::canary") + " }"
    environment.filters["gemm"] = lambda x, y: x @ y
    environment.filters["tuple"] = tuple

    os.makedirs(destination, exist_ok=True)

    for entry in os.scandir(source):
        if entry.name.endswith(".cpp"):
            environment.get_template(entry.name).stream().dump(os.path.join(destination, entry.name))

run("result", "template")
