import jinja2
import numpy
import onnx
import onnxruntime

import itertools
import os

def infer(operator, *inputs, **attributes):
    tags = {
        numpy.void:       onnx.TensorProto.UNDEFINED,
        numpy.float32:    onnx.TensorProto.FLOAT,
        numpy.uint8:      onnx.TensorProto.UINT8,
        numpy.int8:       onnx.TensorProto.INT8,
        numpy.uint16:     onnx.TensorProto.UINT16,
        numpy.int16:      onnx.TensorProto.INT16,
        numpy.int32:      onnx.TensorProto.INT32,
        numpy.int64:      onnx.TensorProto.INT64,
        numpy.bytes_:     onnx.TensorProto.STRING,
        numpy.str_:       onnx.TensorProto.STRING,
        numpy.bool_:      onnx.TensorProto.BOOL,
        numpy.float16:    onnx.TensorProto.FLOAT16,
        numpy.float64:    onnx.TensorProto.DOUBLE,
        numpy.uint32:     onnx.TensorProto.UINT32,
        numpy.uint64:     onnx.TensorProto.UINT64,
        numpy.complex64:  onnx.TensorProto.COMPLEX64,
        numpy.complex128: onnx.TensorProto.COMPLEX128,
    }

    def length(operator, split="**", **attributes):
        return len(split) if operator == "Split" else 1

    arity = range(len(inputs))
    split = range(length(operator, **attributes))
    itensors = [onnx.helper.make_tensor_value_info(f"%i{k}", tags[inputs[k].dtype.type], inputs[k].shape) for k in arity]
    otensors = [onnx.helper.make_empty_tensor_value_info(f"%o{k}") for k in split]
    node = onnx.helper.make_node(operator, [t.name for t in itensors], [t.name for t in otensors], **attributes)
    graph = onnx.helper.make_graph([node], operator, itensors, otensors)
    model = onnx.helper.make_model(graph)
    sess = onnxruntime.InferenceSession(model.SerializeToString())
    outputs = sess.run([f"%o{k}" for k in split], { f"%i{k}": inputs[k] for k in arity })

    if operator == "Split":
        return outputs

    [output] = outputs
    return output

def run(destination, source):
    environment = jinja2.Environment(loader = jinja2.FileSystemLoader(source))
    environment.globals["itertools"] = itertools
    environment.globals["numpy"] = numpy

    environment.filters["array"] = lambda x: "{ " + (", ".join(map(str, x)) or "dragonite::canary") + " }"
    environment.filters["infer"] = infer
    environment.filters["gemm"] = lambda x, y: x @ y
    environment.filters["tuple"] = tuple

    os.makedirs(destination, exist_ok=True)

    for entry in os.scandir(source):
        if entry.name.endswith(".cpp"):
            environment.get_template(entry.name).stream().dump(os.path.join(destination, entry.name))

run("result", "template")
