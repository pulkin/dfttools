from __future__ import absolute_import
import json
import numpy


class JSONEncoderWithArray(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, numpy.ndarray):
            if numpy.iscomplexobj(o):
                return dict(
                    _custom_type="numpy-complex",
                    data_r=o.real.tolist(),
                    data_i=o.imag.tolist(),
                )
            else:
                return dict(
                    _custom_type="numpy",
                    data=o.tolist(),
                )
        else:
            return super(JSONEncoderWithArray, self).default(o)


def object_hook(d):
    if "_custom_type" in d:
        tp = d["_custom_type"]
        if tp == "numpy-complex":
            return numpy.vectorize(complex)(d["data_r"], numpy.array(d["data_i"]))
        elif tp == "numpy":
            return numpy.array(d["data"])
        else:
            raise TypeError("Unknown type to deserialize: {}".format(tp))
    else:
        return d


def dump(*args, **kwargs):
    kwargs["cls"] = JSONEncoderWithArray
    return json.dump(*args, **kwargs)


def dumps(*args, **kwargs):
    kwargs["cls"] = JSONEncoderWithArray
    return json.dumps(*args, **kwargs)


def load(*args, **kwargs):
    kwargs["object_hook"] = object_hook
    return json.load(*args, **kwargs)


def loads(*args, **kwargs):
    kwargs["object_hook"] = object_hook
    return json.loads(*args, **kwargs)
