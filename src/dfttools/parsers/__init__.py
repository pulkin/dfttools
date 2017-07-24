from ..types import Basis

def __import_all_parsers__():
    import importlib, pkgutil
    for loader, module_name, is_pkg in pkgutil.iter_modules(__path__):
        importlib.import_module("."+module_name, package = __name__)

def default_real_space_basis(vectors, kind = 'default', meta = None):
    "A default basis in the real space with units assigned."
    return Basis(vectors, kind = kind, meta = meta, units = "angstrom")
