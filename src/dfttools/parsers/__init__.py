def __import_all_parsers__():
    import importlib, pkgutil
    for loader, module_name, is_pkg in pkgutil.iter_modules(__path__):
        importlib.import_module("." + module_name, package=__name__)
