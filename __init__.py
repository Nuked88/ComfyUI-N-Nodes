# code based on pysssss repo
import importlib.util
import glob
import os
import sys
from .nnodes import init, get_ext_dir

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def install_and_import(package):
    import importlib
    try:
        print("Detected: ", package)
        importlib.import_module(package)
    except ImportError:
        import pip
        pip.main(['install', package])
    finally:
        globals()[package] = importlib.import_module(package)

def check_module(package):
    import importlib
    try:
        print("Detected: ", package)
        importlib.import_module(package)
        return True
    except ImportError:
        return False



if init():
    py = get_ext_dir("py")
    files = glob.glob("*.py", root_dir=py, recursive=False)
    install_and_import('moviepy')
    install_and_import('cv2')
    for file in files:
        try:
            name = os.path.splitext(file)[0]
            spec = importlib.util.spec_from_file_location(name, os.path.join(py, file))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            if hasattr(module, "NODE_CLASS_MAPPINGS") and getattr(module, "NODE_CLASS_MAPPINGS") is not None:
                NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
                if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS") and getattr(module, "NODE_DISPLAY_NAME_MAPPINGS") is not None:
                    NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
        except Exception as e:
            print(e)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]