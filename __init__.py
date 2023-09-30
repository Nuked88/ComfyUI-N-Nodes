# code based on pysssss repo
import importlib.util
import glob
import os
import sys
from .nnodes import init, get_ext_dir
import folder_paths
import requests

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
import zipfile
def downloader(link):
    print("Downloading dependencies...")
    response = requests.get(link, stream=True)
    try:
        os.makedirs(folder_paths.get_temp_directory())
    except:
        pass
    temp_file = os.path.join(folder_paths.get_temp_directory(), "file.zip")
    with open(temp_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk: # Filtra i chunk vuoti
                f.write(chunk) # Scrivi il chunk nel file




    # Estrai lo zip nella cartella desiderata
    zip_file = zipfile.ZipFile(temp_file) # Apri il file zip
    target_dir = os.path.join(folder_paths.folder_names_and_paths["custom_nodes"][0][0],"ComfyUI-N-Nodes","libs","rifle") # Cartella dove estrarre lo zip

    zip_file.extractall(target_dir) # Estrai tutti i file dello zip nella cartella





  

if init():
    py = get_ext_dir("py")
    files = glob.glob("*.py", root_dir=py, recursive=False)
    install_and_import('moviepy')
    install_and_import('cv2')
    install_and_import('git')
    install_and_import('zipfile')

    #git clone https://github.com/hzwer/Practical-RIFE.git
    from git import Repo
    if not os.path.exists(os.path.join(folder_paths.folder_names_and_paths["custom_nodes"][0][0],"ComfyUI-N-Nodes","libs","rifle")):
        Repo.clone_from("https://github.com/hzwer/Practical-RIFE.git", os.path.join(folder_paths.folder_names_and_paths["custom_nodes"][0][0],"ComfyUI-N-Nodes","libs","rifle"))
    #if train_log folder not exists
    if not os.path.exists(os.path.join(folder_paths.folder_names_and_paths["custom_nodes"][0][0],"ComfyUI-N-Nodes","libs","rifle","train_log")):
        downloader("https://www.animecast.net/download/RIFE_trained_model_v4.7.zip")
                   #"https://github.com/Nuked88/DreamingAI/raw/main/RIFE_trained_model_v4.7.zip")

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