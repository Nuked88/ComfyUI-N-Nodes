import importlib.util
import os
import sys
from .nnodes import init, get_ext_dir,check_and_install,downloader,get_commit,color
import folder_paths
import traceback
from pathlib import Path
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


if init():
    print("------------------------------------------")
    print(f"{color.BLUE}### N-Suite Revision:{color.END} {color.GREEN}{get_commit()} {color.END}")
    py = Path(get_ext_dir("py"))
    files = list(py.glob("*.py"))
    check_and_install('packaging')
    check_and_install('py-cpuinfo',"cpuinfo")
    check_and_install('gitpython','git')
    check_and_install('moviepy')
    check_and_install("opencv-python","cv2") 
    check_and_install('scikit-build',"skbuild")
    #LLAMA DEPENTENCIES
    check_and_install('typing')
    check_and_install('diskcache')
    check_and_install('llama_cpp')
    check_and_install('timm',"timm","0.9.12",reboot=True)
    check_and_install('gitpython',"git")
    #check_and_install('sentencepiece')
    #check_and_install("accelerate")
    #check_and_install('transformers','transformers',"4.36.2")
    
    


    #git clone https://github.com/hzwer/Practical-RIFE.git
    from git import Repo
    if not os.path.exists(os.path.join(folder_paths.folder_names_and_paths["custom_nodes"][0][0],"ComfyUI-N-Nodes","libs","rifle")):
        Repo.clone_from("https://github.com/hzwer/Practical-RIFE.git", os.path.join(folder_paths.folder_names_and_paths["custom_nodes"][0][0],"ComfyUI-N-Nodes","libs","rifle"))

    if not os.path.exists(os.path.join(folder_paths.folder_names_and_paths["custom_nodes"][0][0],"ComfyUI-N-Nodes","libs","moondream_repo")):
        repo = Repo.clone_from("https://github.com/Nuked88/moondream.git", os.path.join(folder_paths.folder_names_and_paths["custom_nodes"][0][0],"ComfyUI-N-Nodes","libs","moondream_repo"))

        #commit_hash = "38af98596e59f2a6c25c6b52b2bd5a672dab4144"
        #repo.git.checkout(commit_hash)

    #if file moondream.py not exist 
    #if not os.path.exists(os.path.join(folder_paths.folder_names_and_paths["custom_nodes"][0][0],"ComfyUI-N-Nodes","libs","moondream_repo","moondream","moondream.py")):
    #    #delete moondream_repo and download repo again
    #    shutil.rmtree(os.path.join(folder_paths.folder_names_and_paths["custom_nodes"][0][0],"ComfyUI-N-Nodes","libs","moondream_repo"))
    #    repo = Repo.clone_from("https://github.com/Nuked88/moondream.git", os.path.join(folder_paths.folder_names_and_paths["custom_nodes"][0][0],"ComfyUI-N-Nodes","libs","moondream_repo"))

    #if train_log folder not exists
    if not os.path.exists(os.path.join(folder_paths.folder_names_and_paths["custom_nodes"][0][0],"ComfyUI-N-Nodes","libs","rifle","train_log")):
        downloader("https://github.com/Nuked88/DreamingAI/raw/main/RIFE_trained_model_v4.7.zip")
                   
    # code based on pysssss repo
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
            traceback.print_exc()


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
WEB_DIRECTORY = "./js"