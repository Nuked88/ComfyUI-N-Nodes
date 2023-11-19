# code based on pysssss repo
import glob
import os
from .nnodes import init, get_ext_dir,check_and_install,downloader
import folder_paths



py = get_ext_dir("py")
files = glob.glob("*.py", root_dir=py, recursive=False)
check_and_install('py-cpuinfo',"cpuinfo")
check_and_install('gitpython','git')
check_and_install('moviepy')
check_and_install("opencv-python","cv2") 
check_and_install('zipfile')
check_and_install('scikit-build',"skbuild")
#LLAMA DEPENTENCIES
check_and_install('typing')
check_and_install('diskcache')
check_and_install('llama_cpp')

#git clone https://github.com/hzwer/Practical-RIFE.git
from git import Repo
if not os.path.exists(os.path.join(folder_paths.folder_names_and_paths["custom_nodes"][0][0],"ComfyUI-N-Nodes","libs","rifle")):
    Repo.clone_from("https://github.com/hzwer/Practical-RIFE.git", os.path.join(folder_paths.folder_names_and_paths["custom_nodes"][0][0],"ComfyUI-N-Nodes","libs","rifle"))
#if train_log folder not exists
if not os.path.exists(os.path.join(folder_paths.folder_names_and_paths["custom_nodes"][0][0],"ComfyUI-N-Nodes","libs","rifle","train_log")):
    downloader("https://github.com/Nuked88/DreamingAI/raw/main/RIFE_trained_model_v4.7.zip")
                   

