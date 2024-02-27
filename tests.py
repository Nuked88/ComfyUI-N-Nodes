import asyncio
import os
import json
import shutil
import inspect

import requests
import subprocess
import platform
import importlib.util

import sys

config = None




def check_nvidia_gpu():
    try:
        # Utilizza torch per verificare la presenza di una GPU NVIDIA
        return torch.cuda.is_available() and 'NVIDIA' in torch.cuda.get_device_name(0)
    except Exception as e:
        print(f"Error while checking for NVIDIA GPU: {e}")
        return False

def get_cuda_version():
    try:
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda.replace(".","").strip()

            return "cu"+cuda_version
        else:
            return "No NVIDIA GPU available"
    except Exception as e:
        print(f"Error while checking CUDA version: {e}")
        return "Unable to determine CUDA version"

def check_avx2_support():
    import cpuinfo
    try:
        info = cpuinfo.get_cpu_info()
        return 'avx2' in info['flags']
    except Exception as e:
        print(f"Error while checking AVX2 support: {e}")
        return False

def get_python_version():
    if "3.9" in platform.python_version():
        return "39"
    elif "3.10" in platform.python_version():
        return "310"
    elif "3.11" in platform.python_version():
        return "311"
    else:
        return None


def get_os():
    return platform.system()

def get_os_bit():
    return platform.architecture()[0].replace("bit","")
import requests

def get_platform_tag(_os):
    #return the first tag in the list of tags
    try:
        import packaging.tags
        response = requests.get("https://api.github.com/repos/abetlen/llama-cpp-python/releases/latest")
        jresponse= response.json()
        
        return extract_platform_tag(jresponse,list(packaging.tags.sys_tags())[0],_os)

    except:
        return None





def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", package])

def install_llama():
    try:
        gpu = check_nvidia_gpu()
        avx2 = check_avx2_support()
        lcpVersion = get_last_llcpppy_version()
        python_version = get_python_version()
        _os = get_os()
        os_bit = get_os_bit()
        platform_tag = get_platform_tag(_os)
        print(f"Python version: {python_version}")
        print(f"OS: {_os}")
        print(f"OS bit: {os_bit}")
        print(f"Platform tag: {platform_tag}")
        if python_version == None:
            print("Unsupported Python version. Please use Python 3.9, 3.10 or 3.11.")
            return
        

        #python -m pip install llama-cpp-python --force-reinstall --no-deps --index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu117
        if avx2:
            avx="AVX2"
        else:
            avx="AVX"

        if gpu:
            cuda = get_cuda_version()
            print(f"--index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/{avx}/{cuda}")
        else:
            print(f"https://github.com/abetlen/llama-cpp-python/releases/download/v{lcpVersion}/llama_cpp_python-{lcpVersion}-{platform_tag}.whl")
    except Exception as e:
        print(f"Error while installing LLAMA: {e}")
# llama wheels https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels

def check_module(package):
    import importlib
    try:
        print("Detected: ", package)
        importlib.import_module(package)
        return True
    except ImportError:
        return False
import zipfile







import re
def extract_platform_tag(jresponse,tag,_os):
    if _os.lower()== "linux" or _os.lower()=="macosx":
        print(jresponse)
        for res in jresponse["assets"]:
            url = res["browser_download_url"]
            print(url)
            pattern = r'.*-((cp\d+-cp\d+)-(manylinux|macosx)_(\d+_\d+)_(x86_64|i686)\.whl)$'
            match = re.match(pattern, url)
            pattern_tag = r'((cp\d+-cp\d+)-(manylinux|macosx)_\d+_\d+_(x86_64|i686))$'
            match_tag = re.match(pattern_tag, tag)
            if match_tag:
                rl_platform_tag = f"{match_tag.group(2)}-{match_tag.group(3)}_**_{match_tag.group(4)}" 
                #print(rl_platform_tag)
                if match:
                    # Estrai il platform tag dal match
                    url_platform_tag = f"{match.group(2)}-{match.group(3)}_**_{match.group(5)}" 
                    final_platform_tag =f"{match.group(2)}-{match.group(3)}_{match.group(4)}_{match.group(5)}" 
                    if rl_platform_tag == url_platform_tag:
                        return final_platform_tag

        return None
    else:
        return tag
    
def get_last_llcpppy_version():
    try:
        import requests
    
        response = requests.get("https://api.github.com/repos/abetlen/llama-cpp-python/releases/latest")
        
        
        return response.json()["tag_name"].replace("v","")
    except:
        return "0.2.20"

   
install_llama()