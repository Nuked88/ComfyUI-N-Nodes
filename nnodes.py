import asyncio
import os
import json
import shutil
import inspect
import aiohttp
from server import PromptServer
from tqdm import tqdm
import requests
import subprocess
import platform
import importlib.util
import torch
import folder_paths
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

def get_platform_tag():
    #return the first tag in the list of tags
    try:
        import packaging.tags
        return list(packaging.tags.sys_tags())[0]
    except:
        return None
def get_last_llcpppy_version():
    try:
        import requests
    
        response = requests.get("https://api.github.com/repos/abetlen/llama-cpp-python/releases/latest")
        
        
        return response.json()["tag_name"].replace("v","")
    except:
        return "0.2.20"

from packaging import version

def check_and_install(package, import_name="", desired_version=None):
    if import_name == "":
        import_name = package
    try:
        library_module = importlib.import_module(import_name)
        current_version  = getattr(library_module, '__version__', None)
        if current_version :
            if current_version:
                print(f"Current version of {import_name}: {current_version}")
            if desired_version:
                if version.parse(current_version) < version.parse(desired_version):
                    print(f"Updating {import_name} to version {desired_version}...")
                    install_package(f"{package}=={desired_version}")
                    print(f"{import_name} updated successfully to version {desired_version}")
                else:
                    print(f"{import_name} is already up-to-date with version {current_version}")

        else:
            print(f"Version of {import_name}: Version information not found")

        
    except ImportError:
        print(f"Installing {import_name}...")
        if package == "llama_cpp":
            install_llama()
        else:
            install_package(package)

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", package])

def install_llama():
    try:
        gpu = check_nvidia_gpu()
        avx2 = check_avx2_support()
        lcpVersion = get_last_llcpppy_version()
        python_version = get_python_version()
        os = get_os()
        os_bit = get_os_bit()
        platform_tag = get_platform_tag()
        print(f"Python version: {python_version}")
        print(f"OS: {os}")
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
            subprocess.check_call([sys.executable, "-m", "pip", "install", "llama-cpp-python", "--no-cache-dir", "--force-reinstall", "--no-deps" , f"--index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/{avx}/{cuda}"])
        else:
            subprocess.check_call([sys.executable, "-m", "pip", "install", f"https://github.com/abetlen/llama-cpp-python/releases/download/v{lcpVersion}/llama_cpp_python-{lcpVersion}-{platform_tag}.whl"])
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
            if chunk: 
                f.write(chunk) 

    zip_file = zipfile.ZipFile(temp_file) 
    target_dir = os.path.join(folder_paths.folder_names_and_paths["custom_nodes"][0][0],"ComfyUI-N-Nodes","libs","rifle") # Cartella dove estrarre lo zip

    zip_file.extractall(target_dir) 





def is_logging_enabled():
    config = get_extension_config()
    if "logging" not in config:
        return False
    return config["logging"]


def log(message, type=None, always=False, name=None):
    if not always and not is_logging_enabled():
        return

    if type is not None:
        message = f"[{type}] {message}"

    if name is None:
        name = get_extension_config()["name"]

    print(f"(nnodes:{name}) {message}")


def get_ext_dir(subpath=None, mkdir=False):
    dir = os.path.dirname(__file__)
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    dir = os.path.abspath(dir)

    if mkdir and not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def get_comfy_dir(subpath=None, mkdir=False):
    dir = os.path.dirname(inspect.getfile(PromptServer))
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    dir = os.path.abspath(dir)

    if mkdir and not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def get_web_ext_dir():
    config = get_extension_config()
    name = config["name"]
    dir = get_comfy_dir("web/extensions/comfyui-n-nodes")
    if not os.path.exists(dir):
        os.makedirs(dir)
    dir = os.path.join(dir, name)
    return dir

def get_extension_config(reload=False):
    global config
    if reload == False and config is not None:
        return config

    config_path = get_ext_dir("nnodes.json")
    if not os.path.exists(config_path):
        log("Missing nnodes.json, this extension may not work correctly. Please reinstall the extension.",
            type="ERROR", always=True, name="???")
        print(f"Extension path: {get_ext_dir()}")
        return {"name": "Unknown", "version": -1}
    with open(config_path, "r") as f:
        config = json.loads(f.read())
    return config



def link_js(src, dst):
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    if os.name == "nt":
        try:
            import _winapi
            _winapi.CreateJunction(src, dst)
            return True
        except:
            pass
    try:
        os.symlink(src, dst)
        return True
    except:
        import logging
        logging.exception('')
        return False

def is_junction(path):
    if os.name != "nt":
        return False
    try:
        return bool(os.readlink(path))
    except OSError:
        return False

def install_js():
    src_dir = get_ext_dir("js")
    if not os.path.exists(src_dir):
        log("No JS")
        return

    dst_dir = get_web_ext_dir()

    if os.path.exists(dst_dir):
        if os.path.islink(dst_dir) or is_junction(dst_dir):
            log("JS already linked")
            return
    elif link_js(src_dir, dst_dir):
        log("JS linked")
        return

    log("Copying JS files")
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)


def init(check_imports=None):
    log("Init")

    if check_imports is not None:
        import importlib.util
        for imp in check_imports:
            spec = importlib.util.find_spec(imp)
            if spec is None:
                log(f"{imp} is required, please check requirements are installed.",
                    type="ERROR", always=True)
                return False

    #install_js()
    return True


def get_async_loop():
    loop = None
    try:
        loop = asyncio.get_event_loop()
    except:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def get_http_session():
    loop = get_async_loop()
    return aiohttp.ClientSession(loop=loop)


async def download(url, stream, update_callback=None, session=None):
    close_session = False
    if session is None:
        close_session = True
        session = get_http_session()
    try:
        async with session.get(url) as response:
            size = int(response.headers.get('content-length', 0)) or None

            with tqdm(
                unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1], total=size,
            ) as progressbar:
                perc = 0
                async for chunk in response.content.iter_chunked(2048):
                    stream.write(chunk)
                    progressbar.update(len(chunk))
                    if update_callback is not None and progressbar.total is not None and progressbar.total != 0:
                        last = perc
                        perc = round(progressbar.n / progressbar.total, 2)
                        if perc != last:
                            last = perc
                            await update_callback(perc)
    finally:
        if close_session and session is not None:
            await session.close()


async def download_to_file(url, destination, update_callback=None, is_ext_subpath=True, session=None):
    if is_ext_subpath:
        destination = get_ext_dir(destination)
    with open(destination, mode='wb') as f:
        download(url, f, update_callback, session)




def is_inside_dir(root_dir, check_path):
    root_dir = os.path.abspath(root_dir)
    if not os.path.isabs(check_path):
        check_path = os.path.abspath(os.path.join(root_dir, check_path))
    return os.path.commonpath([check_path, root_dir]) == root_dir


def get_child_dir(root_dir, child_path, throw_if_outside=True):
    child_path = os.path.abspath(os.path.join(root_dir, child_path))
    if is_inside_dir(root_dir, child_path):
        return child_path
    if throw_if_outside:
        raise NotADirectoryError(
            "Saving outside the target folder is not allowed.")
    return None
