import folder_paths
import os
from pathlib import Path
import sys
import torch
from huggingface_hub import snapshot_download, hf_hub_download
sys.path.append(os.path.join(str(Path(__file__).parent.parent),"libs"))
import joytag_models
try:
    from moondream_repo.moondream.moondream import Moondream
    moondream_loaded = True
except Exception as e:
    moondream_loaded = False
    print(f"Moondream error: reinstall N-Suite dependencies with ComfyUI Manager or requirements.txt.\nTorch must be >= 2.1.0 (ERROR: {e})")
from PIL import Image
from transformers import CodeGenTokenizerFast as Tokenizer
#,AutoTokenizer, AutoModelForCausalLM
import numpy as np

models_base_path = os.path.join(folder_paths.models_dir, "GPTcheckpoints")
_choice = ["YES", "NO"]
_folders_whitelist = ["moondream","joytag"]#,"internlm"]


def env_or_def(env, default):
	if (env in os.environ):
		return os.environ[env]
	return default

def get_model_path(folder_list, model_name):
    for folder_path in folder_list:
        if folder_path.endswith(model_name):
            return folder_path
        
def get_model_list(models_base_path,supported_gpt_extensions):
    all_models = []
    try:
        for file in os.listdir(models_base_path):
            
            if os.path.isdir(os.path.join(models_base_path, file)):
                if  file in _folders_whitelist:
                    all_models.append(os.path.join(models_base_path, file))
            
            else:
                if file.endswith(tuple(supported_gpt_extensions)):
                    all_models.append(os.path.join(models_base_path, file))
    except:
        print(f"Path {models_base_path} not valid.")
    return all_models


def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def detect_device():
    """
    Detects the appropriate device to run on, and return the device and dtype.
    """
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    elif torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    else:
        return torch.device("cpu"), torch.float32


def load_joytag(ckpt_path,cpu=False):
    print("JOYTAG MODEL DETECTED")
    jt_config = os.path.join(models_base_path,"joytag","config.json")
    jt_readme= os.path.join(models_base_path,"joytag","README.md")
    jt_top_tags= os.path.join(models_base_path,"joytag","top_tags.txt")
    jt_model= os.path.join(models_base_path,"joytag","model.safetensors")


    if os.path.exists(jt_config)==False or os.path.exists(jt_readme)==False or os.path.exists(jt_top_tags)==False or os.path.exists(jt_model)==False:
        snapshot_download("fancyfeast/joytag",local_dir = os.path.join(models_base_path,"joytag"),local_dir_use_symlinks = False,)
    model = joytag_models.VisionModel.load_model(ckpt_path)
    model.eval()
    if cpu:
        return model.to('cpu')
    else:
        return model.to('cuda')

def run_joytag(images, prompt, max_tags, model_funct):
    with open(os.path.join(models_base_path,'joytag','top_tags.txt') , 'r') as f:
        top_tags = [line.strip() for line in f.readlines() if line.strip()]
        
    if images is None:
        raise ValueError("No image provided")
    top_tags_processed = []
    for image in images:
        _, scores = joytag_models.predict(image, model_funct, top_tags)
        top_tags_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:max_tags]
        # Extract the tags from the pairs
        top_tags_processed.append(', '.join([tag for tag, _ in top_tags_scores]))
    
    return top_tags_processed


def load_moondream(ckpt_path,cpu=False):

    dtype = torch.float32

    if cpu:
        device=torch.device("cpu")
    else:
        device = torch.device("cuda")
 

    config_json=os.path.join(os.path.join(models_base_path,"moondream"),'config.json')
    if os.path.exists(config_json)==False:
        hf_hub_download("vikhyatk/moondream1",
                                    local_dir=os.path.join(models_base_path,"moondream"),
                                    local_dir_use_symlinks=True,
                                    filename="config.json",
                                    endpoint='https://hf-mirror.com')
    
    model_safetensors=os.path.join(models_base_path,"moondream",'model.safetensors')
    if os.path.exists(model_safetensors)==False:
        hf_hub_download("vikhyatk/moondream1",
                                   local_dir=os.path.join(models_base_path,"moondream"),
                                   local_dir_use_symlinks=True,
                                   filename="model.safetensors",
                                   endpoint='https://hf-mirror.com')
    
    tokenizer_json=os.path.join(models_base_path,"moondream",'tokenizer.json')
    if os.path.exists(tokenizer_json)==False:
        hf_hub_download("vikhyatk/moondream1",
                                   local_dir=os.path.join(models_base_path,"moondream"),
                                   local_dir_use_symlinks=True,
                                   filename="tokenizer.json",
                                   endpoint='https://hf-mirror.com')
    
    if moondream_loaded:
        tokenizer = Tokenizer.from_pretrained(os.path.join(models_base_path,"moondream"))
        moondream = Moondream.from_pretrained(os.path.join(models_base_path,"moondream")).to(device=device, dtype=dtype)
        moondream.eval()
    else:
        tokenizer=None
        moondream=None





    return ([moondream, tokenizer])
    


def run_moondream(images, prompt, max_tags, model_funct):
    from PIL import Image
    moondream = model_funct[0]
    tokenizer = model_funct[1]
    list_descriptions = []
    for image in images:
        im=tensor2pil(image)

        image_embeds = moondream.encode_image(im)
        try:
            list_descriptions.append(moondream.answer_question(image_embeds, prompt,tokenizer))
        except ValueError:
            print("\n\n\n")
            raise ModuleNotFoundError("Moondream requires the dependency versions declared in N-Suite requirements.txt. Reinstall dependencies with ComfyUI Manager.")



    return list_descriptions

"""
def load_internlm(ckpt_path,cpu=False):
    
    
    
    local_dir=os.path.join(os.path.join(models_base_path,"internlm"))
    local_model_1 = os.path.join(local_dir,"pytorch_model-00001-of-00002.bin")
    local_model_2 = os.path.join(local_dir,"pytorch_model-00002-of-00002.bin")
        
    if os.path.exists(local_model_1) and os.path.exists(local_model_2):
        model_path = local_dir
    else:
        model_path = snapshot_download("internlm/internlm-xcomposer2-vl-7b", local_dir=local_dir, revision="f8e6ab8d7ff14dbd6b53335c93ff8377689040bf", local_dir_use_symlinks=False)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    if torch.cuda.is_available() and cpu == False:
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype="auto", 
                trust_remote_code=True,
                device_map="auto"
            ).eval()

    else:
        model = model.cpu().float().eval()
        
    model.tokenizer = tokenizer

    #device = device
    #dtype = dtype
    name = "internlm"
    #low_memory = low_memory
    
    return ([model, tokenizer])


def run_internlm(image, prompt, max_tags, model_funct):
    model = model_funct[0]
    tokenizer = model_funct[1]
    low_memory = True
    import tempfile
    image = Image.fromarray(np.clip(255. * image[0].cpu().numpy(),0,255).astype(np.uint8))
    #image = model.vis_processor(image)
    temp_dir = tempfile.mkdtemp()
    image_path = os.path.join(temp_dir,"input.jpg")
    image.save(image_path)
    #image = tensor2pil(image)
    if torch.cuda.is_available():
        with torch.cuda.amp.autocast(): 
            response, _ = model.chat(
                    query=prompt, 
                    image=image_path, 
                    tokenizer= tokenizer,
                    history=[], 
                    do_sample=True
                        )
        if low_memory:
            torch.cuda.empty_cache()
            print(f"Memory usage: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
            model.to("cpu", dtype=torch.float16)
            print(f"Memory usage: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    else:
        response, _ = model.chat(
                query=prompt,
                image=image, 
                tokenizer= tokenizer,
                history=[], 
                do_sample=True
            )

    return response
 """   

     


if not os.path.isdir(models_base_path):
        os.mkdir(models_base_path)

#create folder if it doesn't exist
if not os.path.isdir(os.path.join(folder_paths.models_dir, "GPTcheckpoints","joytag")):
        os.mkdir(os.path.join(folder_paths.models_dir, "GPTcheckpoints","joytag"))

if not os.path.isdir(os.path.join(folder_paths.models_dir, "GPTcheckpoints","moondream")):
        os.mkdir(os.path.join(folder_paths.models_dir, "GPTcheckpoints","moondream"))

"""#internlm
if not os.path.isdir(os.path.join(folder_paths.models_dir, "GPTcheckpoints","internlm")):
        os.mkdir(os.path.join(folder_paths.models_dir, "GPTcheckpoints","internlm"))
"""
#folder_paths.folder_names_and_paths["GPTcheckpoints"] += (os.listdir(models_base_path),)



MODEL_FUNCTIONS = {
'joytag': run_joytag,
'moondream': run_moondream
}
MODEL_LOAD_FUNCTIONS = {
'joytag': load_joytag,
'moondream': load_moondream
}





all_models = get_model_list(models_base_path, set())
all_models_names = [os.path.basename(model) for model in all_models]



class GPTLoaderSimple:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
              "ckpt_name": (all_models_names, ),
              "gpu_layers": ("INT", {"default": 27, "min": 0, "max": 100, "step": 1}),
              "n_threads": ("INT", {"default": 8, "min": 1, "max": 100, "step": 1}),
              "max_ctx": ("INT", {"default": 2048, "min": 300, "max": 100000, "step": 64}),
                             }}
    


    RETURN_TYPES = ("CUSTOM", )
    RETURN_NAMES = ("model",)
    FUNCTION = "load_gpt_checkpoint"
    DESCRIPTION = "Loads a Moondream or JoyTag image-captioning model. GGUF and LLaVA support was removed in version 1.2.0."

    CATEGORY = "N-Suite/loaders"
 
    def load_gpt_checkpoint(self, ckpt_name, gpu_layers, n_threads, max_ctx):
        ckpt_path = get_model_path(all_models,ckpt_name)
        if ckpt_name not in MODEL_LOAD_FUNCTIONS:
            raise ValueError(f"Unsupported model: {ckpt_name}")
        cpu = gpu_layers == 0
        llm = MODEL_LOAD_FUNCTIONS[ckpt_name](ckpt_path, cpu)

        return ([llm, ckpt_name, ckpt_path],)


class GPTSampler:
    
    """
    A custom node for text generation using GPT

    Attributes
    ----------
    max_tokens (`int`): Maximum number of tokens in the generated text.
    temperature (`float`): Temperature parameter for controlling randomness (0.2 to 1.0).
    top_p (`float`): Top-p probability for nucleus sampling.
    logprobs (`int`|`None`): Number of log probabilities to output alongside the generated text.
    echo (`bool`): Whether to print the input prompt alongside the generated text.
    stop (`str`|`List[str]`|`None`): Tokens at which to stop generation.
    frequency_penalty (`float`): Frequency penalty for word repetition.
    presence_penalty (`float`): Presence penalty for word diversity.
    repeat_penalty (`float`): Penalty for repeating a prompt's output.
    top_k (`int`): Top-k tokens to consider during generation.
    stream (`bool`): Whether to generate the text in a streaming fashion.
    tfs_z (`float`): Temperature scaling factor for top frequent samples.
    model (`str`): The GPT model to use for text generation.
    """
    def __init__(self):
        self.temp_prompt = ""
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                
                "model": ("CUSTOM", {"default": ""}),
                "max_tokens": ("INT", {"default": 2048}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.2, "max": 1.0}),
                "top_p": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0}),
                "logprobs": ("INT", {"default": 0}),
                "echo": (["enable", "disable"], {"default": "disable"}),
                "stop_token": ("STRING", {"default": "STOPTOKEN"}),
                "frequency_penalty": ("FLOAT", {"default": 0.0}),
                "presence_penalty": ("FLOAT", {"default": 0.0}),
                "repeat_penalty": ("FLOAT", {"default": 1.17647}),
                "top_k": ("INT", {"default": 40}),
                "tfs_z": ("FLOAT", {"default": 1.0}),
                "print_output": (["enable", "disable"], {"default": "disable"}),
                "cached": (_choice,{"default": "NO"} ),
                "prefix": ("STRING", {"default": "### Instruction: "}),
                "suffix": ("STRING", {"default": "### Response: "}),
                "max_tags": ("INT", {"default": 50}),
                
            },
             "optional": {
             "prompt": ("STRING",{"forceInput": True} ),
             "image": ("IMAGE",),
             }
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "generate_text"
    CATEGORY = "N-Suite/Sampling"

    

    def generate_text(self, max_tokens, temperature, top_p, logprobs, echo, stop_token, frequency_penalty, presence_penalty, repeat_penalty, top_k, tfs_z, model,print_output,cached,prefix,suffix,max_tags,image=None,prompt=None):
        model_funct = model[0]
        model_name = model[1]
        model_path = model[2]


        if cached == "NO":
            if model_name in MODEL_FUNCTIONS and os.path.isdir(model_path):
                cont = MODEL_FUNCTIONS[model_name](image, prompt, max_tags, model_funct)
            else:
                raise ValueError(f"Unsupported model: {model_name}")
        else:
            cont = self.temp_prompt 
        #remove fist 30 characters of cont
        try:
            if print_output == "enable":
                print(f"Input: {prompt}\nGenerated Text: {cont}")
            return {"ui": {"text": cont}, "result": (cont,)}

        except:
            if print_output == "enable":
                print(f"Input: {prompt}\nGenerated Text: ")
            return {"ui": {"text": " "}, "result": (" ",)}


NODE_CLASS_MAPPINGS = {
    "GPT Loader Simple [n-suite]": GPTLoaderSimple,
    "GPT Sampler [n-suite]": GPTSampler
}
# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "GPT Loader Simple [n-suite]": "GPT Loader Simple [🅝-🅢🅤🅘🅣🅔]",
    "GPT Sampler [n-suite]": "Image Caption Sampler [🅝-🅢🅤🅘🅣🅔]"

}
