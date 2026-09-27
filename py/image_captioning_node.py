import folder_paths
import os
from pathlib import Path
import sys
import torch
from huggingface_hub import snapshot_download
sys.path.append(os.path.join(str(Path(__file__).parent.parent),"libs"))
import joytag_models
from PIL import Image
from transformers import AutoModelForCausalLM, CodeGenTokenizerFast as Tokenizer, GenerationConfig, GenerationMixin, PreTrainedModel
from transformers.dynamic_module_utils import HF_MODULES_CACHE
from server import PromptServer
#,AutoTokenizer, AutoModelForCausalLM
import numpy as np

models_base_path = os.path.join(folder_paths.models_dir, "GPTcheckpoints")
MOONDREAM_REVISION = "f6e9da68e8f1b78b8f3ee10905d56826db7a5802"
JOYTAG_REVISION = "6b7f16331a6ccf0fdce37d5a9564715f6e772b22"
MODEL_DOWNLOADS = {
    "moondream": ("Moondream", 3.72),
    "joytag": ("JoyTag", 0.37),
}
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
        snapshot_download(
            "fancyfeast/joytag",
            revision=JOYTAG_REVISION,
            local_dir=os.path.join(models_base_path, "joytag"),
            allow_patterns=["README.md", "config.json", "model.safetensors", "top_tags.txt"],
        )
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
 

    model_dir = os.path.join(models_base_path, "moondream")
    snapshot_download(
        "vikhyatk/moondream1",
        revision=MOONDREAM_REVISION,
        local_dir=model_dir,
        allow_patterns=[
            "config.json", "configuration_moondream.py", "moondream.py",
            "modeling_phi.py", "text_model.py", "vision_encoder.py",
            "model.safetensors", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "added_tokens.json", "merges.txt", "vocab.json",
        ],
    )
    patch_moondream_model_code(model_dir)
    tokenizer = Tokenizer.from_pretrained(model_dir)
    moondream = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
    enable_moondream_generation(moondream)
    moondream = moondream.to(device=device, dtype=dtype)
    moondream.eval()
    return [moondream, tokenizer]


def enable_moondream_generation(moondream):
    """Restore generation for Moondream1's legacy Phi model on Transformers 4.50+."""
    text_model = moondream.text_model
    if getattr(text_model, "_n_suite_generation_compat", False):
        return

    model_class = type(text_model)
    original_prepare = model_class.prepare_inputs_for_generation

    def prepare_inputs_for_generation(
        self, input_ids=None, inputs_embeds=None, past_key_values=None,
        attention_mask=None, **kwargs,
    ):
        prepared = original_prepare(
            self, input_ids=input_ids, inputs_embeds=inputs_embeds,
            past_key_values=past_key_values, attention_mask=attention_mask,
            **kwargs,
        )
        # Moondream supplies image embeddings without padding. The newer
        # generation API otherwise builds a mask one token too long.
        prepared["attention_mask"] = None
        return prepared

    bases = (model_class,) if isinstance(text_model, GenerationMixin) else (model_class, GenerationMixin)
    text_model.__class__ = type(
        "GeneratingPhiForCausalLM", bases,
        {"prepare_inputs_for_generation": prepare_inputs_for_generation},
    )
    text_model._n_suite_generation_compat = True
    if text_model.generation_config is None:
        text_model.generation_config = GenerationConfig.from_model_config(text_model.config)


def patch_moondream_model_code(model_dir):
    """Add GenerationMixin to the pinned Phi source before Transformers imports it."""
    original_import = "from transformers import PretrainedConfig, PreTrainedModel"
    parent_base = "class PhiPreTrainedModel(PreTrainedModel):"
    previous_patch = "class PhiPreTrainedModel(PreTrainedModel, GenerationMixin):"
    model_bases = (
        ("class PhiModel(PhiPreTrainedModel):", "class PhiModel(PhiPreTrainedModel, GenerationMixin):"),
        ("class PhiForCausalLM(PhiPreTrainedModel):", "class PhiForCausalLM(PhiPreTrainedModel, GenerationMixin):"),
    )
    needs_patch = not issubclass(PreTrainedModel, GenerationMixin)

    def patch_source(path):
        source = path.read_text()
        if original_import not in source:
            raise RuntimeError("Unexpected Moondream1 Phi source; cannot apply the generation compatibility patch")
        if previous_patch in source:
            source = source.replace(previous_patch, parent_base, 1)
        for original, patched in model_bases:
            if original not in source and patched not in source:
                raise RuntimeError("Unexpected Moondream1 Phi source; cannot apply the generation compatibility patch")
            if needs_patch:
                source = source.replace(original, patched, 1)
            else:
                source = source.replace(patched, original, 1)
        patched_import = original_import + ", GenerationMixin"
        if needs_patch:
            source = source.replace(original_import, patched_import, 1) if patched_import not in source else source
        else:
            source = source.replace(patched_import, original_import, 1)
        if source != path.read_text():
            path.write_text(source)

    patch_source(Path(model_dir) / "modeling_phi.py")
    cached_source = Path(HF_MODULES_CACHE) / "transformers_modules" / Path(model_dir).name / "modeling_phi.py"
    if cached_source.is_file():
        patch_source(cached_source)
    


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

     


os.makedirs(models_base_path, exist_ok=True)

#create folder if it doesn't exist
os.makedirs(os.path.join(models_base_path, "joytag"), exist_ok=True)

os.makedirs(os.path.join(models_base_path, "moondream"), exist_ok=True)

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
                             },
                "hidden": {"unique_id": "UNIQUE_ID"}}
    


    RETURN_TYPES = ("CUSTOM", )
    RETURN_NAMES = ("model",)
    FUNCTION = "load_gpt_checkpoint"
    DESCRIPTION = "Loads Moondream (~3.72 GB) or JoyTag (~0.37 GB). The first use downloads the selected model; watch the ComfyUI console for progress."

    CATEGORY = "N-Suite/loaders"
 
    def load_gpt_checkpoint(self, ckpt_name, gpu_layers, n_threads, max_ctx, unique_id=None):
        ckpt_path = get_model_path(all_models,ckpt_name)
        if ckpt_name not in MODEL_LOAD_FUNCTIONS:
            raise ValueError(f"Unsupported model: {ckpt_name}")
        model_dir = os.path.join(models_base_path, ckpt_name)
        if not os.path.isfile(os.path.join(model_dir, "model.safetensors")):
            model_name, size_gb = MODEL_DOWNLOADS[ckpt_name]
            message = (f"{model_name}: downloading approximately {size_gb:.2f} GB on first use. "
                       "This may take a while; watch the ComfyUI console for progress.")
            print(f"[N-Suite] {message}", flush=True)
            if PromptServer.instance is not None:
                PromptServer.instance.send_sync(
                    "n-suite-model-download",
                    {"node_id": unique_id, "model": model_name, "size_gb": size_gb, "message": message},
                )
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
