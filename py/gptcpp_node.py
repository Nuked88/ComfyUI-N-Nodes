import folder_paths
import os
from llama_cpp import Llama
import copy
from typing_extensions import TypedDict, Literal
from typing import List, Optional


_choice = ["YES", "NO"]
def env_or_def(env, default):
	if (env in os.environ):
		return os.environ[env]
	return default


supported_gpt_extensions = set([ '.bin','.gguf'])



try:
    folder_paths.folder_names_and_paths["GPTcheckpoints"] = (folder_paths.folder_names_and_paths["GPTcheckpoints"][0], supported_gpt_extensions)
except:
    # check if GPTcheckpoints exists otherwise create
    if not os.path.isdir(os.path.join(folder_paths.models_dir, "GPTcheckpoints")):
        os.mkdir(os.path.join(folder_paths.models_dir, "GPTcheckpoints"))
        
    folder_paths.folder_names_and_paths["GPTcheckpoints"] = ([os.path.join(folder_paths.models_dir, "GPTcheckpoints")], supported_gpt_extensions)
    


class GPTLoaderSimple:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
              "ckpt_name": (folder_paths.get_filename_list("GPTcheckpoints"), ),
              "gpu_layers": ("INT", {"default": 27, "min": 0, "max": 100, "step": 1}),
              "n_threads": ("INT", {"default": 8, "min": 1, "max": 100, "step": 1}),
              "max_ctx": ("INT", {"default": 2048, "min": 300, "max": 100000, "step": 64}),
                             }}
    


    RETURN_TYPES = ("CUSTOM","STRING")
    FUNCTION = "load_gpt_checkpoint"

    CATEGORY = "loaders"
    print()
    def load_gpt_checkpoint(self, ckpt_name, gpu_layers,n_threads,max_ctx):
        ckpt_path = folder_paths.get_full_path("GPTcheckpoints", ckpt_name)
        llm = Llama(model_path=ckpt_path,n_gpu_layers=gpu_layers,verbose=False,n_threads=n_threads, n_ctx=4000, )

        return llm, ckpt_path


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
                "prompt": ("STRING",{"forceInput": True} ),
                "model": ("CUSTOM", {"default": ""}),
                "model_path": ("STRING", {"default": "","forceInput": True}),
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
                
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text"
    CATEGORY = "sampling"

    def generate_text(self,prompt, max_tokens, temperature, top_p, logprobs, echo, stop_token, frequency_penalty, presence_penalty, repeat_penalty, top_k, tfs_z, model,model_path,print_output,cached,prefix,suffix):
        
        
        if cached == "NO":
            # Call your GPT generation function here using the provided parameters
            composed_prompt = f"{prefix} {prompt} {suffix}"
            cont =""
            stream = model( max_tokens=max_tokens, stop=[stop_token], stream=False,frequency_penalty=frequency_penalty,presence_penalty=presence_penalty ,repeat_penalty=repeat_penalty,temperature=temperature,top_k=top_k,top_p=top_p,model=model_path,prompt=composed_prompt)
            print(len(stream))
            print(stream)
            cont= stream["choices"][0]["text"]
            self.temp_prompt  = cont
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
    "GPT Loader Simple": GPTLoaderSimple,
    "GPTSampler": GPTSampler
}
# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "GPT Loader Simple": "GPT Loader Simple",
    "GPTSampler": "GPT Text Sampler"

}



