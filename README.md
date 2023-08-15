# ComfyUI-N-Nodes
A suite of custom nodes for ComfyUI, for now i just put Integer, string and float variable nodes

# Installation

1. Clone the repository:
`git clone https://github.com/Nuked88/ComfyUI-N-Nodes.git`  
to your ComfyUI `custom_nodes` directory

   ComfyUI will then automatically load all custom scripts and nodes at the start.  


- For uninstallation:
  - Delete the cloned repo in `custom_nodes`


# Update
1. Navigate to the cloned repo e.g. `custom_nodes/ComfyUI-N-Nodes`
2. `git pull`

# Features

## Variables
Since the primitive node has limitations in links (for example at the time i'm writing you cannot link "start_at_step" and "steps" of another ksampler toghether), I decided to create these simple node-variables to bypass this limitation
The node-variables are:
- Integer
- Float
- String


## GPTLoaderSimple and GPTSampler

These custom nodes are designed to enhance the capabilities of the ConfyUI framework by enabling text generation using GPTQ GPT models. This README provides an overview of the two custom nodes and their usage within ConfyUI.

You can add in the _extra_model_paths.yaml_ the path where your model GPTQ are in this way (example):

`other_ui:
          base_path: I:\\temp3\\pygmaion\\text-generation-webui
          GPTcheckpoints: models/`
          
Otherwise it will create a GPTcheckpoints folder in the model folder of ComfyUI where you can place your .bin models.

### GPTLoaderSimple

The `GPTLoaderSimple` node is responsible for loading GPT model checkpoints and creating an instance of the Llama library for text generation. It provides an interface to configure GPU layers, the number of threads, and maximum context for text generation.

#### Input Fields

- `ckpt_name`: Select the GPT checkpoint name from the available options.
- `gpu_layers`: Specify the number of GPU layers to use (default: 27).
- `n_threads`: Specify the number of threads for text generation (default: 8).
- `max_ctx`: Specify the maximum context length for text generation (default: 2048).

#### Output

The node returns an instance of the Llama library (MODEL) and the path to the loaded checkpoint (STRING).

### GPTSampler

The `GPTSampler` node facilitates text generation using GPT models based on the input prompt and various generation parameters. It allows you to control aspects like temperature, top-p sampling, penalties, and more.

#### Input Fields

- `prompt`: Enter the input prompt for text generation.
- `model`: Choose the GPT model to use for text generation.
- `model_path`: Specify the path to the GPT model checkpoint.
- `max_tokens`: Set the maximum number of tokens in the generated text (default: 128).
- `temperature`: Set the temperature parameter for randomness (default: 0.7).
- `top_p`: Set the top-p probability for nucleus sampling (default: 0.5).
- `logprobs`: Specify the number of log probabilities to output (default: 0).
- `echo`: Enable or disable printing the input prompt alongside the generated text.
- `stop_token`: Specify the token at which text generation stops.
- `frequency_penalty`, `presence_penalty`, `repeat_penalty`: Control word generation penalties.
- `top_k`: Set the top-k tokens to consider during generation (default: 40).
- `tfs_z`: Set the temperature scaling factor for top frequent samples (default: 1.0).
- `print_output`: Enable or disable printing the generated text to the console.
- `cached`: Choose whether to use cached generation (default: NO).
- `prefix`, `suffix`: Specify text to prepend and append to the prompt.

#### Output

The node returns the generated text along with a UI-friendly representation.


## Dynamic Prompt


The `DynamicPrompt` node generates prompts by combining a fixed prompt with a random selection of tags from a variable prompt. This enables flexible and dynamic prompt generation for various use cases.

#### Input Fields

- `variable_prompt`: Enter the variable prompt for tag selection.
- `cached`: Choose whether to cache the generated prompt (default: NO).
- `number_of_random_tag`: Choose between "Fixed" and "Random" for the number of random tags to include.
- `fixed_number_of_random_tag`: If `number_of_random_tag` if "Fixed" Specify the number of random tags to include (default: 1).
- `fixed_prompt` (Optional): Enter the fixed prompt for generating the final prompt.

#### Output

The node returns the generated prompt, which is a combination of the fixed prompt and selected random tags.

## Example Usage

- Just fill the `variable_prompt` field with tag comma separated, the `fixed_prompt` is optional
