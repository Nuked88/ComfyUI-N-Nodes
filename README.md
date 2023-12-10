[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/C0C0AJECJ)

# ComfyUI-N-Nodes
A suite of custom nodes for ComfyUI that includes Integer, string and float variable nodes, GPT nodes and video nodes.

> [!IMPORTANT]  
> These nodes were tested primarily in Windows in the default environment provided by ComfyUI and in the environment created by the [notebook](https://github.com/comfyanonymous/ComfyUI/blob/master/notebooks/comfyui_colab.ipynb) for paperspace specifically with the cyberes/gradient-base-py3.10:latest docker image.
**Any other environment has not been tested.**


# Installation

1. Clone the repository:
`git clone https://github.com/Nuked88/ComfyUI-N-Nodes.git`  
to your ComfyUI `custom_nodes` directory

2. ~~IMPORTANT: If you want the GPT nodes on GPU you'll need to run **install_dependency bat files**. 
There are 2 versions: ***install_dependency_ggml_models.bat*** for the old ggmlv3 models and ***install_dependency_gguf_models.bat*** for all the new models (GGUF).
YOU CAN ONLY USE ONE OF THEM AT A TIME!
Since _llama-cpp-python_ needs to be compiled from source code to enable it to use the GPU, you will first need to have [CUDA](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64)  and visual studio 2019 or 2022  (in the case of my bat) installed to compile it. For details and the full guide you can go [HERE](https://github.com/abetlen/llama-cpp-python).~~ 

ComfyUI will then automatically load all custom scripts and nodes at the start.  

> [!NOTE]  
> The llama-cpp-python installation will be done automatically by the script. If you have an NVIDIA GPU NO MORE CUDA BUILD IS NECESSARY thanks to [jllllll](https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/) repo. I've also dropped the support to GGMLv3 models since all notable models should have switched to the latest version of GGUF by now.



- For uninstallation:
  - Delete the `ComfyUI-N-Nodes` folder in `custom_nodes`






# Update
1. Navigate to the cloned repo e.g. `custom_nodes/ComfyUI-N-Nodes`
2. `git pull`

# Features

## 📽️ Video Nodes 📽️

### LoadVideo
The LoadVideo node allows loading a video file and extracting frames from it.

#### Input Fields
- `video`: Select the video file to load.
- `framerate`: Choose whether to keep the original framerate or reduce to half or quarter speed.
- `resize_by`: Select how to resize frames - 'none', 'height', or 'width'.
- `size`: Target size if resizing by height or width.
- `images_limit`: Limit number of frames to extract.
- `batch_size`: Batch size for encoding frames.
- `starting_frame`: Select which frame to start from.
- `autoplay`: Select whether to autoplay the video.

#### Output

- `IMAGES`: Extracted frame images as PyTorch tensors.
- `LATENT`: Empty latent vectors.
- `METADATA`: Video metadata - FPS and number of frames.
- `WIDTH:` Frame width.
- `HEIGHT`: Frame height.

The node extracts frames from the input video at the specified framerate. It resizes frames if chosen and returns them as batches of PyTorch image tensors along with latent vectors, metadata, and frame dimensions.

### SaveVideo
The SaveVideo node takes in extracted frames and saves them back as a video file.

#### Input Fields
- `images`: Frame images as tensors.
- `METADATA`: Metadata from LoadVideo node.
- `SaveVideo`: Toggle saving output video file.
- `SaveFrames`: Toggle saving frames to a folder.
- `CompressionLevel`: PNG compression level for saving frames.
#### Output
Saves output video file and/or extracted frames.

The node takes extracted frames and metadata and can save them as a new video file and/or individual frame images. Video compression and frame PNG compression can be configured.
NOTE: If you are using **LoadVideo** as source of the frames, the audio of the original file will be maintained but only in case **images_limit** and **starting_frame** are equal to Zero.

### LoadFramesFromFolder
The LoadFramesFromFolder node allows loading image frames from a folder and returning them as a batch.

#### Input Fields
- `folder`: Path to the folder containing the frame images.Must be png format, named with a number (eg. 1.png or even 0001.png).The images will be loaded sequentially.
- `fps`: Frames per second to assign to the loaded frames.

#### Output
- `IMAGES`: Batch of loaded frame images as PyTorch tensors.
- `METADATA`: Metadata containing the set FPS value.

The node loads all image files from the specified folder, converts them to PyTorch tensors, and returns them as a batched tensor along with simple metadata containing the set FPS value.

This allows easily loading a set of frames that were extracted and saved previously, for example, to reload and process them again. By setting the FPS value, the frames can be properly interpreted as a video sequence.


### FrameInterpolator

The FrameInterpolator node allows interpolating between extracted video frames to increase the frame rate and smooth motion.

#### Input Fields

- `images`: Extracted frame images as tensors.
- `METADATA`: Metadata from video - FPS and number of frames.
- `multiplier`: Factor by which to increase frame rate. 

#### Output  

- `IMAGES`: Interpolated frames as image tensors.
- `METADATA`: Updated metadata with new frame rate.

The node takes extracted frames and metadata as input. It uses an interpolation model (RIFE) to generate additional in-between frames at a higher frame rate. 

The original frame rate in the metadata is multiplied by the `multiplier` value to get the new interpolated frame rate.

The interpolated frames are returned as a batch of image tensors, along with updated metadata containing the new frame rate.

This allows increasing the frame rate of an existing video to achieve smoother motion and slower playback. The interpolation model creates new realistic frames to fill in the gaps rather than just duplicating existing frames.

The original code has been taken from [HERE](https://github.com/hzwer/Practical-RIFE/tree/main)

## Variables
Since the primitive node has limitations in links (for example at the time i'm writing you cannot link "start_at_step" and "steps" of another ksampler toghether), I decided to create these simple node-variables to bypass this limitation
The node-variables are:
- Integer
- Float
- String


## 🤖 GPTLoaderSimple and GPTSampler 🤖

These custom nodes are designed to enhance the capabilities of the ConfyUI framework by enabling text generation using GGUF GPT models. This README provides an overview of the two custom nodes and their usage within ConfyUI.

You can add in the _extra_model_paths.yaml_ the path where your model GGUF are in this way (example):

`other_ui:
          base_path: I:\\text-generation-webui
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

#### Example Usage

- Just fill the `variable_prompt` field with tag comma separated, the `fixed_prompt` is optional




## Troubleshooting

- ~~ **SaveVideo - Preview not working**: is related to a conflict with animateDiff, i've already opened a [PR](https://github.com/ArtVentureX/comfyui-animatediff/pull/64) to solve this issue. Meanwhile you can download my patched version from [here](https://github.com/Nuked88/comfyui-animatediff)~~ pull has been merged so this problem should be fixed now!

## Contributing

Feel free to contribute to this project by reporting issues or suggesting improvements. Open an issue or submit a pull request on the GitHub repository.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.