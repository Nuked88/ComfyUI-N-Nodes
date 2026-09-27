[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/C0C0AJECJ)

# ComfyUI-N-Suite
A suite of custom nodes for ComfyUI that includes integer, string and float variable nodes, image-captioning nodes and video nodes.

> [!IMPORTANT]  
> These nodes were tested primarily in Windows in the default environment provided by ComfyUI and in the environment created by the [notebook](https://github.com/comfyanonymous/ComfyUI/blob/master/notebooks/comfyui_colab.ipynb) for paperspace specifically with the cyberes/gradient-base-py3.10:latest docker image.
**Any other environment has not been tested.**


# Installation

1. Clone the repository:
`git clone https://github.com/Nuked88/ComfyUI-N-Nodes.git`  
to your ComfyUI `custom_nodes` directory

2. Install it through **ComfyUI Manager** (recommended), which installs the dependencies declared by the project, or run `python -m pip install -r requirements.txt` in ComfyUI's Python environment after a manual clone.
3. Restart ComfyUI.

ComfyUI automatically loads all custom scripts and nodes at startup.

> [!IMPORTANT]
> **Breaking change in 1.2.0:** `llama-cpp-python` integration has been removed because its platform-specific installation was the main source of installation and startup failures. GGUF text-generation models and LLaVA nodes are therefore no longer available in N-Suite. Existing workflows using `Llava Clip Loader` must remove that node; `GPT Loader Simple` and `GPT Sampler` now support only Moondream and JoyTag. N-Suite no longer detects, downloads, or installs `llama-cpp-python`.

> [!WARNING]
> **The `Llava Clip Loader` node and the GGUF text-generation path of `GPT Loader Simple` / `GPT Sampler` have been removed.** To keep using those legacy nodes, install the last revision that contains them with `git checkout ae7cc84`. That revision is unsupported and retains the `llama-cpp-python` installation problems; use it in a separate ComfyUI installation or Python environment.

> [!NOTE]  
> Since 14/02/2024, the node has undergone a massive rewrite, which also led to the change of all node names in order to avoid any conflicts with other extensions in the future (or at least I hope so). Consequently, the old workflows are no longer compatible and will require manual replacement of each node.
> To avoid this, I have created a tool that allows for automatic replacement.
> On Windows, simply drag any *.json workflow onto the migrate.bat file located in (custom_nodes/ComfyUI-N-Nodes), and another workflow with the suffix _migrated will be created in the same folder as the current workflow.
> On Linux, you can use the script in the following way: python libs/migrate.py path/to/original/workflow/.
> For security reasons, the original workflow will not be deleted."
> For install the last version of this repository before this changes from the Comfyui-N-Suite execute **git checkout 29b2e43baba81ee556b2930b0ca0a9c978c47083**


- For uninstallation:
  - Delete the `ComfyUI-N-Nodes` folder in `custom_nodes`
  - Delete the `comfyui-n-nodes` folder in  `ComfyUI\web\extensions`
  - Delete the `n-styles.csv` and `n-styles.csv.backup` file in `ComfyUI\styles`
  - Delete the `GPTcheckpoints` folder in `ComfyUI\models`





# Update
1. Navigate to the cloned repo e.g. `custom_nodes/ComfyUI-N-Nodes`
2. `git pull`

# Features

## 📽️ Video Nodes 📽️

### LoadVideo

![alt text](./img/image-13.png)

The LoadVideoAdvanced node allows loading a video file and extracting frames from it.
The name has been changed from `LoadVideo` to `LoadVideoAdvanced` in order to avoid conflicts with the `LoadVideo` animatediff node.


#### Input Fields
- `video`: Select the video file to load.
- `framerate`: Choose whether to keep the original framerate or reduce to half or quarter speed.
- `resize_by`: Select how to resize frames - 'none', 'height', or 'width'.
- `size`: Target size if resizing by height or width.
- `images_limit`: Limit number of frames to extract.
- `batch_size`: Batch size for encoding frames.
- `starting_frame`: Select which frame to start from.
- `autoplay`: Select whether to autoplay the video.
- `use_ram`: Use RAM instead of disk for decompressing video frames.  

#### Output

- `IMAGES`: Extracted frame images as PyTorch tensors.
- `LATENT`: Empty latent vectors.
- `METADATA`: Video metadata - FPS and number of frames.
- `WIDTH:` Frame width.
- `HEIGHT`: Frame height.
- `META_FPS`: Frame rate.
- `META_N_FRAMES`: Number of frames.


The node extracts frames from the input video at the specified framerate. It resizes frames if chosen and returns them as batches of PyTorch image tensors along with latent vectors, metadata, and frame dimensions.

### SaveVideo
The SaveVideo node takes in extracted frames and saves them back as a video file.
![alt text](./img/image-3.png)

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
![alt text](./img/image.png)

The LoadFramesFromFolder node allows loading image frames from a folder and returning them as a batch.


#### Input Fields
- `folder`: Path to the folder containing the frame images.Must be png format, named with a number (eg. 1.png or even 0001.png).The images will be loaded sequentially.
- `fps`: Frames per second to assign to the loaded frames.

#### Output
- `IMAGES`: Batch of loaded frame images as PyTorch tensors.
- `METADATA`: Metadata containing the set FPS value.
- `MAX_WIDTH`: Maximum frame width.
- `MAX_HEIGHT`: Maximum frame height.
- `FRAME COUNT`: Number of frames in the folder.
- `PATH`: Path to the folder containing the frame images.
- `IMAGE LIST`: List of frame images in the folder (not a real list just a string divided by \n).

The node loads all image files from the specified folder, converts them to PyTorch tensors, and returns them as a batched tensor along with simple metadata containing the set FPS value.

This allows easily loading a set of frames that were extracted and saved previously, for example, to reload and process them again. By setting the FPS value, the frames can be properly interpreted as a video sequence.

### SetMetadataForSaveVideo

![alt text](./img/image-1.png)

The SetMetadataForSaveVideo node allows setting metadata for the SaveVideo node.

### FrameInterpolator

![alt text](./img/image-4.png)

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


## 🤖 Image captioning: GPTLoaderSimple and GPTSampler 🤖

The legacy node identifiers are retained so existing Moondream and JoyTag workflows continue to load, but these nodes are now limited to image captioning:

- **Moondream:** its `config.json`, `model.safetensors`, and `tokenizer.json` files are downloaded automatically from the `vikhyatk/moondream1` Hugging Face repository the first time Moondream is loaded.
- **JoyTag:** its model snapshot is downloaded automatically from the `fancyfeast/joytag` Hugging Face repository the first time JoyTag is loaded.
- **LLaVA:** was never downloaded automatically. Its GGUF model and projector had to be installed manually; support has now been removed together with `llama-cpp-python`.

Downloads happen on first model use, not merely when ComfyUI starts. An internet connection and sufficient disk space are required for that initial load. Models are stored under `ComfyUI/models/GPTcheckpoints/moondream` and `ComfyUI/models/GPTcheckpoints/joytag`.

### GPTLoaderSimple

`GPTLoaderSimple` loads either Moondream or JoyTag. The `gpu_layers` field is retained for workflow compatibility: set it to `0` for CPU, or to a value greater than zero for GPU. The old `n_threads` and `max_ctx` fields are also retained so saved workflows continue to deserialize, but they do not affect these image-captioning models.

### GPTSampler

Connect an image and, for Moondream, a question or instruction in `prompt`. JoyTag uses `max_tags` to limit the number of returned tags. The advanced text-generation controls remain visible for workflow compatibility but no longer apply to GGUF text generation.

### Why LLaVA was removed

The old LLaVA implementation was not independent: both `Llama` and `Llava15ChatHandler` came from `llama-cpp-python`. Automatically downloading the LLaVA GGUF files would therefore not solve the native-library installation failure. Restoring LLaVA without restoring `llama-cpp-python` requires a new backend (for example Transformers) and a migration path for existing workflows. Such a replacement should download model weights only when the user executes the loader, show the repository and approximate download size, and use the normal Hugging Face cache rather than downloading during N-Suite import.

## Installation and maintenance status

N-Suite is published through the Comfy Registry and is installable by current versions of ComfyUI Manager. Dependencies are declared in both `pyproject.toml` and `requirements.txt`; the extension no longer runs `pip install` itself during import. A manual Git installation remains supported, but the user must install `requirements.txt` into the exact Python environment used by ComfyUI.

### Testing this release through Manager Nightly

ComfyUI Manager's **nightly** entry is the current Git revision from the repository's default branch; it is not a separate version published to the Comfy Registry. Consequently, a pull request cannot be selected as nightly while it is still unmerged. The test flow for this release is:

1. merge the pull request into `main`;
2. open ComfyUI Manager and select N-Suite's `nightly` version;
3. restart ComfyUI and validate the nodes and existing workflows;
4. only after validation, manually run the `Publish to Comfy registry` GitHub Actions workflow to publish version `1.2.0` as stable.

The Registry workflow is deliberately manual: merging a change to `pyproject.toml` no longer publishes an untested stable version automatically. Before the merge, testers can still clone the pull-request branch manually under `custom_nodes`, but Manager will not label that branch as `nightly`.

Practical-RIFE and the Moondream helper remain intentionally separate upstream repositories: their code is **not** copied, merged, or vendored into N-Suite. N-Suite clones each repository into its own directory under `libs/`, as before. To avoid silently running newer upstream code that has not been tested with this suite, each checkout is fixed to a known revision:

- Practical-RIFE: `a8a8035323b1c1a4a20753c751780e5b0a879455` (12 August 2024);
- N-Suite Moondream helper: `38af98596e59f2a6c25c6b52b2bd5a672dab4144` (29 January 2024);
- the RIFE 4.7 model archive URL is fixed to DreamingAI revision `572480112b87f9bfbff7579b8a38b483766e455f` instead of the moving `main` branch.

On startup, an existing managed checkout is returned to the corresponding revision. Local changes inside `libs/rifle` or `libs/moondream_repo` must therefore be committed or moved elsewhere before starting ComfyUI; Git deliberately refuses the checkout rather than overwriting them.

### Known open issues checked on 27 September 2026

- [#90](https://github.com/Nuked88/ComfyUI-N-Nodes/issues/90) reports that MoviePy 2.x removed `moviepy.editor`. N-Suite currently uses that API, so dependencies are constrained to `moviepy<2` until the video nodes are migrated.
- [#87](https://github.com/Nuked88/ComfyUI-N-Nodes/issues/87), [#84](https://github.com/Nuked88/ComfyUI-N-Nodes/issues/84), [#83](https://github.com/Nuked88/ComfyUI-N-Nodes/issues/83), [#63](https://github.com/Nuked88/ComfyUI-N-Nodes/issues/63), and [#56](https://github.com/Nuked88/ComfyUI-N-Nodes/issues/56) report missing nodes or installation/import failures. Declaring all direct Python dependencies and removing runtime `pip` calls addresses part, but not necessarily all, of this group.
- [#80](https://github.com/Nuked88/ComfyUI-N-Nodes/issues/80), [#79](https://github.com/Nuked88/ComfyUI-N-Nodes/issues/79), and [#75](https://github.com/Nuked88/ComfyUI-N-Nodes/issues/75) are specifically about missing or incompatible `llama_cpp`; version 1.2 removes that failing integration.
- [#89](https://github.com/Nuked88/ComfyUI-N-Nodes/issues/89) reports creation of an output folder at startup; [#78](https://github.com/Nuked88/ComfyUI-N-Nodes/issues/78), [#62](https://github.com/Nuked88/ComfyUI-N-Nodes/issues/62), [#60](https://github.com/Nuked88/ComfyUI-N-Nodes/issues/60), and [#59](https://github.com/Nuked88/ComfyUI-N-Nodes/issues/59) cover video frame, memory and path handling and remain separate work.
- [#66](https://github.com/Nuked88/ComfyUI-N-Nodes/issues/66) and [#65](https://github.com/Nuked88/ComfyUI-N-Nodes/issues/65) report frontend compatibility problems. The video previews now use ComfyUI's managed `addDOMWidget` lifecycle, the undefined global from #65 has been removed, and widget removal uses `removeWidget`; further reports should be checked against a current frontend build.

This list is a triage summary, not a claim that the referenced issues are fixed merely by the packaging changes above.

## Image Pad For Outpainting Advanced 
![alt text](./img/image-14.png)

The `ImagePadForOutpaintingAdvanced` node is an alternative to the `ImagePadForOutpainting` node that applies the technique seen in [this video](https://www.youtube.com/@robadams2451) under the outpainting mask.
The color correction part was taken from [this](https://github.com/sipherxyz/comfyui-art-venture) custom node from Sipherxyz

#### Input Fields

- `image`: Image input.
- `left`: pixel to extend from left,
- `top`: pixel to extend from top,
- `right`: pixel to extend from right,
- `bottom`: pixel to extend from bottom.
- `feathering`: feathering strength
- `noise`: blend strenght from noise and the copied border
- `pixel_size`: how big will be the pixel in the pixellated effect
- `pixel_to_copy`: how many pixels to copy (from each side)
- `temperature`: color correction setting that is only applied to the mask part.
- `hue`: color correction setting that is only applied to the mask part.
- `brightness`: color correction setting that is only applied to the mask part.
- `contrast`: color correction setting that is only applied to the mask part.
- `saturation`: color correction setting that is only applied to the mask part.
- `gamma`: color correction setting that is only applied to the mask part.

#### Output

The node returns the processed image and the mask.

## Dynamic Prompt

![alt text](./img/image-9.png)

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


## CLIP Text Encode Advanced (Experimental)

![alt text](./img/image-10.png)

The `CLIP Text Encode Advanced` node is an alternative to the standard `CLIP Text Encode` node. It offers support for Add/Replace/Delete styles, allowing for the inclusion of both positive and negative prompts within a single node.

The base style file is called `n-styles.csv` and is located in the `ComfyUI\styles` folder.
The styles file follows the same format as the current `styles.csv` file utilized in A1111 (at the time of writing).

NOTE: this note is experimental and still have alot of bugs

#### Input Fields

- `clip`: clip input 
- `style`: it will automatically fill the positive and negative prompts based on the choosen style

#### Output
- `positive`: positive conditions
- `negative`: negative conditions






## Troubleshooting

- ~~**SaveVideo - Preview not working**: is related to a conflict with animateDiff, i've already opened a [PR](https://github.com/ArtVentureX/comfyui-animatediff/pull/64) to solve this issue. Meanwhile you can download my patched version from [here](https://github.com/Nuked88/comfyui-animatediff)~~ pull has been merged so this problem should be fixed now!

## Contributing

Feel free to contribute to this project by reporting issues or suggesting improvements. Open an issue or submit a pull request on the GitHub repository.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
