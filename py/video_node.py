import torch
import folder_paths
from PIL import Image, ImageOps
import numpy as np
import safetensors.torch
import hashlib
import os
import cv2
import os
import imageio
import shutil
from moviepy.editor import VideoFileClip, AudioFileClip
import random
import math
import json
from comfy.cli_args import args
import time
import concurrent.futures

# Brutally copied from comfy_extras/nodes_rebatch.py and modified
class LatentRebatch:

    @staticmethod
    def get_batch(latents, list_ind, offset):
        '''prepare a batch out of the list of latents'''
        samples = latents[list_ind]['samples']
        shape = samples.shape
        mask = latents[list_ind]['noise_mask'] if 'noise_mask' in latents[list_ind] else torch.ones((shape[0], 1, shape[2]*8, shape[3]*8), device='cpu')
        if mask.shape[-1] != shape[-1] * 8 or mask.shape[-2] != shape[-2]:
            torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(shape[-2]*8, shape[-1]*8), mode="bilinear")
        if mask.shape[0] < samples.shape[0]:
            mask = mask.repeat((shape[0] - 1) // mask.shape[0] + 1, 1, 1, 1)[:shape[0]]
        if 'batch_index' in latents[list_ind]:
            batch_inds = latents[list_ind]['batch_index']
        else:
            batch_inds = [x+offset for x in range(shape[0])]
        return samples, mask, batch_inds

    @staticmethod
    def get_slices(indexable, num, batch_size):
        '''divides an indexable object into num slices of length batch_size, and a remainder'''
        slices = []
        for i in range(num):
            slices.append(indexable[i*batch_size:(i+1)*batch_size])
        if num * batch_size < len(indexable):
            return slices, indexable[num * batch_size:]
        else:
            return slices, None
    
    @staticmethod
    def slice_batch(batch, num, batch_size):
        result = [LatentRebatch.get_slices(x, num, batch_size) for x in batch]
        return list(zip(*result))

    @staticmethod
    def cat_batch(batch1, batch2):
        if batch1[0] is None:
            return batch2
        result = [torch.cat((b1, b2)) if torch.is_tensor(b1) else b1 + b2 for b1, b2 in zip(batch1, batch2)]
        return result

    def rebatch(self, latents, batch_size):
        batch_size = batch_size[0]

        output_list = []
        current_batch = (None, None, None)
        processed = 0

        for i in range(len(latents)):
            # fetch new entry of list
            #samples, masks, indices = self.get_batch(latents, i)
            next_batch = self.get_batch(latents, i, processed)
            processed += len(next_batch[2])
            # set to current if current is None
            if current_batch[0] is None:
                current_batch = next_batch
            # add previous to list if dimensions do not match
            elif next_batch[0].shape[-1] != current_batch[0].shape[-1] or next_batch[0].shape[-2] != current_batch[0].shape[-2]:
                sliced, _ = self.slice_batch(current_batch, 1, batch_size)
                output_list.append({'samples': sliced[0][0], 'noise_mask': sliced[1][0], 'batch_index': sliced[2][0]})
                current_batch = next_batch
            # cat if everything checks out
            else:
                current_batch = self.cat_batch(current_batch, next_batch)

            # add to list if dimensions gone above target batch size
            if current_batch[0].shape[0] > batch_size:
                num = current_batch[0].shape[0] // batch_size
                sliced, remainder = self.slice_batch(current_batch, num, batch_size)
                
                for i in range(num):
                    output_list.append({'samples': sliced[0][i], 'noise_mask': sliced[1][i], 'batch_index': sliced[2][i]})

                current_batch = remainder

        #add remainder
        if current_batch[0] is not None:
            sliced, _ = self.slice_batch(current_batch, 1, batch_size)
            output_list.append({'samples': sliced[0][0], 'noise_mask': sliced[1][0], 'batch_index': sliced[2][0]})

        #get rid of empty masks
        for s in output_list:
            if s['noise_mask'].mean() == 1.0:
                del s['noise_mask']

        return output_list


input_dir = os.path.join(folder_paths.get_input_directory(),"n-suite")
output_dir = os.path.join(folder_paths.get_output_directory(),"n-suite")
frames_output_dir = os.path.join(folder_paths.get_output_directory(),"frames")
videos_output_dir = os.path.join(folder_paths.get_output_directory(),"videos")
audios_output_temp_dir = os.path.join(folder_paths.get_temp_directory(),"audio.mp3")
videos_output_temp_dir = os.path.join(folder_paths.get_temp_directory(),"video.mp4")
video_preview_output_temp_dir = os.path.join(folder_paths.get_output_directory(),"videos")
_resize_type = ["none","width", "height"]
_framerate = ["original","half", "quarter"]
_choice = ["Yes", "No"]
try:
    os.mkdir(input_dir)
except:
    pass

try:
    os.mkdir(videos_output_dir)
except:
    pass

try:
    os.mkdir(frames_output_dir)
except:
    pass

try:
    os.mkdir(folder_paths.get_temp_directory())
except:
    pass


def calc_resize_image(input_path, target_size, resize_by):
    image = cv2.imread(input_path)
    height, width = image.shape[:2]

    if resize_by == 'width':
       
        new_width = target_size
        new_height = int(height * (target_size / width))
       
    elif resize_by == 'height':
       
        new_height = target_size
        new_width = int(width * (target_size / height))
    
    else:

        new_height = height
        new_width = width
    
    return  new_width, new_height
        

def resize_image(input_path, new_width, new_height):

    image = cv2.imread(input_path)
    height, width = image.shape[:2]

    if height != new_height or width != new_width:
        resized_image = cv2.resize(image, (new_width, new_height))
    else:
        resized_image = image
 
        
    pil_image = Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))


    return pil_image

""" def extract_frames_from_video(video_path, output_folder):

    list_files = []
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
        list_files.append(frame_filename)
        cv2.imwrite(frame_filename, frame)

    cap.release()
    print(f"{frame_count} frames have been extracted from the video and saved in {output_folder}")
    return list_files """


def extract_frames_from_video(video_path, output_folder, target_fps=30):
    list_files = []
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    # Ottieni il framerate originale del video
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    # Calcola il rapporto per ridurre il framerate
    frame_skip_ratio = original_fps // target_fps
    real_frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Estrai solo ogni "frame_skip_ratio"-esimo fotogramma
        if frame_count % frame_skip_ratio == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
            list_files.append(frame_filename)
            cv2.imwrite(frame_filename, frame)
            real_frame_count += 1

    cap.release()
    print(f"{real_frame_count} frames have been extracted from the video and saved in {output_folder}")
    return list_files


def extract_frames_from_gif(gif_path, output_folder, target_fps=30):
    list_files = []
    os.makedirs(output_folder, exist_ok=True)
    real_frame_count = 0
    metadata = imageio.v3.immeta(gif_path)

    gif_frames = imageio.mimread(gif_path)
    original_fps = len(gif_frames)
    frame_skip_ratio = original_fps // original_fps
    frame_count = 0
    for frame in gif_frames:
        frame_count += 1

        if frame_count % frame_skip_ratio == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
            cv2.imwrite(frame_filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            list_files.append(frame_filename)
            real_frame_count += 1
    
    print(f"{frame_count} frames have been extracted from the GIF and saved in {output_folder}")
    return list_files,metadata


def get_output_filename(input_file_path, output_folder, file_extension,suffix="") :
    input_filename = os.path.basename(input_file_path)
    input_filename_without_extension = os.path.splitext(input_filename)[0]

    existing_files = [f for f in os.listdir(output_folder) if f.startswith(input_filename_without_extension)]
    max_progressive = 0
    for filename in existing_files:
        parts_ext = filename.split(".")
        parts = parts_ext[0].split("_")
        if len(parts) == 2 and parts[1].isdigit():
            progressive = int(parts[1])
            max_progressive = max(max_progressive, progressive)


    
    new_progressive = max_progressive + 1
    new_filename = f"{input_filename_without_extension}_{new_progressive:02d}{suffix}{file_extension}"

    return os.path.join(output_folder, new_filename), new_filename

def image_preprocessing(i):
    i = ImageOps.exif_transpose(i)
    image = i.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    return image

def create_video_from_frames(frame_folder, output_video, frame_rate = 30.0): 
    frame_filenames = [os.path.join(frame_folder, filename) for filename in os.listdir(frame_folder) if filename.endswith(".png")]
    frame_filenames.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    first_frame = cv2.imread(frame_filenames[0])
    height, width, layers = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

    for frame_filename in frame_filenames:
        frame = cv2.imread(frame_filename)
        out.write(frame)

    out.release()
    print(f"Frames have been successfully reassembled into {output_video}")

def create_gif_from_frames(frame_folder, output_gif, metadata):
    frame_filenames = [os.path.join(frame_folder, filename) for filename in os.listdir(frame_folder) if filename.endswith(".png")]
    frame_filenames.sort()

    frames = [imageio.imread(frame_filename) for frame_filename in frame_filenames]

    # imageio
    imageio.mimsave(output_gif, frames, loop=metadata[3], duration=metadata[4])  


    print(f"Frames have been successfully assembled into {output_gif}")


temp_dir= folder_paths.temp_directory


class VideoLoader:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required": {"video": (sorted(files), {"image_upload": True} ), 
                            "local_url": ("STRING",  {"default": ""} ),  
                            "framerate": (_framerate, {"default": "original"} ), 
                            "resize_by": (_resize_type,{"default": "none"} ),
                              "size": ("INT", {"default": 512, "min": 512, "step": 64}),
                              "images_limit": ("INT", {"default": 0, "min": 0, "step": 1}),
                              "batch_size": ("INT", {"default": 0, "min": 0, "step": 1})
                          
                            },}


    RETURN_TYPES = ("IMAGE","LATENT","STRING","INT","INT",)
    OUTPUT_IS_LIST = (True, True, False, False,False, )   
    RETURN_NAMES = ("IMAGES","EMPTY LATENT","METADATA","WIDTH","HEIGHT")
    CATEGORY = "video"
    FUNCTION = "encode"


    @staticmethod
    def vae_encode_crop_pixels(pixels):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        return pixels
    
    def load_video(self, video,framerate, local_url):
        
        file_path = folder_paths.get_annotated_filepath(os.path.join("n-suite",video))
        cap = cv2.VideoCapture(file_path)
        # Check if the video was opened successfully
        if not cap.isOpened():
            print("Unable to open the video.")
        else:
            # Get the FPS of the video
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            print(f"The video has {fps} frames per second.")

        #shutil.rmtree(output_dir)
        #print(f"Temporary folder {output_dir} has been emptied.")
        
        #set new framerate
        
        if "half" in framerate:
            fps = fps // 2
            print (f"The video has been reduced to {fps} frames per second.")
        elif "quarter" in framerate:
            fps = fps // 4
            print (f"The video has been reduced to {fps} frames per second.")


        # Estract frames
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == ".mp4":
            list_files = extract_frames_from_video(file_path, output_dir, target_fps=fps)
            meta = {"loop": 0, "duration": 0}

            audio_clip = VideoFileClip(file_path).audio
            try:
                #save audio
                audio_clip.write_audiofile(audios_output_temp_dir)
            except:
                pass

               
        elif file_extension == ".gif":
            list_files,meta = extract_frames_from_gif(file_path, output_dir)
            #create_gif_from_frames(output_dir, output_video2)
        
        else:
            print("Format not supported. Please provide an MP4 or GIF file.")

        
        return list_files,fps,file_extension,meta["loop"],meta["duration"]

    def generate_latent(self, width, height, batch_size=1):
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return {"samples":latent}
    
    def process_image(self,args):
        image_path, width, height = args
        # Funzione per ridimensionare e pre-elaborare un'immagine
        image = resize_image(image_path, width, height)
        image = image_preprocessing(image)
        return torch.tensor(image)
    
    def encode(self,video,framerate, local_url, resize_by, size, images_limit,batch_size):
        metadata = []
        FRAMES,fps,file_extension,loop,duration = self.load_video(video,framerate, local_url)
        pool_size=5
        t_list = []
        i_list = [] 
        i = 0
        o = 0
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            width, height = calc_resize_image(FRAMES[0], size, resize_by)


            for batch_start in range(0, len(FRAMES), pool_size):
                batch_images = FRAMES[batch_start:batch_start + pool_size]
                #remove audio if image_limit > 0
                if images_limit != 0:
                    try:
                        os.remove(audios_output_temp_dir)
                    except:
                        pass
      
                    if o >= images_limit:
                        break

                for image_path in batch_images:
                    args = (image_path, width, height)
                    futures.append(executor.submit(self.process_image, args))
                    o += 1
                    if images_limit != 0:
                        if o >= images_limit:
                            break
                    

                i += len(batch_images)

            # Attendi il completamento delle operazioni in parallelo
            concurrent.futures.wait(futures)

            # Recupera i risultati
            for future in futures:
                batch_i_tensors = future.result()
                i_list.extend(batch_i_tensors)

        i_tensor = torch.stack(i_list, dim=0)
       

        if images_limit != 0:
            b_size=images_limit
        else:
            b_size=len(FRAMES)


        latent = self.generate_latent( width, height, batch_size=b_size)
        
        metadata.append(fps)
        metadata.append(b_size)
        metadata.append(file_extension)
        metadata.append(loop)
        metadata.append(duration)

        if batch_size != 0:
            rebatcher = LatentRebatch()
            rebatched_latent = rebatcher.rebatch([latent], [batch_size])
            n_chunks = b_size//batch_size
            i_tensor_batches = torch.chunk(i_tensor, n_chunks, dim=0)
           
            return (i_tensor_batches,rebatched_latent,metadata, width, height,)
        
        return ( [i_tensor],[latent],metadata, width, height,) 
    
    
class VideoSaver:
    def __init__(self):
        
        self.type = "output"

     
    @classmethod
    def INPUT_TYPES(s):
        s.video_file_path,s.video_filename = get_output_filename("video", videos_output_dir, ".mp4")
        s.gif_file_path,s.gif_filename = get_output_filename("gif", videos_output_dir, ".gif")
        
        try:
            shutil.rmtree(frames_output_dir)
            os.mkdir(frames_output_dir)
        except:
            pass
        

        print(f"Temporary folder {frames_output_dir} has been emptied.")
        return {"required": 
                    {"images": ("IMAGE", ),
                     "METADATA": ("STRING",  {"default": "", "forceInput": True}  ),  
                      "SaveVideo": (_choice,{"default": "No"} ),
                     },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }
                

    RETURN_TYPES = ()
    FUNCTION = "save_video"

    OUTPUT_NODE = True

    CATEGORY = "video"

    def save_video(self, images,METADATA,SaveVideo, prompt=None, extra_pnginfo=None):
       
        fps = METADATA[0]
        frame_number = METADATA[1]
        file_extension = METADATA[2]
  

        results = list()
        
        #full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path("", frames_output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for image in images:

            full_output_folder,file = get_output_filename("frame", frames_output_dir, ".png") 

            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
           

            #file = f"frame_{counter:05}_.png"
            img.save(full_output_folder, pnginfo=metadata, compress_level=4)
            results.append({
                "filename": file,
                "subfolder": "frames",
                "type": self.type
            })
        try:
            file_name_number = int(file.split(".")[0].split("_")[1])
        except:
            file_name_number = 0

        if(file_name_number >= frame_number):
            if file_extension == ".mp4":
                create_video_from_frames(frames_output_dir, videos_output_temp_dir,frame_rate=fps)
            
                video_clip = VideoFileClip(videos_output_temp_dir)
                try:
                    audio_clip =  AudioFileClip(audios_output_temp_dir)
                    video_clip = video_clip.set_audio(audio_clip)
                except:
                    pass
                if SaveVideo == "Yes":
                    video_clip.write_videofile(self.video_file_path)
                    file_name = self.video_filename
                else:
                    #delete all temporary files that start with video_preview
                    for file in os.listdir(video_preview_output_temp_dir):
                        if file.startswith("video_preview"):
                            os.remove(os.path.join(video_preview_output_temp_dir,file))
                    #random number
                    suffix = str(random.randint(1,100000))
                    file_name = f"video_preview_{suffix}.mp4"
                    video_clip.write_videofile(os.path.join(video_preview_output_temp_dir,file_name))
            elif file_extension == ".gif":

                if SaveVideo == "Yes":
                    create_gif_from_frames(frames_output_dir, os.path.join(video_preview_output_temp_dir,self.gif_filename),METADATA)
                    file_name = self.gif_filename
                else:
                    #delete all temporary files that start with video_preview
                    for file in os.listdir(video_preview_output_temp_dir):
                        if file.startswith("gif_preview"):
                            os.remove(os.path.join(video_preview_output_temp_dir,file))
                    #random number
                    suffix = str(random.randint(1,100000))
                    file_name = f"gif_preview_{suffix}.gif"
                    create_gif_from_frames(frames_output_dir,os.path.join(video_preview_output_temp_dir,file_name),METADATA)
                    
                
          



        return {"ui": {"text": [file_name],}}
    
class LoadFramesFromFolder:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "folder":("STRING",  {"default": ""} ),
                             "fps":("INT", {"default": 30}),
                             "loop_for_gif": ("INT", {"default": 0}),
                             "duration_for_gif":("INT", {"default": 0}),
                             
                             }}
    

    RETURN_TYPES = ("IMAGE","STRING",)
    RETURN_NAMES = ("IMAGES","METADATA")
  
    FUNCTION = "load_images"
    OUTPUT_IS_LIST = (True,False,)
    CATEGORY = "video"

    def load_images(self, folder,fps,loop_for_gif,duration_for_gif):
        image_list = []
        METADATA = [fps, len(os.listdir(folder))]
        
        images = [os.path.join(folder, filename) for filename in os.listdir(folder) if filename.endswith(".png")]
        images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        for image in images:
        
            image_list.append(image_preprocessing(Image.open(image)))

        #i_tensor = torch.stack(image_list, dim=0)

        return (image_list,METADATA,)



# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VideoLoader": VideoLoader,
    "VideoSaver":VideoSaver,
    "LoadFramesFromFolder": LoadFramesFromFolder
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Video": "VideoLoader",
    "Video": "VideoSaver",
    "Video": "LoadFramesFromFolder"
}
