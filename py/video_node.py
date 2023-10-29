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
import skbuild




YELLOW = '\33[33m'
END = '\33[0m'

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
output_dir = os.path.join(folder_paths.get_output_directory(),"n-suite","frames_out")
temp_output_dir = os.path.join(folder_paths.get_temp_directory(),"n-suite","frames_out")
frames_output_dir = os.path.join(folder_paths.get_temp_directory(),"n-suite","frames")
videos_output_dir = os.path.join(folder_paths.get_output_directory(),"n-suite","videos")
audios_output_temp_dir = os.path.join(folder_paths.get_temp_directory(),"audio.mp3")
videos_output_temp_dir = os.path.join(folder_paths.get_temp_directory(),"video.mp4")
video_preview_output_temp_dir = os.path.join(folder_paths.get_output_directory(),"n-suite","videos")
_resize_type = ["none","width", "height"]
_framerate = ["original","half", "quarter"]
_choice = ["Yes", "No"]
try:
    os.makedirs(input_dir)
except:
    pass
try:
    os.makedirs(output_dir)
except:
    pass
 
try:
    os.makedirs(temp_output_dir)
except:
    pass

try:
    os.makedirs(videos_output_dir)
except:
    pass

try:
    os.makedirs(frames_output_dir)
except:
    pass

try:
    os.makedirs(folder_paths.get_temp_directory())
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
            frame_filename = os.path.join(output_folder, f"{frame_count:07d}.png")
            list_files.append(frame_filename)
            cv2.imwrite(frame_filename, frame)
            real_frame_count += 1

    cap.release()
    print(f"{real_frame_count} frames have been extracted from the video and saved in {output_folder}")
    return list_files


def extract_frames_from_gif(gif_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    gif_frames = imageio.mimread(gif_path)
    
    frame_count = 0
    for frame in gif_frames:
        frame_count += 1
        frame_filename = os.path.join(output_folder, f"{frame_count:07d}.png")
        cv2.imwrite(frame_filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    print(f"{frame_count} frames have been extracted from the GIF and saved in {output_folder}")

def get_output_filename(input_file_path, output_folder, file_extension,suffix="") :
    existing_files = [f for f in os.listdir(output_folder)]
    max_progressive = 0
    for filename in existing_files:
        parts_ext = filename.split(".")
        parts = parts_ext[0]

        if len(parts) > 2 and parts.isdigit():
            progressive = int(parts)
            max_progressive = max(max_progressive, progressive)


    
    new_progressive = max_progressive + 1
    new_filename = f"{new_progressive:07d}{suffix}{file_extension}"

    return os.path.join(output_folder, new_filename), new_filename


def get_output_filename_video(input_file_path, output_folder, file_extension,suffix="") :
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

def create_gif_from_frames(frame_folder, output_gif):
    frame_filenames = [os.path.join(frame_folder, filename) for filename in os.listdir(frame_folder) if filename.endswith(".png")]
    frame_filenames.sort()

    frames = [imageio.imread(frame_filename) for frame_filename in frame_filenames]

    # imageio
    imageio.mimsave(output_gif, frames, duration=0.1)  


    print(f"Frames have been successfully assembled into {output_gif}")


temp_dir= folder_paths.temp_directory


class LoadVideo:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required": {"video": (sorted(files), ), 
                            "local_url": ("STRING",  {"default": ""} ),  
                            "framerate": (_framerate, {"default": "original"} ), 
                            "resize_by": (_resize_type,{"default": "none"} ),
                              "size": ("INT", {"default": 512, "min": 512, "step": 64}),
                              "images_limit": ("INT", {"default": 0, "min": 0, "step": 1}),
                              "batch_size": ("INT", {"default": 0, "min": 0, "step": 1}),
                              "starting_frame": ("INT", {"default": 0, "min": 0, "step": 1}), 
                              "autoplay":("BOOLEAN",{"default": True} ),
                            },}


    RETURN_TYPES = ("IMAGE","LATENT","STRING","INT","INT",)
    OUTPUT_IS_LIST = (True, True, False, False,False, )   
    RETURN_NAMES = ("IMAGES","EMPTY LATENTS","METADATA","WIDTH","HEIGHT")
    CATEGORY = "N-Suite/Video"
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

        try:
            shutil.rmtree(os.path.join(temp_output_dir,video.split(".")[0]))
        except:
            print("Video Path already deleted")
    

        full_temp_output_dir = os.path.join(temp_output_dir,video.split(".")[0])
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
            list_files = extract_frames_from_video(file_path, full_temp_output_dir, target_fps=fps)

            audio_clip = VideoFileClip(file_path).audio
            try:
                #save audio
                audio_clip.write_audiofile(os.path.join(temp_output_dir,video.split(".")[0],"audio.mp3"))
            except:
                print("Could not save audio")
                pass

            """        
        elif file_extension == ".gif":
            extract_frames_from_gif(file_path, output_dir)
            #create_gif_from_frames(output_dir, output_video2)
            """
        else:
            print("Format not supported. Please provide an MP4 or GIF file.")

        
        return list_files,fps

    def generate_latent(self, width, height, batch_size=1):
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return {"samples":latent}
    
    def process_image(self,args):
        image_path, width, height = args
        # Funzione per ridimensionare e pre-elaborare un'immagine
        image = resize_image(image_path, width, height)
        image = image_preprocessing(image)
        return torch.tensor(image)
    
    def encode(self,video,framerate, local_url, resize_by, size, images_limit,batch_size,starting_frame,autoplay):
        metadata = []
        FRAMES,fps = self.load_video(video,framerate, local_url)
        max_frames = len(FRAMES)

        
      
        if images_limit>0 and starting_frame>0:
            images_limit = images_limit + starting_frame
       
        
        #if images_limit==0:
        #    images_limit = max_frames


        print(f"images_limit {images_limit}")



        #if starting frame is too high do the last frames only  
        if starting_frame>max_frames:
            starting_frame = max_frames-1
            print(f"{YELLOW}WARNING: The starting frame is greater than the number of frames in the video. Only the last frame of the video will be used ({starting_frame}). {END}")


        #if images_limit > max_frames
        if images_limit > max_frames:
            images_limit = max_frames
            print(f"{YELLOW}WARNING: The number of images to extract is greater than the number of frames in the video. Images_limit has been reduced to the number of frames ({images_limit}). {END}")
            

        #if batch_size > max_frames
        if batch_size > max_frames:
            print(f"{YELLOW}WARNING: The batch size is greater than the number of frames requested. Batch size has been reduced. {END}")
            batch_size = max_frames

       #if batch_size > images_limit
        if images_limit!=0 and batch_size > images_limit:
            print(f"{YELLOW}WARNING: The batch size is greater than the number of frames requested. Batch size has been reduced to the number of images_limit. {END}")
            batch_size = images_limit



        pool_size=5
        t_list = []
        i_list = [] 
        i = 0
        o = 0
        final_count_frame=0
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            width, height = calc_resize_image(FRAMES[0], size, resize_by)


            for batch_start in range(0, len(FRAMES), pool_size):
                batch_images = FRAMES[batch_start:batch_start + pool_size]
                
                #remove audio if image_limit > 0 or starting_frame>0
                if images_limit != 0 or starting_frame != 0:
                    try:
                        os.remove(os.path.join(temp_output_dir,video.split(".")[0],"audio.mp3"))
                    except:
                        pass
        
                    #if o >= images_limit:
                    #    break

                for image_path in batch_images:
                    # loop only when it reaches the starting_frame 
                    
                    if o>=starting_frame and (o<images_limit or images_limit==0):
                        args = (image_path, width, height)
                        futures.append(executor.submit(self.process_image, args))
                        final_count_frame += 1
                        
                    o += 1
          
                            
                
                i += len(batch_images)

            # Attendi il completamento delle operazioni in parallelo
            concurrent.futures.wait(futures)

            # Recupera i risultati
            for future in futures:
                batch_i_tensors = future.result()
                i_list.extend(batch_i_tensors)

        i_tensor = torch.stack(i_list, dim=0)
       

        if images_limit != 0 or starting_frame != 0:
            
            b_size=final_count_frame
        else:
            b_size=len(FRAMES)


        latent = self.generate_latent( width, height, batch_size=b_size)
        
        metadata.append(fps)
        metadata.append(b_size)
        try:
            metadata.append(video.split(".")[0])
        except:
            print("No video name")

        if batch_size != 0:
            rebatcher = LatentRebatch()
            rebatched_latent = rebatcher.rebatch([latent], [batch_size])
            n_chunks = b_size//batch_size
            i_tensor_batches = torch.chunk(i_tensor, n_chunks, dim=0)
           
            return (i_tensor_batches,rebatched_latent,metadata, width, height,)
        
        return ( [i_tensor],[latent],metadata, width, height,) 
    
    
class SaveVideo:
    def __init__(self):
        
        self.type = "output"

     
    @classmethod
    def INPUT_TYPES(s):
        s.video_file_path,s.video_filename = get_output_filename_video("video", videos_output_dir, ".mp4")
        
        try:
            shutil.rmtree(frames_output_dir)
            os.mkdir(frames_output_dir)
        except:
            pass
        

        #print(f"Temporary folder {frames_output_dir} has been emptied.")
        return {"required": 
                    {"images": ("IMAGE", ),
                     "METADATA": ("STRING",  {"default": "", "forceInput": True}  ),  
                      "SaveVideo": ("BOOLEAN",{"default": False} ),
                      "SaveFrames": ("BOOLEAN",{"default": False} ),
                      "CompressionLevel":  ("INT", {"default": 2, "min": 0, "max":9, "step": 1}),
                      
                     },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }
                

    RETURN_TYPES = ()
    FUNCTION = "save_video"

    OUTPUT_NODE = True

    CATEGORY = "N-Suite/Video"

    def save_video(self, images,METADATA,SaveVideo,SaveFrames, CompressionLevel, prompt=None, extra_pnginfo=None):
 
        fps = METADATA[0]
        frame_number = METADATA[1]
        video_filename_original = METADATA[2]
    
        
        #full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path("", frames_output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
    
        for image in images:
            
            full_output_folder,file = get_output_filename("", frames_output_dir, ".png") 
            file_name = file
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
           

            #file = f"frame_{counter:05}_.png"
            img.save(full_output_folder, pnginfo=metadata, compress_level=CompressionLevel)
            results.append({
                "filename": file,
                "subfolder": "frames",
                "type": self.type
            })
            
        try:
            file_name_number = int(file.split(".")[0])
        except:
            file_name_number = 0

        if(file_name_number >= frame_number):
            create_video_from_frames(frames_output_dir, videos_output_temp_dir,frame_rate=fps)
        
            video_clip = VideoFileClip(videos_output_temp_dir)
            try:
                audio_clip =  AudioFileClip(os.path.join(temp_output_dir,video_filename_original,"audio.mp3"))
                video_clip = video_clip.set_audio(audio_clip)
            except:
                print("No audio found")
                pass
            
            if SaveFrames == True:
                #copy frames_output_dir to self.video_file_path/self.video_filename
                frame_folder=os.path.join(videos_output_dir,self.video_filename.split(".")[0])
                
                shutil.copytree(frames_output_dir, frame_folder)

            if SaveVideo == True:
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
                
          



        return {"ui": {"text": [file_name],}}
    
class LoadFramesFromFolder:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "folder":("STRING",  {"default": ""} ),
                             "fps":("INT", {"default": 30})
                            
                             
                             
                             }}
    

    RETURN_TYPES = ("IMAGE","STRING",)
    RETURN_NAMES = ("IMAGES","METADATA")
    FUNCTION = "load_images"
    OUTPUT_IS_LIST = (True,False,)
    CATEGORY = "N-Suite/Video"

    def load_images(self, folder,fps):
        image_list = []
        METADATA = [fps, len(os.listdir(folder)),"load"]
        
        images = [os.path.join(folder, filename) for filename in os.listdir(folder) if filename.endswith(".png")]
        images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        
        for image in images:
        
            image_list.append(image_preprocessing(Image.open(image)))

        #i_tensor = torch.stack(image_list, dim=0)
        return (image_list,METADATA,)

    
class SetMetadata:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "number_of_frames":("INT",  {"default": 1, "min": 1, "step": 1}),
                             "fps":("INT", {"default": 30, "min": 1, "step": 1}),
                               "VideoName": ("STRING",  {"default": "manual"}  )
                             
                             
                             }}
    

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("METADATA",)
    FUNCTION = "set_metadata"
    OUTPUT_IS_LIST = (False,)
    CATEGORY = "N-Suite/Video"

    def set_metadata(self, number_of_frames,fps,VideoName):
     
        METADATA = [fps, number_of_frames,VideoName]
        return (METADATA,)




# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "LoadVideo": LoadVideo,
    "SaveVideo":SaveVideo,
    "LoadFramesFromFolder": LoadFramesFromFolder,
    "SetMetadataForSaveVideo": SetMetadata
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Video": "LoadVideo",
    "Video": "SaveVideo",
    "Video": "LoadFramesFromFolder",
    "Video": "SetMetadataForSaveVideo"
}
