import os
import cv2
import sys
import torch
import argparse
from PIL import Image, ImageOps
import folder_paths
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import _thread
from queue import Queue, Empty
from pathlib import Path


sys.path.append(os.path.join(str(Path(__file__).parent.parent),"libs","rifle"))

from model.pytorch_msssim import ssim_matlab
interpolation_temp_input_folder = os.path.join(folder_paths.get_temp_directory(),"n-suite","interpolation_input")
interpolation_temp_output_folder = os.path.join(folder_paths.get_temp_directory(),"n-suite","interpolation_output")

try:
    os.makedirs(interpolation_temp_input_folder)
except:
    pass


try:
    os.makedirs(interpolation_temp_output_folder)
except:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scale=1
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


try:
    from  train_log.RIFE_HDv3 import Model
except:
    print("Please download our model from model list")


model = Model()
if not hasattr(model, 'version'):
    model.version = 0

model_folder= os.path.join(folder_paths.folder_names_and_paths["custom_nodes"][0][0],'ComfyUI-N-Nodes','libs','rifle','train_log')
model.load_model(model_folder, -1)
print("Loaded 3.x/4.x HD model.")
model.eval()
model.device()





output_frames = []
def clear_write_buffer(user_args, write_buffer,output_folder):
    cnt = 0
    
    while True:
        item = write_buffer.get()
        if item is None:
            break
      
        cv2.imwrite(os.path.join(output_folder, '{:0>7d}.png'.format(cnt)), item[:, :, ::-1])
        cnt += 1 



def build_read_buffer(img, read_buffer, videogen):
    try:
        for frame in videogen:
            if not img is None:
                frame = cv2.imread(os.path.join(img, frame), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
 
            read_buffer.put(frame)
    except:
        pass
    read_buffer.put(None)

def make_inference(I0, I1, n):    
    global model
    if model.version >= 3.9:
        res = []
        for i in range(n):
            res.append(model.inference(I0, I1, (i+1) * 1. / (n+1), scale))
        return res
    else:
        middle = model.inference(I0, I1, scale)
        if n == 1:
            return [middle]
        first_half = make_inference(I0, middle, n=n//2)
        second_half = make_inference(middle, I1, n=n//2)
        if n%2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]


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

def image_preprocessing(i):
    i = ImageOps.exif_transpose(i)
    image = i.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    return image




_choice = ["YES", "NO"]
_range = ["Fixed", "Random"]
class FrameInterpolator:
    def __init__(self):
        
        self.type = "output"

     
    @classmethod
    def INPUT_TYPES(s):
        #clear directory
        try:
            for file in os.listdir(interpolation_temp_input_folder):
                os.remove(os.path.join(interpolation_temp_input_folder,file))
            for file in os.listdir(interpolation_temp_output_folder):
                os.remove(os.path.join(interpolation_temp_output_folder,file))
        except:
            pass

        return {"required": 
                    {"images": ("IMAGE", ),
                     "METADATA": ("STRING",  {"default": "", "forceInput": True}  ),
                     "multiplier": ("INT", {"default": 2, "min": 1, "step": 1}),
                     
                     },

                }
                


    RETURN_TYPES = ()
    FUNCTION = "save_video"

    OUTPUT_NODE = True

    CATEGORY = "N-Suite/Video"

    RETURN_TYPES = ("IMAGE","STRING",)
    OUTPUT_IS_LIST = (True, False, )   
    RETURN_NAMES = ("IMAGES","METADATA",)
   
    FUNCTION = "interpolate"



    def interpolate(self,images,multiplier,METADATA):
        fps = METADATA[0]*multiplier
        frame_number = METADATA[1]
        video_name = METADATA[2]
 


        for image in images:
            
            full_input_temp_frame_folder,file = get_output_filename("", interpolation_temp_input_folder, ".png") 
            file_name = file
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
           

            #file = f"frame_{counter:05}_.png"
            img.save(full_input_temp_frame_folder, pnginfo=metadata, compress_level=0)


        try:
            file_name_number = int(file.split(".")[0])
        except:
            file_name_number = 0
        image_list = []
        if(file_name_number >= frame_number):
                
            videogen = []
            for f in os.listdir(interpolation_temp_input_folder):
                if 'png' in f:
                    videogen.append(f)
            tot_frame = len(videogen)
            videogen.sort(key= lambda x:int(x[:-4]))
            lastframe = cv2.imread(os.path.join(interpolation_temp_input_folder, videogen[0]), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
            videogen = videogen[1:]
            h, w, _ = lastframe.shape



            tmp = max(128, int(128 / scale))
            ph = ((h - 1) // tmp + 1) * tmp
            pw = ((w - 1) // tmp + 1) * tmp
            padding = (0, pw - w, 0, ph - h)
            pbar = tqdm(total=tot_frame)

            write_buffer = Queue(maxsize=500)
            read_buffer = Queue(maxsize=500)
            _thread.start_new_thread(build_read_buffer, (interpolation_temp_input_folder, read_buffer, videogen))
            _thread.start_new_thread(clear_write_buffer, (interpolation_temp_input_folder, write_buffer, interpolation_temp_output_folder))

            I1 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
            I1 = F.pad(I1, padding)
            temp = None # save lastframe when processing static frame

            


            while True:
                if temp is not None:
                    frame = temp
                    temp = None
                else:
                    frame = read_buffer.get()
                if frame is None:
                    break
                I0 = I1
                I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
                I1 = F.pad(I1, padding)
                I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
                I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
                ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

                break_flag = False
                if ssim > 0.996:
                    frame = read_buffer.get() # read a new frame
                    if frame is None:
                        break_flag = True
                        frame = lastframe
                    else:
                        temp = frame
                    I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
                    I1 = F.pad(I1, padding)
                    I1 = model.inference(I0, I1, scale)
                    I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
                    ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
                    frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
                    
                if ssim < 0.2:
                    output = []
                    for i in range(multiplier - 1):
                        output.append(I0)

                else:
                    output = make_inference(I0, I1, multiplier-1)


                write_buffer.put(lastframe)
                for mid in output:
                    mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
                    write_buffer.put(mid[:h, :w])
                    
                pbar.update(1)
                lastframe = frame
                if break_flag:
                    break


            write_buffer.put(lastframe)


            import time
            while(not write_buffer.empty()):
                time.sleep(0.1)
            pbar.close()


            
            METADATA = [fps, len(os.listdir(interpolation_temp_output_folder)),video_name]
            
            images = [os.path.join(interpolation_temp_output_folder, filename) for filename in os.listdir(interpolation_temp_output_folder) if filename.endswith(".png")]
            images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
            
            for image in images:
            
                image_list.append(image_preprocessing(Image.open(image)))




        return ( image_list,METADATA) 



# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "FrameInterpolator": FrameInterpolator,

}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Video": "FrameInterpolator"
}




