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


def image_preprocessing(i):
    i = ImageOps.exif_transpose(i)
    image = i.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    return image     
class LoadImageFromFolder:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "folder":("STRING",  {"default": ""} ),
                             "fps":("INT", {"default": 30})
                             }}
    

    RETURN_TYPES = ("IMAGE","INT","INT","INT","STRING","STRING",)
    RETURN_NAMES = ("IMAGES","MAX WIDTH","MAX HEIGHT","IMAGE COUNT","PATH","IMAGE LIST")
    FUNCTION = "load_images"
    OUTPUT_IS_LIST = (True,False,False,False,False,False,)

    CATEGORY = "N-Suite/Experimental"

    def load_images(self, folder,fps):
        image_list = []
        image_names = []
        max_width = 0
        max_height = 0
        frame_count = 0
   
        
        images = [os.path.join(folder, filename) for filename in os.listdir(folder) if filename.endswith(".png") or filename.endswith(".jpg")]
        
        
        for image_path in images:
            #get image name
            image_names.append(image_path.split("/")[-1])
            image = Image.open(image_path)
            width, height = image.size
            max_width = max(max_width, width)
            max_height = max(max_height, height)
            image_list.append((image_preprocessing(image)))
            frame_count += 1
    
        image_names_final='\n'.join(image_names)
        print (f"Details: {frame_count} frames, {max_width}x{max_height}")

        return (image_list, max_width, max_height,frame_count,folder,image_names_final,)


class SaveCaptionsFromImageList:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "folder":("STRING",  {"default": ""} ),
                             "fps":("INT", {"default": 30})
                             }}
    

    RETURN_TYPES = ("IMAGE","INT","INT","INT","STRING","STRING",)
    RETURN_NAMES = ("IMAGES","MAX WIDTH","MAX HEIGHT","IMAGE COUNT","PATH","IMAGE LIST")
    FUNCTION = "load_images"
    OUTPUT_IS_LIST = (True,False,False,False,False,False,)

    CATEGORY = "LJRE/Loader"

    def load_images(self, folder,fps):
        image_list = []
        image_names = []
        max_width = 0
        max_height = 0
        frame_count = 0
   
        
        images = [os.path.join(folder, filename) for filename in os.listdir(folder) if filename.endswith(".png") or filename.endswith(".jpg")]
        
        
        for image_path in images:
            #get image name
            image_names.append(image_path.split("/")[-1])
            image = Image.open(image_path)
            width, height = image.size
            max_width = max(max_width, width)
            max_height = max(max_height, height)
            image_list.append((image_preprocessing(image)))
            frame_count += 1
    
        image_names_final='\n'.join(image_names)
        print (f"Details: {frame_count} frames, {max_width}x{max_height}")

        return (image_list, max_width, max_height,frame_count,folder,image_names_final,)

# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "LoadImageFromFolder [n-suite]": LoadImageFromFolder,

}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageFromFolder [n-suite]": "Load Image From Folder [🅝-🅢🅤🅘🅣🅔]"
}




