import torch 
import math
import numpy as np
import os
import random

MAX_RESOLUTION = 4096



class ImagePadForOutpaintAdvanced:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "left": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "top": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "right": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "feathering": ("INT", {"default": 40, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "noise_type": ("INT", {"default": 1, "min": 1, "max": 3, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "expand_image"

    CATEGORY = "image"

    def expand_image(self, image, left, top, right, bottom, feathering,noise_type):
        d1, max_row, max_col, d4 = image.size()

        new_image = torch.zeros(
            (d1, max_row + top + bottom, max_col + left + right, d4),
            dtype=torch.float32,
        )
        new_image[:, top:top + max_row, left:left + max_col, :] = image

        mask = torch.ones(
            (max_row + top + bottom, max_col + left + right),
            dtype=torch.float32,
        )

        t = torch.zeros(
            (max_row, max_col),
            dtype=torch.float32
        )

        if feathering > 0 and feathering * 2 < max_row and feathering * 2 < max_col:
            padding_left = feathering
            padding_right = feathering
            padding_top = feathering
            padding_bottom = feathering
            grow = 0
            #max_row = vertical
            #max_col = horizontal
            for row in range(max_row):
                for col in range(max_col):
                    #print(max_row) #768
                    #print((max_row - padding + 5)) #743
                                            
                    if  col <= padding_left - grow:
                        dt = 0

                    elif col >= max_col - padding_right + grow:
                        dt = 0
                        
                    elif row <= padding_top + grow:
                        dt = 0
                    elif row >= max_row - padding_bottom + grow:
                        dt = 0
                    else:
                        dt = max_row
                        



                    grow += 1
                    if grow > padding_left:
                        grow=0    

                        
                    db = max_row
                    dl =  max_col
                    dr = max_col
                    d = min(dt, db, dl, dr)
                    if d >= feathering:
                        continue
                    v = (feathering - d) / feathering
                    t[row, col] = v * v

        mask[top:top + max_row, left:left + max_col] = t

        return (new_image, mask)
    

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ImagePadForOutpaintAdvanced [n-suite]": ImagePadForOutpaintAdvanced
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImagePadForOutpaintAdvanced [n-suite]": "Image Pad For Outpaint Advanced [🅝-🅢🅤🅘🅣🅔]"
}
