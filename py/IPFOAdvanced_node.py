import torch 
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import cv2
MAX_RESOLUTION = 4096


def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Adapt from https://github.com/sipherxyz/comfyui-art-venture
def color_correct(
        image,
        temperature: float,
        hue: float,
        brightness: float,
        contrast: float,
        saturation: float,
        gamma: float,
    ):
    
   

        brightness /= 100
        contrast /= 100
        saturation /= 100
        temperature /= 100

        brightness = 1 + brightness
        contrast = 1 + contrast
        saturation = 1 + saturation


        modified_image = image

 

    

        # brightness
        modified_image = ImageEnhance.Brightness(modified_image).enhance(brightness)

        # contrast
        modified_image = ImageEnhance.Contrast(modified_image).enhance(contrast)
        modified_image = np.array(modified_image).astype(np.float32)

        # temperature
        if temperature > 0:
            modified_image[:, :, 0] *= 1 + temperature
            modified_image[:, :, 1] *= 1 + temperature * 0.4
        elif temperature < 0:
            modified_image[:, :, 2] *= 1 - temperature
        modified_image = np.clip(modified_image, 0, 255) / 255

        # gamma
        modified_image = np.clip(np.power(modified_image, gamma), 0, 1)

        # saturation
        hls_img = cv2.cvtColor(modified_image, cv2.COLOR_RGB2HLS)
        hls_img[:, :, 2] = np.clip(saturation * hls_img[:, :, 2], 0, 1)
        modified_image = cv2.cvtColor(hls_img, cv2.COLOR_HLS2RGB) * 255

        # hue
        hsv_img = cv2.cvtColor(modified_image, cv2.COLOR_RGB2HSV)
        hsv_img[:, :, 0] = (hsv_img[:, :, 0] + hue) % 360
        modified_image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

     

        modified_image = modified_image.astype(np.uint8)
        #modified_image = modified_image / 255
       
        #modified_image = torch.from_numpy(modified_image).unsqueeze(0)
     

        return modified_image




def extract_pixels(image, side, num_pixels):
 

    # Ottieni le dimensioni dell'immagine
    width, height = image.size

    # Determina la regione di ritaglio in base al lato specificato
    if side == "l":
        crop_box = (0, 0, num_pixels, height)
    elif side == "r":
        crop_box = (width - num_pixels, 0, width, height)
    elif side == "t":
        crop_box = (0, 0, width, num_pixels)
    elif side == "b":
        crop_box = (0, height - num_pixels, width, height)
    else:
        raise ValueError("Il lato specificato non è valido. Utilizzare 'sinistro', 'destro', 'alto' o 'basso'.")

    # Esegui il ritaglio dell'immagine
    cropped_image = image.crop(crop_box)

    return cropped_image


from PIL import Image



def make_pixelated(image, pixel_size):
    if pixel_size > image.width or pixel_size > image.height:
        raise ValueError("Top, bottom, left, and right padding must higher than the pixel_size!")

    small_image = image.resize((image.width // pixel_size, image.height // pixel_size),   Image.Resampling.NEAREST)

    pixelated_image = small_image.resize(image.size,   Image.Resampling.NEAREST)

    return pixelated_image



def flip_and_stretch(image, flip_direction, stretch_value):

    # Inverti l'immagine in base alla direzione specificata
     # Calcola le nuove dimensioni dell'immagine con stretching
    original_width, original_height = image.size

    if flip_direction == "h":
        flipped_image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        stretched_height = original_height
        stretched_width = stretch_value
    elif flip_direction == "v":
        flipped_image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        stretched_width = original_width
        stretched_height = stretch_value
    else:
        raise ValueError("La direzione specificata non è valida. Utilizzare 'orizzontale' o 'verticale'.")

   
    
    

    # "Stretcha" l'immagine alle nuove dimensioni
    stretched_image = flipped_image.resize((stretched_width, stretched_height))

    return stretched_image


def create_noise_image(width, height):

    noise_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

    # Crea un'immagine PIL utilizzando i valori dei pixel generati
    noise_image = Image.fromarray(noise_array)

    return noise_image


def blend_images(image1, image2, blend_percentage):



    # Assicurati che le due immagini abbiano le stesse dimensioni
    if image1.size != image2.size:
        raise ValueError("Le dimensioni delle due immagini devono essere uguali.")

    # Blend delle due immagini in base alla percentuale specificata
    blended_image = Image.blend(image1, image2, blend_percentage)

    return blended_image


def image_paste(main_image, image_to_paste,side):


    # Ottieni le dimensioni delle immagini
    width_main, height_main = main_image.size
    width_paste, height_paste  = image_to_paste.size




    # Calcola le coordinate di incollaggio in base alla posizione desiderata
    if side == "t":
        # Crea una nuova immagine che sarà la combinazione delle due immagini
        new_width = width_main
        new_height = height_main + height_paste

        new_image = Image.new("RGB", (new_width, new_height))
        new_image.paste(image_to_paste, (0,0))
        new_image.paste(main_image, (0,height_paste))
    
    elif side == "b":
        new_width = width_main
        new_height = height_main + height_paste

        new_image = Image.new("RGB", (new_width, new_height))
        new_image.paste(image_to_paste, (0,height_main)) 
        new_image.paste(main_image, (0,0))
    
    elif side == "r":
        new_width = width_main + width_paste
        new_height = height_main
        new_image = Image.new("RGB", (new_width, new_height))
        new_image.paste(image_to_paste, (width_main, 0))
        new_image.paste(main_image, (0, 0))
    
        
    elif side == "l":
        new_width = width_main + width_paste
        new_height = height_main
      
        new_image = Image.new("RGB", (new_width, new_height))
        new_image.paste(image_to_paste, (0, 0))
        new_image.paste(main_image, (width_paste, 0))
    

    return new_image
  


def resize_image(image,noise,pixel_size, pixel_to_copy, left, right, top, bottom, temperature=5.0,hue=0,brightness=32,contrast=0,saturation=0,gamma=2):
    

    # RIGHT SIDE
    if right != 0:
        r_image = extract_pixels(image, "r", pixel_to_copy)
        r_image= flip_and_stretch(r_image, "h", right)
        r_image = make_pixelated(r_image, pixel_size)
        r_noise = create_noise_image(r_image.size[0], r_image.size[1])
        r_image = blend_images(r_image, r_noise, noise)
        r_image = color_correct(r_image, temperature,hue,brightness,contrast,saturation,gamma)
        r_image= Image.fromarray(r_image)

        r_image = image_paste(image, r_image,"r")
    else:
        r_image = image

    # LEFT SIDE
    if left != 0:
        l_image = extract_pixels(r_image, "l", pixel_to_copy)
        l_image = flip_and_stretch(l_image, "h", left)
        l_image = make_pixelated(l_image, pixel_size)
        l_noise = create_noise_image(l_image.size[0], l_image.size[1])
        l_image = blend_images(l_image, l_noise, noise)
        l_image = color_correct(l_image, temperature,hue,brightness,contrast,saturation,gamma)
        l_image= Image.fromarray(l_image)

        l_image = image_paste(r_image, l_image,"l")
    else:
        l_image = r_image

    # TOP
    if top != 0:
        t_image = extract_pixels(l_image, "t", pixel_to_copy)
        t_image = flip_and_stretch(t_image, "v", top)
        t_image = make_pixelated(t_image, pixel_size)
        t_noise = create_noise_image(t_image.size[0], t_image.size[1])
        t_image = blend_images(t_image, t_noise, noise)
        t_image = color_correct(t_image, temperature,hue,brightness,contrast,saturation,gamma)
        t_image= Image.fromarray(t_image)

        t_image = image_paste(l_image, t_image,"t")
    else:
        t_image = l_image

    # BOTTOM
    if bottom != 0:
        b_image = extract_pixels(t_image, "b", pixel_to_copy)
        b_image = flip_and_stretch(b_image, "v", bottom)
        b_image = make_pixelated(b_image, pixel_size)
        b_noise = create_noise_image(b_image.size[0], b_image.size[1])
        b_image = blend_images(b_image, b_noise, noise)
        b_image = color_correct(b_image, temperature,hue,brightness,contrast,saturation,gamma)
        b_image= Image.fromarray(b_image)
        
        b_image = image_paste(t_image, b_image,"b")
    else:
        b_image = t_image

    



    final_image = b_image

    return final_image


  


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
                "noise": ("FLOAT", {"default": 0.1, "min": 0, "max": 1.0, "step": 0.01}),
                "pixel_size": ("INT", {"default": 8, "min": 8, "max": 64, "step": 8}),
                "pixel_to_copy": ("INT", {"default": 32, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "temperature": ("FLOAT",{"default": 0, "min": -100, "max": 100, "step": 5},),
                "hue": ("FLOAT", {"default": 0, "min": -90, "max": 90, "step": 5}),
                "brightness": ("FLOAT",{"default": 0, "min": -100, "max": 100, "step": 5},),
                "contrast": ("FLOAT",{"default": 0, "min": -100, "max": 100, "step": 5},),
                "saturation": ("FLOAT",{"default": 0, "min": -100, "max": 100, "step": 5},),
                "gamma": ("FLOAT", {"default": 1, "min": 0.2, "max": 2.2, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "expand_image"

    CATEGORY = "image"
    

    def expand_image(self,image,feathering,noise,pixel_size, pixel_to_copy, left, right, top, bottom, temperature=5.0,hue=0,brightness=32,contrast=0,saturation=0,gamma=2):
        d1, d2, d3, d4 = image.size()
   

        #new_image = torch.zeros(
        #    (d1, d2 + top + bottom, d3 + left + right, d4),
        #    dtype=torch.float32,
        #)
        #new_image[:, top:top + d2, left:left + d3, :] = image

        image = tensor2pil(image)
        #image = Image.fromarray(image.astype(np.uint8))
        
        new_image = resize_image(image,noise,pixel_size, pixel_to_copy, left, right, top, bottom, temperature,hue,brightness,contrast,saturation,gamma)



        i = ImageOps.exif_transpose(new_image)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        new_image = torch.from_numpy(image)[None,]




        mask = torch.ones(
            (d2 + top + bottom, d3 + left + right),
            dtype=torch.float32,
        )

        t = torch.zeros(
            (d2, d3),
            dtype=torch.float32
        )

        if feathering > 0 and feathering * 2 < d2 and feathering * 2 < d3:

            for i in range(d2):
                for j in range(d3):
                    dt = i if top != 0 else d2
                    db = d2 - i if bottom != 0 else d2

                    dl = j if left != 0 else d3
                    dr = d3 - j if right != 0 else d3

                    d = min(dt, db, dl, dr)

                    if d >= feathering:
                        continue

                    v = (feathering - d) / feathering

                    t[i, j] = v * v

        mask[top:top + d2, left:left + d3] = t

        return (new_image, mask)
    

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ImagePadForOutpaintAdvanced [n-suite]": ImagePadForOutpaintAdvanced
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImagePadForOutpaintAdvanced [n-suite]": "Image Pad For Outpainting Advanced [🅝-🅢🅤🅘🅣🅔]"
}
