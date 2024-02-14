import random
import folder_paths
import os
import json
import csv
import server
from aiohttp import web
_choice = ["YES", "NO"]
_range = ["Fixed", "Random"]


def loadCSVStyle():
    csv_dir = os.path.join(folder_paths.base_path,"styles")
    csv_path = os.path.join(csv_dir,"n-styles.csv")
    #make directory if it doesn't exist
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)


    styles = []
    if os.path.exists(csv_path):
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                styles.append(row)
    else:
        #create a file containing name,prompt,negative_prompt\n
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("name,prompt,negative_prompt\n")
            f.write('NAI,"masterpiece, best quality, masterpiece, asuka langley sitting cross legged on a chair","lowres, bad anatomy, bad hands"\n')
            styles.append({'name': 'NAI', 'prompt': 'masterpiece, best quality, masterpiece, asuka langley sitting cross legged on a chair', 'negative_prompt': 'lowres, bad anatomy, bad hands'})

    if len(styles) == 0:
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("name,prompt,negative_prompt\n")
            f.write('NAI,"masterpiece, best quality, masterpiece, asuka langley sitting cross legged on a chair","lowres, bad anatomy, bad hands"\n')


        styles.append({'name': 'NAI', 'prompt': 'masterpiece, best quality, masterpiece, asuka langley sitting cross legged on a chair', 'negative_prompt': 'lowres, bad anatomy, bad hands'})
    return (styles, )
    



def addStyle(name, positive_prompt, negative_prompt):
    csv_dir = os.path.join(folder_paths.base_path,"styles","n-styles.csv")
    #make directory if it doesn't exist
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    
    #backup file
    backup_dir = os.path.join(folder_paths.base_path,"styles","n-styles-backup.csv")
    if os.path.exists(backup_dir):
        os.remove(backup_dir)
    os.rename(csv_dir, backup_dir)

    #edit style if it already exists else add it

    with open(backup_dir, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(csv_dir, "w", encoding="utf-8") as f:
        for line in lines:
            name_style = line.split(",")[0]
            if name == name_style:
                f.write(f'{name},"{positive_prompt}","{negative_prompt}"\n')
                continue
            f.write(line)
        f.write(f'{name},"{positive_prompt}","{negative_prompt}"\n')




styles = loadCSVStyle()

def deleteStyle(name):
    csv_dir = os.path.join(folder_paths.base_path,"styles","n-styles.csv")
    #make directory if it doesn't exist
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    #backup file
    backup_dir = os.path.join(folder_paths.base_path,"styles","n-styles-backup.csv")
    if os.path.exists(backup_dir):
        os.remove(backup_dir)
    os.rename(csv_dir, backup_dir)

    with open(csv_dir, "r", encoding="utf-8") as f:
        lines = f.readlines()
    

    with open(backup_dir, "w", encoding="utf-8") as f:
        for line in lines:
            if name in line:
                continue
            f.write(line)

@server.PromptServer.instance.routes.get("/nsuite/styles" )
async def style_get(request):
    result = {"styles": loadCSVStyle()}
    return web.json_response(result, content_type='application/json')



@server.PromptServer.instance.routes.post("/nsuite/styles/add" )
async def style_add(request):
    data = await request.json()
    name = data["name"]
    positive_prompt = data["positive_prompt"]
    negative_prompt = data["negative_prompt"]
    addStyle(name, positive_prompt, negative_prompt)

    result = {"error": "none"}
    return web.json_response(result, content_type='application/json')


@server.PromptServer.instance.routes.post("/nsuite/styles/update" )
async def style_add(request):
    data = await request.json()
    name = data["name"]
    positive_prompt = data["positive_prompt"]
    negative_prompt = data["negative_prompt"]
    addStyle(name, positive_prompt, negative_prompt)

    result = {"error": "none"}
    return web.json_response(result, content_type='application/json')


@server.PromptServer.instance.routes.post("/nsuite/styles/remove" )
async def style_delete(request):
    data = await request.json()
    name = data["name"]

    deleteStyle(name)
    result = {"error": "none"}
    return web.json_response(result, content_type='application/json')



class CLIPTextEncodeAdvancedNSuite:
 
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                {   "styles": ([x['name'] for x in styles[0]],),
                    "positive_prompt": ("STRING", {"multiline": True}), 
                    "negative_prompt": ("STRING", {"multiline": True}), 
                    "clip": ("CLIP", )}
                }
    RETURN_TYPES = ("CONDITIONING","CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "encode"

    CATEGORY = "N-Suite/Conditioning"

    def encode(self, clip, positive_prompt, negative_prompt,styles):
        p_tokens = clip.tokenize(positive_prompt)
        n_tokens = clip.tokenize(negative_prompt)
        p_cond, p_pooled = clip.encode_from_tokens(p_tokens, return_pooled=True)
        n_cond, n_pooled = clip.encode_from_tokens(n_tokens, return_pooled=True)
        return ([[p_cond, {"pooled_output": p_pooled}]],[[n_cond, {"pooled_output": n_pooled}]], )



# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeAdvancedNSuite [n-suite]": CLIPTextEncodeAdvancedNSuite
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTextEncodeAdvancedNSuite [n-suite]": "CLIP Text Encode Advanced [🅝-🅢🅤🅘🅣🅔]"
}
