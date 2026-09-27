"""Generate the all-node smoke test from a running ComfyUI instance.

Usage: python examples/generate_test_workflow.py http://127.0.0.1:8188
"""

import json
import sys
import uuid
from pathlib import Path
from urllib.request import urlopen


url = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8188"
schema = json.load(urlopen(f"{url}/object_info"))
nodes = []
links = []


def add(kind, pos, values=None, title=None, size=None):
    info = schema[kind]
    values = values or {}
    inputs, widgets = [], []
    for name, spec in {**info["input"].get("required", {}), **info["input"].get("optional", {})}.items():
        raw_type = spec[0]
        input_type = "COMBO" if isinstance(raw_type, list) else raw_type
        options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
        entry = {"name": name, "type": input_type, "link": None}
        if input_type in ("COMBO", "STRING", "INT", "FLOAT", "BOOLEAN") and not options.get("forceInput"):
            entry["widget"] = {"name": name}
            default = options.get("default", raw_type[0] if isinstance(raw_type, list) and raw_type else "")
            widgets.append(values.get(name, default))
        inputs.append(entry)
    if kind == "LoadImage":
        inputs.append({"name": "upload", "type": "IMAGEUPLOAD", "widget": {"name": "upload"}, "link": None})
        widgets.append("image")
    node_id = len(nodes) + 1
    node = {
        "id": node_id, "type": kind, "pos": pos, "size": size or [350, 180],
        "flags": {}, "order": node_id - 1, "mode": 0, "inputs": inputs,
        "outputs": [{"name": name, "type": typ, "links": []} for name, typ in
                    zip(info.get("output_name", info["output"]), info["output"])],
        "properties": {"Node name for S&R": kind}, "widgets_values": widgets,
    }
    if title:
        node["title"] = title
    nodes.append(node)
    return node_id


def connect(source, slot, target, input_name):
    origin = nodes[source - 1]
    dest = nodes[target - 1]
    dest_slot = next(i for i, item in enumerate(dest["inputs"]) if item["name"] == input_name)
    assert dest["inputs"][dest_slot]["link"] is None
    link_id = len(links) + 1
    links.append([link_id, source, slot, target, dest_slot, origin["outputs"][slot]["type"]])
    origin["outputs"][slot]["links"].append(link_id)
    dest["inputs"][dest_slot]["link"] = link_id


image = add("LoadImage", [80, 100], {"image": "example.png"}, "Scegli la tua immagine", [380, 330])
questions = add("String Variable [n-suite]", [80, 500],
                {"string": "What is in this image?,What colors are in this image?"}, "Domande di prova")
dynamic = add("DynamicPrompt [n-suite]", [520, 490],
              {"cached": "NO", "number_of_random_tag": "Fixed", "fixed_number_of_random_tag": 1})
caption_model = add("GPT Loader Simple [n-suite]", [520, 100], {"ckpt_name": "moondream"})
caption = add("GPT Sampler [n-suite]", [930, 100],
              {"max_tokens": 128, "cached": "NO", "print_output": "enable"}, size=[390, 700])
caption_preview = add("PreviewAny", [1400, 150], title="Risposta Moondream")
noise = add("Float Variable [n-suite]", [80, 1040], {"value": 0.1})
pad = add("ImagePadForOutpaintAdvanced [n-suite]", [500, 930],
          {"left": 32, "right": 32, "top": 32, "bottom": 32}, size=[430, 590])
padded_preview = add("PreviewImage", [1030, 970], title="Immagine con bordo", size=[350, 300])
mask_to_image = add("MaskToImage", [1030, 1330])
mask_preview = add("PreviewImage", [1410, 1320], title="Maschera del bordo", size=[350, 300])
clip = add("CLIPLoader", [2030, 100], {"clip_name": "clip_l.safetensors", "type": "stable_diffusion"})
encode = add("CLIPTextEncodeAdvancedNSuite [n-suite]", [2460, 100],
             {"styles": "NAI", "positive_prompt": "a small test image", "negative_prompt": "blurry"}, size=[400, 350])
positive_preview = add("PreviewAny", [2940, 100], title="Condizionamento positivo")
negative_preview = add("PreviewAny", [2940, 400], title="Condizionamento negativo")
multiplier = add("Integer Variable [n-suite]", [80, 2060], {"value": 2})
video = add("LoadVideo [n-suite]", [430, 1950],
            {"video": "SELECT_VIDEO.mp4", "framerate": "original", "resize_by": "none",
             "images_limit": 0, "batch_size": 0, "starting_frame": 0, "autoplay": False, "use_ram": False},
            size=[420, 570])
interpolator = add("FrameInterpolator [n-suite]", [940, 1990])
interpolated_video = add("SaveVideo [n-suite]", [1430, 1970],
                         {"SaveVideo": True, "SaveFrames": False, "filename_prefix": "n_suite_test_interpolated"})
video_info = add("PreviewAny", [940, 2290], title="Metadati video originale")
folder = add("String Variable [n-suite]", [80, 2990],
             {"string": "/workspace/ComfyUI/input/n-suite/test_frames"}, "Cartella frame: cambia qui", [500, 130])
image_folder = add("LoadImageFromFolder [n-suite]", [660, 2860])
image_folder_preview = add("PreviewImage", [1100, 2840], title="Immagini dalla cartella", size=[330, 260])
manual_metadata = add("SetMetadataForSaveVideo [n-suite]", [1100, 3210],
                      {"fps": 24, "VideoName": "n_suite_folder"})
manual_video = add("SaveVideo [n-suite]", [1550, 2890],
                   {"SaveVideo": True, "SaveFrames": False, "filename_prefix": "n_suite_test_manual_metadata"})
frame_folder = add("LoadFramesFromFolder [n-suite]", [660, 3530], {"fps": 24})
frame_folder_preview = add("PreviewImage", [1100, 3520], title="Frame numerati", size=[330, 260])
frame_video = add("SaveVideo [n-suite]", [1550, 3510],
                  {"SaveVideo": True, "SaveFrames": False, "filename_prefix": "n_suite_test_folder_frames"})

for args in [
    (questions, 0, dynamic, "variable_prompt"), (dynamic, 0, caption, "prompt"),
    (caption_model, 0, caption, "model"), (image, 0, caption, "image"),
    (caption, 0, caption_preview, "source"), (image, 0, pad, "image"),
    (noise, 0, pad, "noise"), (pad, 0, padded_preview, "images"),
    (pad, 1, mask_to_image, "mask"), (mask_to_image, 0, mask_preview, "images"),
    (clip, 0, encode, "clip"), (encode, 0, positive_preview, "source"),
    (encode, 1, negative_preview, "source"), (video, 0, interpolator, "images"),
    (video, 2, interpolator, "METADATA"), (multiplier, 0, interpolator, "multiplier"),
    (interpolator, 0, interpolated_video, "images"),
    (interpolator, 1, interpolated_video, "METADATA"), (video, 2, video_info, "source"),
    (folder, 0, image_folder, "folder"), (folder, 0, frame_folder, "folder"),
    (image_folder, 0, image_folder_preview, "images"),
    (image_folder, 0, manual_video, "images"),
    (image_folder, 3, manual_metadata, "number_of_frames"),
    (manual_metadata, 0, manual_video, "METADATA"),
    (frame_folder, 0, frame_folder_preview, "images"),
    (frame_folder, 0, frame_video, "images"),
    (frame_folder, 1, frame_video, "METADATA"),
]:
    connect(*args)

groups = [
    ("01 FOTO + MOONDREAM: scegli la foto in LoadImage", [40, 40, 1750, 790], "#3f789e"),
    ("02 IMAGE PAD: controlla immagine e maschera", [40, 870, 1760, 790], "#637c49"),
    ("03 CLIP: usa il modello clip_l presente", [1980, 40, 1400, 680], "#76578e"),
    ("04 VIDEO: copia un MP4 in input/n-suite, ricarica e selezionalo", [40, 1880, 1790, 690], "#896a3c"),
    ("05 CARTELLA: aggiungi 0001.png e 0002.png in test_frames", [40, 2780, 1920, 1050], "#3f789e"),
]
used = {node["type"] for node in nodes if "[n-suite]" in node["type"].lower()}
expected = {name for name in schema if "[n-suite]" in name.lower()}
assert used == expected, f"Missing N-Suite nodes: {sorted(expected - used)}"
workflow = {
    "id": str(uuid.uuid4()), "revision": 0, "last_node_id": len(nodes), "last_link_id": len(links),
    "nodes": nodes, "links": links,
    "groups": [{"id": i, "title": title, "bounding": bounds, "color": color, "flags": {}}
               for i, (title, bounds, color) in enumerate(groups, 1)],
    "config": {}, "extra": {"ds": {"scale": 0.55, "offset": [70, 70]}}, "version": 0.4,
}
destination = Path(__file__).with_name("N-Suite-all-nodes-test.json")
destination.write_text(json.dumps(workflow, ensure_ascii=False, indent=2) + "\n")
print(f"Saved {destination}: {len(nodes)} nodes, {len(links)} links, {len(used)} N-Suite types")
