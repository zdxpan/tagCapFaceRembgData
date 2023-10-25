#
# demo from https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags/blob/main/app.py
#

# DanBooru IMage Utility functions
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

import argparse
import functools
import html
import os

import gradio as gr
import huggingface_hub
import numpy as np
import onnxruntime as rt
import pandas as pd
import piexif
import piexif.helper
import PIL.Image

# from Utils import dbimutils form Utils import fuc*

def smart_imread(img, flag=cv2.IMREAD_UNCHANGED):
    if img.endswith(".gif"):
        img = Image.open(img)
        img = img.convert("RGB")
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(img, flag)
    return img


def smart_24bit(img):
    if img.dtype is np.dtype(np.uint16):
        img = (img / 257).astype(np.uint8)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        trans_mask = img[:, :, 3] == 0
        img[trans_mask] = [255, 255, 255, 255]
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def make_square(img, target_size):
    old_size = img.shape[:2]
    desired_size = max(old_size)
    desired_size = max(desired_size, target_size)

    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]
    new_im = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return new_im


def smart_resize(img, size):
    # Assumes the image has already gone through make_square
    if img.shape[0] > size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    elif img.shape[0] < size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return img




TITLE = "WaifuDiffusion v1.4 Tags"
DESCRIPTION = """
Demo for:
- [SmilingWolf/wd-v1-4-moat-tagger-v2](https://huggingface.co/SmilingWolf/wd-v1-4-moat-tagger-v2)
- [SmilingWolf/wd-v1-4-swinv2-tagger-v2](https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger-v2)
- [SmilingWolf/wd-v1-4-convnext-tagger-v2](https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger-v2)
- [SmilingWolf/wd-v1-4-convnextv2-tagger-v2](https://huggingface.co/SmilingWolf/wd-v1-4-convnextv2-tagger-v2)
- [SmilingWolf/wd-v1-4-vit-tagger-v2](https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger-v2)
Includes "ready to copy" prompt and a prompt analyzer.
Modified from [NoCrypt/DeepDanbooru_string](https://huggingface.co/spaces/NoCrypt/DeepDanbooru_string)  
Modified from [hysts/DeepDanbooru](https://huggingface.co/spaces/hysts/DeepDanbooru)
PNG Info code forked from [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
Example image by [ほし☆☆☆](https://www.pixiv.net/en/users/43565085)
"""

# HF_TOKEN = os.environ["HF_TOKEN"]
MOAT_MODEL_REPO = "SmilingWolf/wd-v1-4-moat-tagger-v2"
MOAT_MODEL_REPO = "/home/dell/workspace/dataset/caption/wd-v14-moat-tagger-v2"

SWIN_MODEL_REPO = "SmilingWolf/wd-v1-4-swinv2-tagger-v2"
CONV_MODEL_REPO = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
CONV2_MODEL_REPO = "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"
VIT_MODEL_REPO = "SmilingWolf/wd-v1-4-vit-tagger-v2"
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-slider-step", type=float, default=0.05)
    parser.add_argument("--score-general-threshold", type=float, default=0.35)
    parser.add_argument("--score-character-threshold", type=float, default=0.85)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def load_model(model_repo: str, model_filename: str) -> rt.InferenceSession:
    if model_repo.startswith("/"):
        path = os.path.join(model_repo, model_filename)
    else:
        path = huggingface_hub.hf_hub_download(
            model_repo, model_filename, 
            # use_auth_token=HF_TOKEN
        )
    print(f">> load model :{path}")
    # model = rt.InferenceSession(path)
    model = rt.InferenceSession(path,providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])

    return model


def change_model(model_name):
    global loaded_models

    if model_name == "MOAT":
        model = load_model(MOAT_MODEL_REPO, MODEL_FILENAME)
    elif model_name == "SwinV2":
        model = load_model(SWIN_MODEL_REPO, MODEL_FILENAME)
    elif model_name == "ConvNext":
        model = load_model(CONV_MODEL_REPO, MODEL_FILENAME)
    elif model_name == "ConvNextV2":
        model = load_model(CONV2_MODEL_REPO, MODEL_FILENAME)
    elif model_name == "ViT":
        model = load_model(VIT_MODEL_REPO, MODEL_FILENAME)

    loaded_models[model_name] = model
    return loaded_models[model_name]


def load_labels() -> list[str]:
    if MOAT_MODEL_REPO.startswith("/"):
        path = os.path.join(MOAT_MODEL_REPO, LABEL_FILENAME)
    else:
        path = huggingface_hub.hf_hub_download(
            MOAT_MODEL_REPO, LABEL_FILENAME, 
            # use_auth_token=HF_TOKEN
        )
    df = pd.read_csv(path)

    tag_names = df["name"].tolist()
    rating_indexes = list(np.where(df["category"] == 9)[0])
    general_indexes = list(np.where(df["category"] == 0)[0])
    character_indexes = list(np.where(df["category"] == 4)[0])
    return tag_names, rating_indexes, general_indexes, character_indexes


def plaintext_to_html(text):
    text = (
        "<p>" + "<br>\n".join([f"{html.escape(x)}" for x in text.split("\n")]) + "</p>"
    )
    return text


def predict(
    image: PIL.Image.Image,
    model_name: str,
    general_threshold: float,
    character_threshold: float,
    tag_names: list[str],
    rating_indexes: list[np.int64],
    general_indexes: list[np.int64],
    character_indexes: list[np.int64],
):
    global loaded_models

    rawimage = image

    model = loaded_models[model_name]
    if model is None:
        model = change_model(model_name)

    _, height, width, _ = model.get_inputs()[0].shape

    # Alpha to white
    image = image.convert("RGBA")
    new_image = PIL.Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, mask=image)
    image = new_image.convert("RGB")
    image = np.asarray(image)

    # PIL RGB to OpenCV BGR
    image = image[:, :, ::-1]

    image = make_square(image, height)
    image = smart_resize(image, height)
    image = image.astype(np.float32)
    image = np.expand_dims(image, 0)

    input_name = model.get_inputs()[0].name
    label_name = model.get_outputs()[0].name
    probs = model.run([label_name], {input_name: image})[0]

    labels = list(zip(tag_names, probs[0].astype(float)))

    # First 4 labels are actually ratings: pick one with argmax
    ratings_names = [labels[i] for i in rating_indexes]
    rating = dict(ratings_names)

    # Then we have general tags: pick any where prediction confidence > threshold
    general_names = [labels[i] for i in general_indexes]
    general_res = [x for x in general_names if x[1] > general_threshold]
    general_res = dict(general_res)

    # Everything else is characters: pick any where prediction confidence > threshold
    character_names = [labels[i] for i in character_indexes]
    character_res = [x for x in character_names if x[1] > character_threshold]
    character_res = dict(character_res)

    b = dict(sorted(general_res.items(), key=lambda item: item[1], reverse=True))
    a = (
        ", ".join(list(b.keys()))
        .replace("_", " ")
        .replace("(", "\(")
        .replace(")", "\)")
    )
    c = ", ".join(list(b.keys()))

    items = rawimage.info
    geninfo = ""

    if "exif" in rawimage.info:
        exif = piexif.load(rawimage.info["exif"])
        exif_comment = (exif or {}).get("Exif", {}).get(piexif.ExifIFD.UserComment, b"")
        try:
            exif_comment = piexif.helper.UserComment.load(exif_comment)
        except ValueError:
            exif_comment = exif_comment.decode("utf8", errors="ignore")

        items["exif comment"] = exif_comment
        geninfo = exif_comment

        for field in [
            "jfif",
            "jfif_version",
            "jfif_unit",
            "jfif_density",
            "dpi",
            "exif",
            "loop",
            "background",
            "timestamp",
            "duration",
        ]:
            items.pop(field, None)

    geninfo = items.get("parameters", geninfo)

    info = f"""
<p><h4>PNG Info</h4></p>    
"""
    for key, text in items.items():
        info += (
            f"""
<div>
<p><b>{plaintext_to_html(str(key))}</b></p>
<p>{plaintext_to_html(str(text))}</p>
</div>
""".strip()
            + "\n"
        )

    if len(info) == 0:
        message = "Nothing found in the image."
        info = f"<div><p>{message}<p></div>"

    return (a, c, rating, character_res, general_res, info)

def predictv2(
    image: PIL.Image.Image,
    model: str,
    general_threshold: float,
    character_threshold: float,
    tag_names: list[str],
    rating_indexes: list[np.int64],
    general_indexes: list[np.int64],
    character_indexes: list[np.int64],
):

    rawimage = image

    assert model is not None

    _, height, width, _ = model.get_inputs()[0].shape

    # Alpha to white
    image = image.convert("RGBA")
    new_image = PIL.Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, mask=image)
    image = new_image.convert("RGB")
    image = np.asarray(image)

    # PIL RGB to OpenCV BGR
    image = image[:, :, ::-1]

    image = make_square(image, height)
    image = smart_resize(image, height)
    image = image.astype(np.float32)
    image = np.expand_dims(image, 0)

    input_name = model.get_inputs()[0].name
    label_name = model.get_outputs()[0].name
    probs = model.run([label_name], {input_name: image})[0]

    labels = list(zip(tag_names, probs[0].astype(float)))

    # First 4 labels are actually ratings: pick one with argmax
    ratings_names = [labels[i] for i in rating_indexes]
    rating = dict(ratings_names)

    # Then we have general tags: pick any where prediction confidence > threshold
    general_names = [labels[i] for i in general_indexes]
    general_res = [x for x in general_names if x[1] > general_threshold]
    general_res = dict(general_res)

    # Everything else is characters: pick any where prediction confidence > threshold
    character_names = [labels[i] for i in character_indexes]
    character_res = [x for x in character_names if x[1] > character_threshold]
    character_res = dict(character_res)

    b = dict(sorted(general_res.items(), key=lambda item: item[1], reverse=True))
    a = (
        ", ".join(list(b.keys()))
        .replace("_", " ")
        .replace("(", "\(")
        .replace(")", "\)")
    )
    c = ", ".join(list(b.keys()))

    items = rawimage.info
    geninfo = ""

    if "exif" in rawimage.info:
        exif = piexif.load(rawimage.info["exif"])
        exif_comment = (exif or {}).get("Exif", {}).get(piexif.ExifIFD.UserComment, b"")
        try:
            exif_comment = piexif.helper.UserComment.load(exif_comment)
        except ValueError:
            exif_comment = exif_comment.decode("utf8", errors="ignore")

        items["exif comment"] = exif_comment
        geninfo = exif_comment

        for field in [
            "jfif",
            "jfif_version",
            "jfif_unit",
            "jfif_density",
            "dpi",
            "exif",
            "loop",
            "background",
            "timestamp",
            "duration",
        ]:
            items.pop(field, None)

    geninfo = items.get("parameters", geninfo)

    info = f"""
<p><h4>PNG Info</h4></p>    
"""
    for key, text in items.items():
        info += (
            f"""
<div>
<p><b>{plaintext_to_html(str(key))}</b></p>
<p>{plaintext_to_html(str(text))}</p>
</div>
""".strip()
            + "\n"
        )

    if len(info) == 0:
        message = "Nothing found in the image."
        info = f"<div><p>{message}<p></div>"

    return (a, c, rating, character_res, general_res, info)

def main():
    global loaded_models
    loaded_models = {
        "MOAT": None,
        "SwinV2": None,
        "ConvNext": None,
        "ConvNextV2": None,
        "ViT": None,
    }

    args = parse_args()

    change_model("MOAT")

    tag_names, rating_indexes, general_indexes, character_indexes = load_labels()

    func = functools.partial(
        predict,
        tag_names=tag_names,
        rating_indexes=rating_indexes,
        general_indexes=general_indexes,
        character_indexes=character_indexes,
    )

    gr.Interface(
        fn=func,
        inputs=[
            gr.Image(type="pil", label="Input"),
            gr.Radio(
                ["MOAT", "SwinV2", "ConvNext", "ConvNextV2", "ViT"],
                value="MOAT",
                label="Model",
            ),
            gr.Slider(
                0,
                1,
                step=args.score_slider_step,
                value=args.score_general_threshold,
                label="General Tags Threshold",
            ),
            gr.Slider(
                0,
                1,
                step=args.score_slider_step,
                value=args.score_character_threshold,
                label="Character Tags Threshold",
            ),
        ],
        outputs=[
            gr.Textbox(label="Output (string)"),
            gr.Textbox(label="Output (raw string)"),
            gr.Label(label="Rating"),
            gr.Label(label="Output (characters)"),
            gr.Label(label="Output (tags)"),
            gr.HTML(),
        ],
        examples=[["../girl100/a1/sa1.jpeg", "MOAT", 0.35, 0.85]],
        title=TITLE,
        description=DESCRIPTION,
        allow_flagging="never",
    ).launch(
        server_name="0.0.0.0",
        enable_queue=True,
        server_port = 7655,
        share=args.share,
    )


if __name__ == "__main__":
    main()
