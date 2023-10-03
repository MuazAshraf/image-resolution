import gradio as gr
import requests
from PIL import Image
import os
import torch
import numpy as np
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

torch.hub.download_url_to_file('https://huggingface.co/spaces/jjourney1125/swin2sr/resolve/main/samples/00003.jpg', '00003.jpg')
torch.hub.download_url_to_file('https://huggingface.co/spaces/jjourney1125/swin2sr/resolve/main/samples/0855.jpg', '0855.jpg')
torch.hub.download_url_to_file('https://huggingface.co/spaces/jjourney1125/swin2sr/resolve/main/samples/ali_eye.jpg', 'ali_eye.jpg')
torch.hub.download_url_to_file('https://huggingface.co/spaces/jjourney1125/swin2sr/resolve/main/samples/butterfly.jpg', 'butterfly.jpg')
torch.hub.download_url_to_file('https://huggingface.co/spaces/jjourney1125/swin2sr/resolve/main/samples/chain-eye.jpg', 'chain-eye.jpg')
torch.hub.download_url_to_file('https://huggingface.co/spaces/jjourney1125/swin2sr/resolve/main/samples/gojou-eyes.jpg', 'gojou-eyes.jpg')
torch.hub.download_url_to_file('https://huggingface.co/spaces/jjourney1125/swin2sr/resolve/main/samples/shanghai.jpg', 'shanghai.jpg')
torch.hub.download_url_to_file('https://huggingface.co/spaces/jjourney1125/swin2sr/resolve/main/samples/vagabond.jpg', 'vagabond.jpg')

processor = AutoImageProcessor.from_pretrained("caidas/swin2SR-classical-sr-x2-64")
model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x2-64")

def enhance(image):
    # prepare image for the model
    inputs = processor(image, return_tensors="pt")

    # forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # postprocess
    output = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.moveaxis(output, source=0, destination=-1)
    output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
    
    return Image.fromarray(output)

title = "Demo: Swin2SR for Image Super-Resolution 🚀🚀🔥"
description = ''' 
**This demo expects low-quality and low-resolution JPEG compressed images.**
**Demo notebook can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Swin2SR/Perform_image_super_resolution_with_Swin2SR.ipynb).
'''
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2209.11345' target='_blank'>Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration</a> | <a href='https://huggingface.co/docs/transformers/main/model_doc/swin2sr' target='_blank'>HuggingFace docs</a></p>"

examples = [['00003.jpg'], ['0855.jpg'], ['ali_eye.jpg'], ['butterfly.jpg'], ['chain-eye.jpg'], ['gojou-eyes.jpg'], ['shanghai.jpg'], ['vagabond.jpg']]

gr.Interface(
    enhance, 
    gr.inputs.Image(type="pil", label="Input").style(height=260),
    gr.inputs.Image(type="pil", label="Ouput").style(height=240),
    title=title,
    description=description,
    article=article,
    examples=examples,
    ).launch(enable_queue=True, share=True, host="0.0.0.0", port=7860)

