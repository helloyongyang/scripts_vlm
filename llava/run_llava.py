import argparse
import torch
import json

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
)

from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re
import os


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args):
    
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, None, model_name, attn_implementation='sdpa'
    )

    with open(args.question_file, "r") as f:
        questions_list = json.load(f)    

    for data_idx, questions in enumerate(questions_list):
        image_files = questions["image"]
        image_files = [os.path.join(args.images_dir, image_file) for image_file in image_files]

        images = load_images(image_files)
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            image_processor,
            model.config
        ).to(model.device, dtype=torch.float16)
        
        input_ids_old = None

        for question_idx, question in enumerate(questions["question"]):
            conv_mode = "llava_v1"
            conv = conv_templates[conv_mode].copy()
            if question_idx > 0:
                conv.system = ""
                qs = question
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + question
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            # print(f"input_ids 1: {input_ids}, {input_ids.shape}")
            if input_ids_old is not None:
                input_ids = torch.cat((input_ids_old, input_ids), dim=1)
            # print(f"input_ids 2: {input_ids}, {input_ids.shape}")

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=False,
                    top_p=None,
                    num_beams=1,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )
                
                # print(f"output_ids: {output_ids}, {output_ids.shape}")

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
            
            print("--------------------------------")
            print(f"data_idx: {data_idx}")
            print(f"question_idx: {question_idx}")
            print(f"question: {question}")
            print(f"outputs: {outputs}")
            print("--------------------------------")

            input_ids_old = torch.cat((input_ids, output_ids), dim=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--images-dir", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    eval_model(args)
