from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info
import argparse
import json
import torch


def eval_model(args):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    with open(args.question_file, "r") as f:
        questions_list = json.load(f)

    for questions in questions_list:
        image_file = questions["image"][0]

        input_ids_old = None

        for question_idx, question in enumerate(questions["question"]):
            if question_idx > 0:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                        ],
                    }
                ]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                # 只取字符串的 "<|im_start|>user" 到 末尾的内容
                text = text[text.find("<|im_start|>user"):]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_file},
                            {"type": "text", "text": question},
                        ],
                    }
                ]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            if input_ids_old is not None:
                inputs["input_ids"] = torch.cat((input_ids_old, inputs["input_ids"]), dim=1)
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"]).cuda()

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            input_ids_old = torch.cat((inputs.input_ids, generated_ids_trimmed[0].unsqueeze(0)), dim=1)
            
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            print("--------------------------------")
            print(f"question_idx: {question_idx}")
            print(f"question: {question}")
            print(f"outputs: {output_text}")
            print("--------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)
