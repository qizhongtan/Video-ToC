# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import yaml
import json
import random
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from qwen_vl_utils import process_vision_info

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


class ImageProcessMixin:
    max_pixels: int
    min_pixels: int

    def process_image(self, image: Union[Dict[str, Any], ImageObject]) -> ImageObject:
        if isinstance(image, dict):
            image = Image.open(BytesIO(image["bytes"]))
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))

        if (image.width * image.height) > self.max_pixels:
            resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < self.min_pixels:
            resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image


class RLHFDataset(Dataset, ImageProcessMixin):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        data_folder: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: str = None,
        max_pixels: int = None,
        min_pixels: int = None,
        num_frames: int = None
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.format_prompt = format_prompt
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.num_frames = num_frames
        self.data_folder = data_folder
        self.dataset = []

        with open(data_path, "r") as file:
            yaml_data = yaml.safe_load(file)
            datasets = yaml_data.get("datasets")
            # file should be in the format of:
            # datasets:
            #   - json_path: xxxx1.json
            #     sampling_strategy: first:1000
            #   - json_path: xxxx2.json
            #     sampling_strategy: end:3000
            #   - json_path: xxxx3.json
            #     sampling_strategy: random:999
            for dataset in datasets:
                json_path = dataset.get("json_path")
                sampling_strategy = dataset.get("sampling_strategy", "all")
                sampling_number = None

                print(f"Loading {json_path} with {sampling_strategy} sampling strategy")

                if json_path.endswith(".jsonl"):
                    cur_data_dict = []
                    with open(json_path, "r") as json_file:
                        for line in json_file:
                            cur_data_dict.append(json.loads(line.strip()))
                elif json_path.endswith(".json"):
                    with open(json_path, "r") as json_file:
                        cur_data_dict = json.load(json_file)
                else:
                    raise ValueError(f"Unsupported file type: {json_path}")

                if ":" in sampling_strategy:
                    sampling_strategy, sampling_number = sampling_strategy.split(":")
                    if "%" in sampling_number:
                        sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                    else:
                        sampling_number = int(sampling_number)

                # Apply the sampling strategy
                if sampling_strategy == "first" and sampling_number is not None:
                    cur_data_dict = cur_data_dict[:sampling_number]
                elif sampling_strategy == "end" and sampling_number is not None:
                    cur_data_dict = cur_data_dict[-sampling_number:]
                elif sampling_strategy == "random" and sampling_number is not None:
                    random.shuffle(cur_data_dict)
                    cur_data_dict = cur_data_dict[:sampling_number]

                print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                self.dataset.extend(cur_data_dict)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_item = self.dataset[index]
        row_dict = dict()
        video_path = os.path.join(self.data_folder, data_item['videos'])
        problem = data_item["question"]
        messages = [{
            "role": "user",
            "content": [{"type": "video", "max_pixels": self.max_pixels, "nframes": self.num_frames, "video": video_path},
                        {"type": "text", "text": problem}]
            }]
        row_dict['problem_type'] = 'mc'
        
        prompt = maybe_apply_chat_template({'prompt': messages}, self.processor)["prompt"]

        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        row_dict["multi_modal_data"] = {'video': video_inputs}
        
        model_inputs = self.processor(
            text=prompt,
            videos=video_inputs,
            return_tensors="pt",
        )
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
        row_dict['multi_modal_inputs'] = dict(model_inputs)
        if "video_grid_thw" in model_inputs:
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                video_grid_thw=model_inputs["video_grid_thw"],
                second_per_grid_ts=model_inputs["second_per_grid_ts"],
                attention_mask=attention_mask,
            )
        elif "image_grid_thw" in model_inputs:
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs["image_grid_thw"],
                attention_mask=attention_mask,
            )

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(prompt, add_special_tokens=False)
        row_dict["ground_truth"] = data_item["answer"]
        row_dict["grade"] = data_item["grade"]
        return row_dict