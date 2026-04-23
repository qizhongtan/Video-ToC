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

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, TypedDict
import re
import torch
from transformers import PreTrainedTokenizer
from typing import Dict
from ...protocol import DataProto
from .config import RewardConfig
import math


def format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<locate>.*?</locate>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else 0.0

def accuracy_reward(predict_str: str, ground_truth: str) -> float:
    content_match = re.search(r"<answer>(.*?)</answer>", predict_str)
    given_answer = content_match.group(1).strip() if content_match else predict_str.strip()
    if given_answer == ground_truth:
        return 1.0
    else:
        return 0.0

def compute_score(predict_str: str, ground_truth: str, difficulty: float) -> Dict[str, float]:
    accuracy_score = accuracy_reward(predict_str, ground_truth)
    return {
        "overall": accuracy_score * difficulty,
        "accuracy": accuracy_score,
        "difficulty": difficulty
    }


@dataclass
class FunctionRewardManager:
    config: RewardConfig
    tokenizer: PreTrainedTokenizer
    def __call__(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i in range(len(data)):
            data_item = data[i]
            response_ids = data_item.batch["responses"]
            response_mask = data_item.batch["response_mask"]
            valid_response_length = response_mask.sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=self.config.skip_special_tokens)
            ground_truth = data_item.non_tensor_batch["ground_truth"]
            grade = data_item.non_tensor_batch["grade"]
            difficulty = math.exp(- float(grade) / 8.0)
            score = compute_score(response_str, ground_truth, difficulty)
            reward_tensor[i, valid_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics
