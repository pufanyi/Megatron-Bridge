# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Provider for datasets from SenseTime.
"""

import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple
from tqdm import tqdm
from transformers import AutoProcessor

from megatron.bridge.data.vlm_datasets.aoss.storage_clients import AOSSConfig, AOSSPatternRule
from megatron.bridge.data.vlm_datasets.conversation_dataset import VLMConversationDataset
from megatron.bridge.data.vlm_datasets.preloaded_provider import _load_preloaded_examples, _record_to_conversation
from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider


def _ensure_aoss_config(config: Any) -> Optional[AOSSConfig]:
    """Convert dict/DictConfig to AOSSConfig if needed.

    This handles the case where aoss_config comes from YAML loading via OmegaConf,
    which produces dicts/DictConfigs instead of dataclass instances.
    """
    if config is None:
        return None
    if isinstance(config, AOSSConfig):
        return config
    if isinstance(config, dict) or hasattr(config, "keys"):
        # Convert dict or DictConfig to AOSSConfig
        pattern_rules = []
        raw_rules = config.get("pattern_rules", [])
        for rule in raw_rules:
            if isinstance(rule, AOSSPatternRule):
                pattern_rules.append(rule)
            elif isinstance(rule, dict) or hasattr(rule, "keys"):
                pattern_rules.append(AOSSPatternRule(
                    pattern=rule.get("pattern", ""),
                    conf_path=rule.get("conf_path", ""),
                ))
        return AOSSConfig(
            default_conf_path=config.get("default_conf_path"),
            pattern_rules=pattern_rules,
        )
    logging.warning(f"Unexpected aoss_config type: {type(config)}. Ignoring.")
    return None


@dataclass(kw_only=True)
class SensetimeDatasetProvider(DatasetProvider):
    """DatasetProvider that builds VLM conversation datasets from SenseTime meta JSON files.

    Example usage with AOSS config:
        provider = SensetimeDatasetProvider(
            sequence_length=4096,
            meta_json_path="/path/to/meta.json",
            aoss_config=AOSSConfig(
                default_conf_path="/mnt/aigc/caizhongang/aoss.conf",
                pattern_rules=[
                    AOSSPatternRule(pattern="^s3://multimodal", conf_path="/mnt/aigc/users/pufanyi/aoss.conf"),
                ]
            )
        )
    """

    sequence_length: int
    hf_processor_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"

    # Path to the meta JSON file containing dataset definitions
    # Expected format: {"dataset_name": {"root": "image_root", "annotation": "jsonl_path", "repeat_time": float, ...}, ...}
    meta_json_path: Optional[str] = None

    # AOSS configuration for downloading images from object storage
    # If not provided, will try to use AOSS_DEFAULT_CONF environment variable
    aoss_config: Optional[AOSSConfig] = None

    skip_getting_attention_mask_from_dataset: bool = True
    dataloader_type: Optional[Literal["single", "cyclic", "external"]] = "single"
    shuffle: bool = True
    seed: int = 42

    def _load_meta_data(self) -> List[Dict[str, Any]]:
        """Load and merge all datasets defined in the meta JSON."""
        with open(self.meta_json_path, "r") as f:
            meta = json.load(f)

        all_examples = []

        # Ensure consistent shuffling
        rng = random.Random(self.seed)

        for ds_name, config in meta.items():
            root = config.get("root", "")
            annotation_path = config.get("annotation")
            repeat_time = config.get("repeat_time", 1.0)

            if not annotation_path or not os.path.exists(annotation_path):
                logging.warning(f"Skipping dataset {ds_name}: annotation path {annotation_path} not found.")
                continue

            examples = _load_preloaded_examples(annotation_path)

            # Apply downsampling if repeat_time < 1
            if repeat_time < 1.0:
                target_count = int(len(examples) * repeat_time)
                # Shuffle before truncation to ensure random sampling
                if self.shuffle:
                    rng.shuffle(examples)
                examples = examples[:target_count]

            # Process examples
            processed_examples = []
            for rec in examples:
                # Use root from meta config as image folder
                conv = _record_to_conversation(rec, image_folder=root)
                if conv is None:
                    continue
                processed_examples.append({"conversation": conv})

            # Apply upsampling if repeat_time > 1
            if repeat_time > 1.0:
                # Integer part repeating
                full_repeats = int(repeat_time)
                remainder = repeat_time - full_repeats

                expanded = processed_examples * full_repeats
                if remainder > 0:
                    # Shuffle again for the remainder part to avoid bias if using same prefix
                    remainder_pool = list(processed_examples)
                    if self.shuffle:
                        rng.shuffle(remainder_pool)
                    expanded += remainder_pool[: int(len(processed_examples) * remainder)]
                processed_examples = expanded

            all_examples.extend(processed_examples)
            logging.info(f"Loaded {len(processed_examples)} examples from {ds_name} (repeat_time={repeat_time})")

        # Final shuffle of the merged dataset
        if self.shuffle:
            rng.shuffle(all_examples)
        
        return all_examples

    def build_datasets(self, context: DatasetBuildContext) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        processor = AutoProcessor.from_pretrained(self.hf_processor_path, trust_remote_code=True)

        # SenseTime meta JSON usually defines the full training set
        # We load everything into a single list and return it as the training dataset
        # Validation/Test splits support can be added if the meta JSON structure distinguishes them

        base_examples = self._load_meta_data()

        if not base_examples:
            logging.warning(f"No usable examples loaded from {self.meta_json_path}")
            return None, None, None

        # Build AOSS client if config is provided
        aoss_client = None
        aoss_cfg = _ensure_aoss_config(self.aoss_config)
        if aoss_cfg is not None:
            try:
                aoss_client = aoss_cfg.build_client()
                logging.info(f"Initialized AOSS client with config: {aoss_cfg}")
            except Exception as e:
                logging.warning(f"Failed to build AOSS client: {e}. Will fall back to global client.")

        train_ds = VLMConversationDataset(
            base_examples=base_examples,
            target_length=context.train_samples,  # Or len(base_examples) if strict epoch alignment needed
            processor=processor,
            aoss_client=aoss_client,
        )

        # Currently only returning train_ds based on the description
        return train_ds, None, None
