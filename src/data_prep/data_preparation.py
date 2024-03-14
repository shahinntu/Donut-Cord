import json

import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class DocDataset(Dataset):
    def __init__(
        self,
        dataset_name_or_path,
        processor,
        max_length,
        split="train",
        ignore_id=-100,
        task_start_token="",
        sort_json_key=True,
    ):
        super().__init__()
        self._processor = processor
        self._max_length = max_length
        self._split = split
        self._ignore_id = ignore_id
        self._task_start_token = task_start_token
        self._sort_json_key = sort_json_key

        self._hf_dataset = load_dataset(dataset_name_or_path, split=self._split)
        self._hf_dataset = self._hf_dataset.map(
            self._prepare_ground_truth, batched=True
        )
        self._hf_dataset = self._hf_dataset.select_columns(
            ["image", "labels", "target_sequence"]
        )

    def __len__(self):
        return len(self._hf_dataset)

    def __getitem__(self, index):
        sample = self._hf_dataset[index]
        pixel_values = self._processor(
            sample["image"],
            random_padding=(self._split == "train"),
            return_tensors="pt",
        ).pixel_values.squeeze(0)

        return (pixel_values, torch.tensor(sample["labels"]), sample["target_sequence"])

    def _prepare_ground_truth(self, examples):
        gt_jsons = [
            json.loads(ground_truth)["gt_parse"]
            for ground_truth in examples["ground_truth"]
        ]
        examples["target_sequence"] = [
            self._processor.json2token(
                gt_json,
                update_tokens_for_json_key=(self._split == "train"),
                sort_json_key=self._sort_json_key,
            )
            for gt_json in gt_jsons
        ]
        prompts = [
            self._task_start_token
            + target_sequence
            + self._processor.tokenizer.eos_token
            for target_sequence in examples["target_sequence"]
        ]
        tokenized_prompts = self._processor.tokenizer(
            prompts,
            add_special_tokens=False,
            max_length=self._max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        examples["labels"] = tokenized_prompts.input_ids
        examples["labels"][
            examples["labels"] == self._processor.tokenizer.pad_token_id
        ] = self._ignore_id

        return examples
