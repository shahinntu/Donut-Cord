import os

import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

from src import Params


class Predictor:
    def __init__(self, model, processor, config, device):
        self._model = model
        self._processor = processor
        self._config = config
        self._device = device

        self._model = self._model.to(self._device)
        self._model.eval()

    @classmethod
    def from_pretrained(cls, model_log_dir, restore_version, device):
        log_dir = os.path.join(model_log_dir, restore_version)
        config = Params(os.path.join(log_dir, "config/config.json"))
        processor = DonutProcessor.from_pretrained(
            os.path.join(log_dir, "state/processor")
        )
        model = VisionEncoderDecoderModel.from_pretrained(
            os.path.join(log_dir, "state/model_best")
        )

        return cls(model, processor, config, device)

    @torch.no_grad()
    def predict(self, images):
        pixel_values = self._processor(images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self._device)

        decoder_input_ids = torch.full(
            (pixel_values.size(0), 1),
            self._model.config.decoder_start_token_id,
            device=self._device,
        )

        outputs = self._model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=self._config.DATA.MAX_LENGTH,
            pad_token_id=self._processor.tokenizer.pad_token_id,
            eos_token_id=self._processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self._processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        predictions = self._processor.tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=False
        )

        pred_jsons = [
            self._processor.token2json(prediction) for prediction in predictions
        ]

        return predictions
