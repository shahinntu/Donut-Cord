import os

from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import (
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
    get_scheduler,
)

from src import (
    DocProcessor,
    DocDataset,
    Params,
    Trainer,
    Evaluator,
    edit_similarity,
    Accuracy,
    F1Score,
)


class MLPipeline:
    def __init__(self, args, mode):
        self._mode = mode

        if self._mode == "train":
            self._config = Params(args.config_path)
            self._doc_processor, self._donut_model = (
                self._initialialize_processor_and_model_for_training()
            )
            self._train_dataloader, self._val_dataloader = self._prepare_train_data()
            self._trainer = self._get_trainer(args)
        elif self._mode == "eval":
            self._config, self._doc_processor, self._donut_model = (
                self._initialialize_config_processor_and_model_for_evaluation(args)
            )
            self._eval_dataloader = self._prepare_eval_data()
            self._evaluator = self._get_evaluator(args)

    @classmethod
    def for_training(cls, args):
        return cls(args, "train")

    @classmethod
    def for_evaluation(cls, args):
        return cls(args, "eval")

    def run(self):
        if self._mode == "train":
            self._trainer.train(self._train_dataloader, self._val_dataloader)
        elif self._mode == "eval":
            self._evaluator.evaluate(self._eval_dataloader)

    def _initialialize_processor_and_model_for_training(self):
        donut_model_config = VisionEncoderDecoderConfig.from_pretrained(
            self._config.MODEL.BASE_MODEL_NAME
        )
        donut_model_config.encoder.image_size = self._config.DATA.IMAGE_SIZE
        donut_model_config.decoder.max_length = self._config.DATA.MAX_LENGTH
        donut_model = VisionEncoderDecoderModel.from_pretrained(
            self._config.MODEL.BASE_MODEL_NAME, config=donut_model_config
        )

        doc_processor = DocProcessor.from_pretrained_with_model(
            donut_model, self._config.MODEL.BASE_MODEL_NAME
        )
        doc_processor.image_processor.size = self._config.DATA.IMAGE_SIZE[::-1]
        doc_processor.image_processor.do_align_long_axis = (
            self._config.DATA.DO_ALIGN_LONG_AXIS
        )
        doc_processor.add_special_tokens([self._config.DATA.TASK_START_TOKEN])

        donut_model.config.pad_token_id = doc_processor.tokenizer.pad_token_id
        donut_model.config.decoder_start_token_id = (
            doc_processor.tokenizer.convert_tokens_to_ids(
                self._config.DATA.TASK_START_TOKEN
            )
        )

        return doc_processor, donut_model

    def _initialialize_config_processor_and_model_for_evaluation(self, args):
        log_dir = os.path.join(args.model_log_dir, args.restore_version)

        config = Params(os.path.join(log_dir, "config", "config.json"))
        donut_model = VisionEncoderDecoderModel.from_pretrained(
            os.path.join(log_dir, "state", "model_best")
        )
        doc_processor = DocProcessor.from_pretrained_with_model(
            donut_model, os.path.join(log_dir, "state", "processor")
        )

        config.add("RESTORED_FROM", args.restore_version)

        return config, doc_processor, donut_model

    def _prepare_train_data(self):
        train_dataset = DocDataset(
            self._config.DATA.DATASET_NAME,
            self._doc_processor,
            self._config.DATA.MAX_LENGTH,
            split="train",
            ignore_id=self._config.DATA.IGNORE_ID,
            task_start_token=self._config.DATA.TASK_START_TOKEN,
            sort_json_key=self._config.DATA.SORT_JSON_KEY,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self._config.TRAINING.BATCH_SIZE.TRAIN,
            shuffle=True,
        )

        val_dataset = DocDataset(
            self._config.DATA.DATASET_NAME,
            self._doc_processor,
            self._config.DATA.MAX_LENGTH,
            split="validation",
            ignore_id=self._config.DATA.IGNORE_ID,
            task_start_token=self._config.DATA.TASK_START_TOKEN,
            sort_json_key=self._config.DATA.SORT_JSON_KEY,
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=self._config.TRAINING.BATCH_SIZE.TEST, shuffle=False
        )

        return train_dataloader, val_dataloader

    def _prepare_eval_data(self):
        eval_dataset = DocDataset(
            self._config.DATA.DATASET_NAME,
            self._doc_processor,
            self._config.DATA.MAX_LENGTH,
            split="test",
            ignore_id=self._config.DATA.IGNORE_ID,
            task_start_token=self._config.DATA.TASK_START_TOKEN,
            sort_json_key=self._config.DATA.SORT_JSON_KEY,
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self._config.TRAINING.BATCH_SIZE.TEST,
            shuffle=False,
        )

        return eval_dataloader

    def _get_trainer(self, args):
        optimizer = Adam(
            self._donut_model.parameters(),
            lr=self._config.TRAINING.ADAM_OPTIMIZER.LEARNING_RATE,
            betas=(
                self._config.TRAINING.ADAM_OPTIMIZER.BETA1,
                self._config.TRAINING.ADAM_OPTIMIZER.BETA2,
            ),
            weight_decay=self._config.TRAINING.ADAM_OPTIMIZER.WEIGHT_DECAY,
            eps=self._config.TRAINING.ADAM_OPTIMIZER.EPSILON,
        )
        lr_scheduler = get_scheduler(
            self._config.TRAINING.LR_SCHEDULER.TYPE,
            optimizer=optimizer,
            num_warmup_steps=self._config.TRAINING.LR_SCHEDULER.WARMUP_STEPS,
            num_training_steps=self._config.TRAINING.EPOCHS
            * len(self._train_dataloader),
        )

        metrics = {
            "edit_similarity": edit_similarity,
            "accuracy": Accuracy(self._doc_processor),
            "f1_score": F1Score(self._doc_processor),
        }
        objective = "accuracy"

        trainer = Trainer(
            self._donut_model,
            self._doc_processor,
            optimizer,
            lr_scheduler,
            self._config,
            args.model_log_dir,
            metrics=metrics,
            objective=objective,
        )

        return trainer

    def _get_evaluator(self, args):
        metrics = {
            "edit_similarity": edit_similarity,
            "accuracy": Accuracy(self._doc_processor),
            "f1_score": F1Score(self._doc_processor),
        }
        evaluator = Evaluator(
            self._donut_model,
            self._doc_processor,
            self._config,
            metrics,
            args.model_log_dir,
        )

        return evaluator
