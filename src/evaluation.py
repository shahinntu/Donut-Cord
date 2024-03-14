import os
import time
import logging

import torch

from src import set_logger, clear_handlers, Trainer


class Evaluator:
    def __init__(self, model, processor, config, metrics, model_log_dir):
        self._config = config
        self._model_log_dir = model_log_dir

        self._trainer = Trainer.for_evaluation(model, processor, self._config, metrics)

    @torch.no_grad()
    def evaluate(self, eval_dataloader):
        if self._trainer.accelerator.is_main_process:
            log_dir = self._create_log_dirs()
            eval_log_path = os.path.join(log_dir, "eval_logs", "eval.log")
            eval_logger = set_logger(eval_log_path)
            self._config.save(os.path.join(log_dir, "config/config.json"))

        eval_dataloader = self._trainer.accelerator.prepare(eval_dataloader)

        eval_metrics_avg_dict = self._trainer.evaluate(eval_dataloader)

        if self._trainer.accelerator.is_main_process:
            logging.info(
                f"- Evaluation metrics: {self._format_metrics(eval_metrics_avg_dict)}"
            )
            clear_handlers(eval_logger)

    def _create_log_dirs(self):
        log_dir = os.path.join(
            self._model_log_dir,
            time.strftime("eval_%y%m%d%H%M%S", time.localtime(time.time())),
        )
        os.mkdir(log_dir)
        os.mkdir(os.path.join(log_dir, "config"))
        os.mkdir(os.path.join(log_dir, "eval_logs"))

        return log_dir

    def _format_metrics(self, metrics_avg_dict):
        def format_value(v):
            if isinstance(v, float):
                if abs(v) < 1e-3 or abs(v) > 1e3:
                    return f"{v:.2e}"
                else:
                    return f"{v:.4f}"
            else:
                return str(v)

        metrics_string = "; ".join(
            f"{k}: {format_value(v)}" for k, v in metrics_avg_dict.items()
        )
        return metrics_string
