# NOTE: can't install bitsandbytes on mac, but runs on kaggle gpus

from typing import Any, Literal, Tuple, TypeAlias, cast

import bitsandbytes as bnb  # type: ignore
import numpy as np
import torch
from datasets import Dataset
from pandas import DataFrame
from peft import LoraConfig
from pydantic import BaseModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoModelForCausalLM as HfAutoModelForCausalLM
from transformers import AutoTokenizer as HfAutoTokenizer
from trl import SFTConfig, SFTTrainer

from velsera.utils.model_utils import (
    calculate_probability_and_text,
    generate_model_outputs,
    get_device,
    predict,
)

Labels: TypeAlias = list[Literal["yes", "no"]]


class Report(BaseModel):
    model_name: str
    accuracy: float
    labels_accuracy: dict[str, float]
    classification_report: str
    confusion_matrix: str


class FineTuner:

    def __init__(self, model_name: str) -> None:
        self.device = get_device()
        self.model_name = model_name
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()

    def _load_model(self):
        model = HfAutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", torch_dtype=torch.float16
        )
        model.config.pretraining_tp = 1
        model.config.use_cache = False
        return model

    def _load_tokenizer(self):
        tokenizer = HfAutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        return tokenizer

    def _generate_output(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> Any:
        """Generates raw model output for given input_ids."""
        return generate_model_outputs(
            model=self.model,
            tokenizer=self.tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    def _calculate_probability_and_text(
        self, generated_outputs: Any, input_ids: torch.Tensor
    ) -> Tuple[str, float]:
        """Calculates sequence probability and decodes generated text."""
        return calculate_probability_and_text(
            tokenizer=self.tokenizer,
            generated_outputs=generated_outputs,
            input_ids=input_ids,
        )

    def _predict_single_with_prob(
        self, prompt: str, labels: Labels
    ) -> Tuple[str, float]:
        """Predicts label and probability for a single prompt."""
        prediction = predict(
            prompt=prompt,
            model=self.model,
            tokenizer=self.tokenizer,
            labels=cast(list[str], labels),
        )

        return prediction.label, prediction.probability

    def predict(self, df: DataFrame, labels: Labels) -> DataFrame:
        """
        Predicts labels and probabilities for the dataframe, adding them as new columns.
        """
        if "text" not in df.columns:
            raise ValueError("Input DataFrame must contain a 'text' column.")

        df_copy = df.copy()

        df_copy[["predicted_label", "predicted_prob"]] = df_copy.apply(
            lambda row: self._predict_single_with_prob(row["text"], labels),
            axis=1,
            result_type="expand",
        )
        return df_copy

    def evaluate(self, y_true, y_pred, labels: Labels):
        """Evaluates the model's performance."""
        mapping = {label: i for i, label in enumerate(labels)}

        def map_func(x):
            return mapping[x]

        y_true_mapped = np.vectorize(map_func)(y_true)
        y_pred_mapped = np.vectorize(map_func)(y_pred)

        accuracy = accuracy_score(y_true=y_true_mapped, y_pred=y_pred_mapped)

        labels_accuracy = {}
        for label in labels:
            label_indices = [
                i for i in range(len(y_true_mapped)) if y_true_mapped[i] == label
            ]
            label_y_true = [y_true_mapped[i] for i in label_indices]
            label_y_pred = [y_pred_mapped[i] for i in label_indices]
            label_accuracy = accuracy_score(label_y_true, label_y_pred)
            labels_accuracy[label] = label_accuracy

        # Generate classification report
        class_report = classification_report(
            y_true=y_true_mapped,
            y_pred=y_pred_mapped,
            target_names=labels,
            labels=list(range(len(labels))),
        )

        conf_matrix = confusion_matrix(
            y_true=y_true_mapped, y_pred=y_pred_mapped, labels=list(range(len(labels)))
        )

        return Report(
            model_name=self.model_name,
            accuracy=accuracy,
            labels_accuracy=labels_accuracy,
            classification_report=str(class_report),
            confusion_matrix=str(conf_matrix),
        )

    def find_all_linear_names(self):
        cls = bnb.nn.Linear4bit
        lora_module_names = set()
        for name, module in self.model.named_modules():
            if isinstance(module, cls):
                names = name.split(".")
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if "lm_head" in lora_module_names:  # needed for 16 bit
            lora_module_names.remove("lm_head")

        return list(lora_module_names)

    def get_peft_config(self) -> LoraConfig:
        return LoraConfig(
            lora_alpha=16,
            lora_dropout=0,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.find_all_linear_names(),
        )

    def get_sft_config(self, output_dir: str) -> SFTConfig:
        training_arguments = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",
            logging_steps=1,
            learning_rate=2e-4,
            weight_decay=0.001,
            fp16=True,
            bf16=False,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=0.03,
            group_by_length=False,
            lr_scheduler_type="cosine",
            report_to="wandb",
            eval_strategy="steps",
            eval_steps=0.2,
            max_seq_length=512,
            dataset_text_field="text",
            packing=False,
            dataset_kwargs={
                "add_special_tokens": False,
                "append_concat_token": False,
            },
        )
        return training_arguments

    def finetune(self, train_df: DataFrame, eval_df: DataFrame, output_dir: str):

        train_data = Dataset.from_pandas(train_df[["text"]])  # type: ignore
        eval_data = Dataset.from_pandas(eval_df[["text"]])  # type: ignore

        peft_config = self.get_peft_config()
        training_arguments = self.get_sft_config(output_dir)

        trainer = SFTTrainer(
            model=self.model,
            args=training_arguments,
            train_dataset=train_data,
            eval_dataset=eval_data,
            peft_config=peft_config,
            processing_class=self.tokenizer,
        )
