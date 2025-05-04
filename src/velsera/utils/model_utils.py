from functools import lru_cache
from typing import Any

import torch
from pydantic import BaseModel


class Prediction(BaseModel):
    label: str
    probability: float


@lru_cache
def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def generate_model_outputs(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    **kwargs,
) -> Any:
    return model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=kwargs.get("do_sample", True),
        max_new_tokens=kwargs.get("max_new_tokens", 2),
        temperature=kwargs.get("temperature", 0.1),
        num_return_sequences=kwargs.get("num_return_sequences", 1),
        output_scores=kwargs.get("output_scores", True),
        return_dict_in_generate=kwargs.get("return_dict_in_generate", True),
        pad_token_id=tokenizer.pad_token_id,
        **kwargs,
    )


def calculate_probability_and_text(
    tokenizer, generated_outputs: Any, input_ids: torch.Tensor
) -> tuple[str, float]:
    gen_sequences = generated_outputs.sequences[:, input_ids.shape[-1] :]
    probs = torch.stack(generated_outputs.scores, dim=1).softmax(-1)
    gen_probs = torch.gather(probs, 2, gen_sequences[:, :, None]).squeeze(-1)
    sequence_prob = gen_probs.prod(-1).item() if gen_probs.numel() > 0 else 0.0

    generated_text = tokenizer.decode(
        gen_sequences[0], skip_special_tokens=True
    ).strip()

    return generated_text, sequence_prob


def predict(prompt: str, model, tokenizer, labels: list[str]) -> Prediction:
    device = get_device()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")

    generated_outputs = generate_model_outputs(
        model, tokenizer, input_ids, attention_mask
    )
    generated_text, probability = calculate_probability_and_text(
        tokenizer, generated_outputs, input_ids
    )

    predicted_label = "none"
    for label in labels:
        if label.lower() in generated_text.lower():
            predicted_label = label
            break

    return Prediction(label=predicted_label, probability=probability)
