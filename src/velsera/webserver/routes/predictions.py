from fastapi import APIRouter, Request
from pydantic import BaseModel

from velsera.finetuning.prompts import TestingPrompt
from velsera.utils.model_utils import Prediction, predict

router = APIRouter(prefix="/v1", tags=["classification predictions"])


class ClassificationInput(BaseModel):
    title: str
    abstract: str


@router.post("/classify_paper")
async def classify_paper(input: ClassificationInput, request: Request) -> Prediction:

    model = request.app.state.model
    tokenizer = request.app.state.tokenizer

    prediction = predict(
        prompt=TestingPrompt(title=input.title, abstract=input.abstract).prompt,
        model=model,
        tokenizer=tokenizer,
        labels=["yes", "no"],
    )
    return prediction
