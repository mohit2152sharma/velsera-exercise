from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from velsera.webserver.routes.predictions import router as PredictionRouter

MODEL_DIR = "model"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # TODO: Download model from s3 at the start of the server
    # and setting up db connection

    device = "cuda" if torch.cuda.is_available() else "cpu"
    peft_config = PeftConfig.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path
    )
    model = PeftModel.from_pretrained(base_model, MODEL_DIR).to(device)
    app.state.model = model
    app.state.tokenizer = tokenizer
    yield

    # TODO: Shutdown activities like closing db connection pool


app = FastAPI(lifespan=lifespan)
app.include_router(PredictionRouter, prefix="/predictions")


def main():
    uvicorn.run("velsera.webserver:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
