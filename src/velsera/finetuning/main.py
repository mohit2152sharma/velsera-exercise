import logging

from velsera.finetuning.fine_tuner import FineTuner
from velsera.finetuning.preprocess import Preprocessor
from velsera.finetuning.reporter import generate_report
from velsera.preprocessing.loader import Loader
from velsera.preprocessing.main import Preprocess as Cleaner


def fine_tune(model_name: str, cancer_dir: str, non_cancer_dir: str):

    loader = Loader(cancer_dir=cancer_dir, non_cancer_dir=non_cancer_dir)
    cancer_df, non_cancer_df = loader.load()
    cleaner = Cleaner(cancer_df, non_cancer_df)
    cleaned_df = cleaner.clean()

    splits = Preprocessor.train_test_split(
        cleaned_df, target_col="label", train_split_ratio=0.6
    )

    finetuner = FineTuner(model_name=model_name)

    # Before finetuning
    val_df_for_finetuning = Preprocessor.to_finetuner(splits.x_val, prompt="testing")
    predictions = finetuner.predict(val_df_for_finetuning, labels=["yes", "no"])
    before_finetuning_report = finetuner.evaluate(
        splits.y_val, predictions["predicted_label"], labels=["yes", "no"]
    )

    # Finetune
    train_df_for_finetuning = Preprocessor.to_finetuner(
        splits.x_train, labels=splits.y_train, prompt="training"
    )

    finetuner.finetune(
        train_df_for_finetuning,
        val_df_for_finetuning,
        output_dir="finetuning-model-dir",
    )

    # Predictions after finetuning
    finetuned_predictions = finetuner.predict(splits.x_test, labels=["yes", "no"])

    # After finetuning
    after_finetuning_report = finetuner.evaluate(
        splits.y_test, finetuned_predictions["predicted_label"], labels=["yes", "no"]
    )

    generate_report(before_finetuning_report, after_finetuning_report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting fine tuning")

    model_names = ["facebook/opt-350m", "facebook/opt-2.7b"]
    for model_name in model_names:
        fine_tune(model_name, "data/cancer", "data/non_cancer")
