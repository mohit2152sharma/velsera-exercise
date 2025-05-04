import logging
import os

from velsera.finetuning.fine_tuner import Report

logger = logging.getLogger(__name__)


def section(report: Report) -> str:
    labels_accuracy = []
    for label, accuracy in report.labels_accuracy.items():
        labels_accuracy.append(f"- Accuracy for {label}: {accuracy}")
    labels_accuracy_string = "\n".join(labels_accuracy)
    return f"""
Accracy: {report.accuracy}

Labels Accuracy: 
{labels_accuracy_string}

Classification Report:
{report.classification_report}

Confusion Matrix:
{report.confusion_matrix}"""


def generate_report(before: Report, after: Report, dir: str = "reports") -> None:

    if dir not in os.listdir():
        os.mkdir(dir)

    model_name = before.model_name.replace("/", "_").replace("-", "_").replace(".", "_")
    file_name = f"{model_name}.md"

    report = f"""
# Model Name: {before.model_name}

## Before Finetuning
{section(before)}

## After
{section(after)}

**The accuracy improved by {after.accuracy - before.accuracy} after finetuning**
"""

    with open(os.path.join(dir, file_name), "w") as f:
        f.write(report)

    logger.info(f"Report generated at {os.path.join(dir, file_name)}")
