import json
import os
import tqdm

from transformers import logging
logging.set_verbosity_error()   # only errors

from typing import TYPE_CHECKING, TypedDict, cast

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BatchEncoding,
)

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from transformers import PreTrainedModel, TokenizersBackend
    from transformers.modeling_outputs import SequenceClassifierOutput

MAX_LEN = 64
BATCH_SIZE = 32
EPOCHS = 5
LR = 2e-5
NUM_LABELS = 3

MODEL_NAMES = ["bert-base-uncased", "distilbert-base-uncased"]


class EvaluationMetricResult(TypedDict):
    """Type definition for the result of model evaluation metrics.

    Attributes
    ----------
    accuracy : float
        The accuracy of the model on the validation set.
    weighted_precision : float
        The weighted precision of the model on the validation set.
    weighted_recall : float
        The weighted recall of the model on the validation set.
    weighted_f1 : float
        The weighted F1 score of the model on the validation set.
    macro_precision : float
        The macro precision of the model on the validation set.
    macro_recall : float
        The macro recall of the model on the validation set.
    macro_f1 : float
        The macro F1 score of the model on the validation set.
    classification_report : str
        A string containing the classification report.
    confusion_matrix : list[list[int]]
        A confusion matrix representing true vs predicted labels.
    """

    accuracy: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    classification_report: str
    confusion_matrix: list[list[int]]


class ModelTrainingResult(TypedDict):
    """Type definition for the result of model training.

    Attributes
    ----------
    model_name : str
        The name of the model used for training.
    tokenizer : TokenizersBackend
        The tokenizer instance used for preparing the data.
    model : PreTrainedModel
        The trained model instance.
    history : list[dict[str, float]]
        A list of dictionaries containing evaluation metrics for each epoch.
    final_validation_metrics : EvaluationMetricResult
        A dictionary containing the final validation metrics after training.
    saved_model_dir : str
        The directory where the trained model is saved.
    """

    model_name: str
    tokenizer: TokenizersBackend
    model: PreTrainedModel
    history: list[dict[str, float]]
    final_validation_metrics: EvaluationMetricResult
    saved_model_dir: str


class DataDataset(Dataset[dict[str, torch.Tensor]]):
    """A custom Dataset class for tokenizing text data and preparing it for model training.

    Attributes
    ----------
    sentences : pd.Series
        A pandas Series containing the text data to be tokenized.
    labels : pd.Series
        A pandas Series containing the labels corresponding to the text data.
    tokenizer : TokenizersBackend
        An instance of a tokenizer from the Hugging Face Transformers library.
    max_length : int
        The maximum length to which the tokenized sequences will be truncated or
        padded.
    """

    def __init__(
        self,
        sentences: pd.Series,
        labels: pd.Series,
        tokenizer: TokenizersBackend,
        max_length: int,
    ) -> None:
        self.encodings = tokenizer(
            sentences.tolist(),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels.values, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item: dict[str, torch.Tensor] = {
            k: v[idx] for k, v in self.encodings.items()
        }
        item["labels"] = self.labels[idx]
        return item


def _evaluate_model(
    model: PreTrainedModel,
    val_loader: DataLoader,
    device: torch.device,
) -> EvaluationMetricResult:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for batch in val_loader:
            inputs: dict[str, torch.Tensor] = {
                k: v.to(device) for k, v in batch.items()
            }
            labels = inputs.pop("labels")
            outputs: SequenceClassifierOutput = model(**inputs)
            preds: np.ndarray[tuple[int], np.dtype[np.int64]] = (
                cast("torch.Tensor", outputs.logits)
                .argmax(dim=-1)
                .cpu()
                .numpy()
            )
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds)

    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
    )

    return {
        "accuracy": float(acc),
        "weighted_precision": float(prec_weighted),
        "weighted_recall": float(rec_weighted),
        "weighted_f1": float(f1_weighted),
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "macro_f1": float(f1_macro),
        "classification_report": cast(
            "str",
            classification_report(
                y_true,
                y_pred,
                labels=[0, 1, 2],
                target_names=["Bearish", "Bullish", "Neutral"],
                digits=4,
                output_dict=True,
                zero_division=0,
            ),
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1, 2]
        ).tolist(),
    }


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_name: str=MODEL_NAMES[0],
    num_labels: int=NUM_LABELS,
    device: torch.device="cuda",
    epochs: int=EPOCHS,
    batch_size: int=BATCH_SIZE,
    lr: float=LR,
    max_length: int=MAX_LEN,
) -> ModelTrainingResult:
    """Train a transformer-based model for sequence classification.

    Parameters
    ----------
    train_df : pd.DataFrame
        A pandas DataFrame containing the training data with columns "text" and
        "label".
    val_df : pd.DataFrame
        A pandas DataFrame containing the validation data with columns "text"
        and "label".
    model_name : str
        The name of the pre-trained transformer model to use
        (e.g., "bert-base-uncased").
    num_labels : int
        The number of unique labels in the classification task.
    device : torch.device
        The device that PyTorch will use for training (e.g., "cuda" or "cpu").
    epochs : int
        The number of epochs to train the model.
    batch_size : int
        The size of each training batch.
    lr : float
        The learning rate for the optimizer.
    max_length : int
        The maximum length to which the tokenized sequences will be truncated or
        padded.

    Returns
    -------
    ModelTrainingResult
        A dictionary containing the results of the model training, including the
        model name, training history, final validation metrics, and the
        directory where the trained model is saved.
    """
    tokenizer: TokenizersBackend = AutoTokenizer.from_pretrained(model_name)
    model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    model.to(device)  # pyright: ignore[reportArgumentType]

    train_ds = DataDataset(
        train_df["text"], train_df["label"], tokenizer, max_length=max_length
    )
    val_ds = DataDataset(
        val_df["text"], val_df["label"], tokenizer, max_length=max_length
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    history = []
    best_state: dict[str, torch.Tensor] | None = None
    best_f1 = -1.0

    for epoch in tqdm.trange(epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            inputs: dict[str, torch.Tensor] = {
                k: v.to(device) for k, v in batch.items()
            }
            labels: torch.Tensor = inputs.pop("labels")

            optimizer.zero_grad()
            outputs: SequenceClassifierOutput = model(**inputs, labels=labels)
            loss: torch.Tensor = cast("torch.Tensor", outputs.loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        validation_metrics = _evaluate_model(model, val_loader, device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "accuracy": validation_metrics["accuracy"],
                "macro_precision": validation_metrics["macro_precision"],
                "macro_recall": validation_metrics["macro_recall"],
                "macro_f1": validation_metrics["macro_f1"],
                "weighted_precision": validation_metrics["weighted_precision"],
                "weighted_recall": validation_metrics["weighted_recall"],
                "weighted_f1": validation_metrics["weighted_f1"],
            }
        )
        print(
            f"{model_name} | epoch {epoch + 1}/{epochs} | train_loss={avg_train_loss:.4f} | val_weighted_f1={validation_metrics['weighted_f1']:.4f} | val_macro_f1={validation_metrics['macro_f1']:.4f}"
        )
        if validation_metrics["weighted_f1"] > best_f1:
            best_f1 = validation_metrics["weighted_f1"]
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
    if best_state is not None:
        model.load_state_dict(best_state)
    final_val = _evaluate_model(model, val_loader, device)

    save_dir = f"src/data/transformer_outputs_{model_name}"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    with open(f"{save_dir}/results.json", "w") as f:
        json.dump(final_val, f, indent=2)

    return {
        "model_name": model_name,
        "tokenizer": tokenizer,
        "model": model,
        "history": history,
        "final_validation_metrics": final_val,
        "saved_model_dir": save_dir,
    }


def predict(
    sentences: list[str],
    model: PreTrainedModel,
    tokenizer: TokenizersBackend=AutoTokenizer.from_pretrained(MODEL_NAMES[0]),
    max_length: int=MAX_LEN,
    device: torch.device="cuda",
) -> list[int]:
    """Predict the labels for a list of input texts using a trained transformer model.

    Parameters
    ----------
    sentences : list[str]
        A list of input texts for which to predict labels.
    model : PreTrainedModel
        The trained transformer model to use for prediction.
    tokenizer : TokenizersBackend
        The tokenizer instance used for preparing the data.
    max_length : int
        The maximum length to which the tokenized sequences will be truncated or
        padded.
    device : torch.device
        The device that PyTorch will use for prediction (e.g., "cuda" or "cpu").

    Returns
    -------
    list[int]
        A list of predicted label indices corresponding to the input texts.
    """
    model.eval()
    encodings: BatchEncoding = tokenizer(
        sentences,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    encodings_dict: dict[str, torch.Tensor] = {
        k: v.to(device) for k, v in encodings.items()
    }
    with torch.no_grad():
        outputs: SequenceClassifierOutput = model(**encodings_dict)
        preds: np.ndarray[tuple[int], np.dtype[np.int64]] = (
            cast("torch.Tensor", outputs.logits).argmax(dim=-1).cpu().numpy()
        )
    return preds.tolist()
