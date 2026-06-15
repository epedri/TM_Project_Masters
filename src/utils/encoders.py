from typing import TYPE_CHECKING, Literal, cast, overload

import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    TokenizersBackend,
    logging,
)

if TYPE_CHECKING:
    import gensim
    from scipy.sparse import csr_matrix
    from transformers.modeling_outputs import BaseModelOutput

logging.set_verbosity_error()  # only errors


@overload
def encode_bow(
    token_series: pd.Series,
    encoder: TfidfVectorizer | None = ...,
    min_df: int = 30,
    *,
    return_bow: Literal[True] = True,
) -> tuple[pd.DataFrame, TfidfVectorizer]: ...


@overload
def encode_bow(
    token_series: pd.Series,
    encoder: TfidfVectorizer | None = ...,
    min_df: int = 30,
    *,
    return_bow: Literal[False],
) -> pd.DataFrame: ...


def encode_bow(
    token_series: pd.Series,
    encoder: TfidfVectorizer | None = None,
    min_df: int = 30,
    *,
    return_bow: bool = True,
) -> pd.DataFrame | tuple[pd.DataFrame, TfidfVectorizer]:
    """Encode provided series of tokens to Bag of Words (TF-IDF).

    Parameters
    ----------
    token_series : pd.Series
        Series of tokens to encode.
    encoder : TfidfVectorizer | None, optional
        A pretrained encoder to apply on new data, by default None
    min_df : int, optional
        Minimum amount of appearances of token to include in encoded data, by
        default 30
    return_bow : bool, optional
        Whether to return the trained model, by default True

    Returns
    -------
    pd.DataFrame
        Encoded Data
    TfidfVectorizer
        Trained Encoder Model
    """
    if encoder is not None:
        bow = encoder
        bow_matrix = cast("csr_matrix", bow.transform(token_series))
    else:
        bow = TfidfVectorizer(analyzer=lambda x: x, min_df=min_df)
        bow_matrix = cast("csr_matrix", bow.fit_transform(token_series))

    bow_df = pd.DataFrame(
        bow_matrix.toarray(),
        columns=bow.get_feature_names_out(),
        index=token_series.index,
    )

    if return_bow:
        return bow_df, bow

    return bow_df


def train_word2vec_model(
    token_series: pd.Series,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 1,
) -> gensim.models.keyedvectors.Doc2VecKeyedVectors:
    """Train a Word2Vec model.

    Parameters
    ----------
    token_series : pd.Series
        Series of tokens to base model on
    vector_size : int, optional
        The output space dimension, by default 100
    window : int, optional
        Maximum distance between the current and predicted word within a
        sentence, by default 5
    min_count : int, optional
        Ignores all words with total frequency lower than this, by default 1

    Returns
    -------
    gensim.models.keyedvectors.Doc2VecKeyedVectors
        A trained Encoder Model
    """
    model = Word2Vec(
        sentences=token_series.tolist(),  # Tokenized dataset
        vector_size=vector_size,  # size of each word vector
        window=window,  # how many words come close in context
        min_count=min_count,  # ignores words that don't appear much
        workers=4,  # number of CPU Threads
    )

    return model.wv


def encode_w2v(
    token_series: pd.Series,
    model: gensim.models.keyedvectors.Doc2VecKeyedVectors,
) -> pd.DataFrame:
    """Encode provided series of tokens to Word2Vec format.

    Parameters
    ----------
    token_series : pd.Series
        Series of tokens to encode.
    model : gensim.models.keyedvectors.Doc2VecKeyedVectors
        A pretrained encoder to apply on the data.

    Returns
    -------
    pd.DataFrame
        Encoded Data
    """
    vector_size = model.vector_size

    embeddings = []

    for tokens in token_series:
        # get encoding of every token in observation
        vectors = [model[w] for w in tokens if w in model]

        if vectors:
            # average tokens out
            embeddings.append(np.mean(vectors, axis=0))
        else:
            # assign embedding with no known tokens to zeros
            embeddings.append(np.zeros(vector_size))

    return pd.DataFrame(
        embeddings,
        dtype=np.float32,
        columns=[f"dimention_{n}" for n in range(vector_size)],
    )


def load_model(
    model_name: str = "bert-base-uncased",
    *,
    device: torch.device,
) -> tuple[TokenizersBackend, PreTrainedModel]:
    """Load tokenizer and model, moved to the given device.

    Parameters
    ----------
    model_name: str, optional
        Model name to load, by default "bert-base-uncased"
        Can be either:
            - A string, the model id of a predefined tokenizer hosted inside a model repo on huggingface.co.
            - A path to a directory containing vocabulary files required by the tokenizer, for instance saved
              using the [~PreTrainedTokenizer.save_pretrained] method, e.g., ./my_model_directory/.
            - A path or url to a single saved vocabulary file if and only if the tokenizer only requires a
              single vocabulary file (like Bert or XLNet), e.g.: ./my_model_directory/vocab.txt. (Not
              applicable to all derived classes)
    device: torch.device
        The desired device of the parameters and buffers in this module.

    Returns
    -------
    tuple[TokenizersBackend, PreTrainedModel]
        A tokenizer and a pretrained model, based on `model_name`.
    """
    tokenizer: TokenizersBackend = AutoTokenizer.from_pretrained(model_name)

    model: PreTrainedModel = AutoModel.from_pretrained(model_name)

    model.to(device)  # pyright: ignore[reportArgumentType]
    model.eval()
    return tokenizer, model


@torch.no_grad()
def encode_trans(
    series: pd.Series,
    tokenizer: TokenizersBackend,
    model: PreTrainedModel,
    device: torch.device,
    max_length: int = 128,
    batch_size: int = 8,
) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
    """Return a numpy array containing the [CLS] token embedding for each text.

    This array has shape (num_texts, hidden_size).

    The function processes texts in batches to limit peak GPU memory usage.
    Lower batch_size if you run out of memory.

    Parameters
    ----------
    series : pd.Series
        Series to encode.
    tokenizer : TokenizersBackend
        Tokenizer to use.
    model : PreTrainedModel
        Encoder to use.
    device : torch.device
        The desired device of the parameters and buffers in this module.
    max_length : int, optional
        Controls the maximum length to use by one of the truncation/padding
        parameters of the `tokenizer`, by default 128
    batch_size : int, optional
        Batch size used to limit peak GPU memory usage, by default 8

    Returns
    -------
    np.ndarray
        Array containing the [CLS] token embeddings for each text.
    """
    texts = series.to_list()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        outputs: BaseModelOutput = model(**inputs)

        cls_embeddings = cast("torch.Tensor", outputs.last_hidden_state)[
            :, 0, :
        ]

        all_embeddings.append(cls_embeddings.cpu())

    return torch.cat(all_embeddings, dim=0).cpu().numpy()
