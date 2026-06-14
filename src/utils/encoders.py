from transformers import logging
logging.set_verbosity_error()   # only errors

import pandas as pd
import numpy as np
import torch
import gensim
from gensim.models import Word2Vec
from transformers import AutoModel, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_NAME = "bert-base-uncased"  # swap for any encoder model (bert, roberta, distilbert, etc.)


def encode_bow(token_series: pd.Series,
               encoder=None,
               min_df:int=30,
               return_bow:bool=True) -> pd.DataFrame:
    """
    Encodes provided series of tokens to Bag of Words (TF-IDF)
    
    Parameters
    ---------
    token_series: pandas.Series
        Series of tokens to encode.

    encoder: sklearn.feature_extraction.text.TfidfVectorizer, default=None
        A pretrained encoder to apply on new data.

    min_df: int, default=30
        Minimum ammount of appearances of token to include in encoded data.

    return_bow: bool, default=True
        Whether to return a trained model.

    Returns
    -------
    bow_df: pandas.DataFrame
        Encoded Data

    bow: sklearn.feature_extraction.text.TfidfVectorizer
        Trained Encoder Model
    """

    if encoder:
        bow = encoder
        bow_matrix = bow.transform(token_series)
    else:
        bow = TfidfVectorizer(analyzer=lambda x: x,
                              min_df=min_df)
        
        bow_matrix = bow.fit_transform(token_series)
    
    bow_df = pd.DataFrame(
        bow_matrix.toarray(),
        columns=bow.get_feature_names_out(),
        index=token_series.index
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
    """
    Trains a Word2Vec model.
    
    Parameters
    ---------
    token_series: pandas.Series
        Series of tokens to base model on.

    vector_size: int, default=100
        The output space dimention

    window: int, default=5
        Maximum distance between the current and predicted word within a sentence.

    min_count: int, default=1
        Ignores all words with total frequency lower than this.

    Returns
    -------
    model.wv: gensim.models.keyedvectors.Doc2VecKeyedVectors
        A trained Encoder Model.
    """

    model = Word2Vec(
        sentences=token_series.tolist(),  #Tokenized dataset
        vector_size=vector_size, #size of each word vector
        window=window, #how many words come close in context
        min_count=min_count, #ignores words that don't appear much
        workers=4, #number of CPU Threads
    )

    return model.wv


def encode_w2v( 
    token_series: pd.Series,
    model: gensim.models.keyedvectors.Doc2VecKeyedVectors,
)->pd.DataFrame:
    """
    Encodes provided series of tokens to Word2Vec format.
    
    Parameters
    ---------
    token_series: pandas.Series
        Series of tokens to encode.

    model: gensim.models.keyedvectors.Doc2VecKeyedVectors
        A pretrained encoder to apply on the data.

    Returns
    -------
    pandas.DataFrame
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

    return pd.DataFrame(embeddings, 
                        dtype=np.float32, 
                        columns=[f"dimention_{n}" for n in range(vector_size)])


def load_model(
    model_name: str = MODEL_NAME,
    device: str | None = None,
):
    """
    Load tokenizer and model, moved to the given (or auto-detected) device.
    
    Parameters
    ---------
    model_name: str, default='bert-base-uncased'
        Can be either:
            - A string, the model id of a predefined tokenizer hosted inside a model repo on huggingface.co.
            - A path to a directory containing vocabulary files required by the tokenizer, for instance saved
              using the [~PreTrainedTokenizer.save_pretrained] method, e.g., ./my_model_directory/.
            - A path or url to a single saved vocabulary file if and only if the tokenizer only requires a
              single vocabulary file (like Bert or XLNet), e.g.: ./my_model_directory/vocab.txt. (Not
              applicable to all derived classes)
    
    device: str | None, default=None
        The desired device of the parameters and buffers in this module.
    
    Returns
    -------
    tokenizer: Any
        A tokenizer based on `model_name` model.
    
    model: Any
        A `model_name` model.

    device: str
        The desired or Best Availiable device of the parameters and buffers in this module.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModel.from_pretrained(model_name)

    model.to(device)
    model.eval()
    return tokenizer, model, device


@torch.no_grad()
def encode_trans(
    series:pd.Series, 
    tokenizer, 
    model, 
    device:str, 
    max_length:int=128, 
    batch_size:int=8
)->torch.Tensor:
    """
    Return a tensor of shape (num_texts, hidden_size) containing the [CLS] token embedding (last hidden state, position 0) for each text.

    Processes texts in batches to limit peak GPU memory usage. Lower
    batch_size further if you still run out of memory.

    Parameters
    ----------
    series: pandas.Series
        Series to encode.

    tokenizer: Any
        Tokenizer to use.

    model: Any
        Encoder to use.

    device: str
        The desired device of the parameters and buffers in this module.

    max_length:int, default=128
        Controls the maximum length to use by one of the truncation/padding parameters of the `tokenizer`.

    batch_size:int, default=8
        Batch size used to limit peak GPU memory usage
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

        outputs = model(**inputs)

        cls_embeddings = outputs.last_hidden_state[:, 0, :]

        all_embeddings.append(cls_embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)
