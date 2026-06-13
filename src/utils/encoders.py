import pandas as pd
import numpy as np
import torch
from gensim.models import Word2Vec
from transformers import AutoModel, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_NAME = "bert-base-uncased"  # swap for any encoder model (bert, roberta, distilbert, etc.)


def encode_bow(token_series: pd.Series,
               encoder=None,
               min_df:int=30,
               return_bow:bool=True) -> pd.DataFrame:

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
    tokenized_series: pd.Series,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 1,
):
    model = Word2Vec(
        sentences=tokenized_series.tolist(),  #Tokenized dataset
        vector_size=vector_size, #size of each word vector
        window=window, #how many words come close in context
        min_count=min_count, #ignores words that don't appear much
        workers=4, #number of CPU Threads
    )

    return model.wv


def encode_w2v( 
    tokenized_series: pd.Series,
    model,
):
    vector_size = model.vector_size

    embeddings = []

    for tokens in tokenized_series:
        vectors = [model[w] for w in tokens if w in model]

        if vectors:
            embeddings.append(np.mean(vectors, axis=0))
        else:
            embeddings.append(np.zeros(vector_size))

    return pd.DataFrame(embeddings, 
                        dtype=np.float32, 
                        columns=[f"dimention_{n}" for n in range(vector_size)])


def load_model(
    model_name: str = MODEL_NAME,
    device: str | None = None,
):
    """Load tokenizer and model, moved to the given (or auto-detected) device."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModel.from_pretrained(model_name)

    model.to(device)
    model.eval()
    return tokenizer, model, device


@torch.no_grad()
def encode_trans(
    series, tokenizer, model, device, max_length=128, batch_size=8
):
    """Return a tensor of shape (num_texts, hidden_size) containing the [CLS] token embedding (last hidden state, position 0) for each text.

    Processes texts in batches to limit peak GPU memory usage. Lower
    batch_size further if you still run out of memory.
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
