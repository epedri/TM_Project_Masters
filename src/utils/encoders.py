import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def encode_bow(token_series: pd.Series) -> pd.DataFrame:

    bow = CountVectorizer(analyzer=lambda x: x)
    
    bow_matrix = bow.fit_transform(token_series)
    
    bow_df = pd.DataFrame(
        bow_matrix.toarray(),
        columns=bow.get_feature_names_out(),
        index=token_series.index
    )
    
    return bow_df


def encode_vec(token_series: pd.Series) -> pd.DataFrame:
    ...


def encode_trans(token_series: pd.Series) -> pd.DataFrame:
    ...
