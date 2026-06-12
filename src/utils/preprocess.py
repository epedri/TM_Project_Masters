import re
import emoji
import contractions

import pandas as pd

from transformers import AutoTokenizer
from typing import Iterable

from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import stopwords

STOPWORDS = list(stopwords.words("English"))
_wordnet_lem = WordNetLemmatizer()


def replace_urls(series: pd.Series,
                 d:str="URL") -> pd.Series:
    return series.str.replace(
        r"(?i)(https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,63}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*))",
        d,
        regex=True,
    )


def replace_mentions(series:pd.Series,
                     d:str="REF") -> pd.Series:
    return series.str.replace(
        r'@\w+', 
        d, 
        regex=True
        )


def stock_mentions(series:pd.Series,
                     d:str="STOCK") -> pd.Series:
    return series.str.replace(
        r'\$[A-Za-z]+', 
        d, 
        regex=True
        )


def replace_punctuations(series:pd.Series)->pd.Series:
    def _replace_punctuations_unit(text):
        text = str(text)

        text = contractions.fix(text) #fixes apostophres, like You're into You are

        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text) #regex to remove these characters, encoding issues or old ASCII formating

        text = re.sub(r"[–—−]", "-", text) #regex to put every dash type to one single type
        
        text = re.sub(r'(\d)\s*-\s*(\d)', r'\1 to \2', text) #if "-" between 2 numbers change it to "to", but remove this if causing problems
        
        text = re.sub(r'\s-(?=\d)', ' minus ', text) #regex to convert a hyphen to a minus sign token if followed by a number

        text = re.sub(r"'s", " ", text)

        text = re.sub(r"[_ƒº½]", " ", text)
        
        text = re.sub(r'%', ' percent', text) #regex for % to appear as percent

        text = re.sub(r"[^\w\s]", " ", text) #removing punctuation but apostophres already taken care by contractions

        text = re.sub(r'\s+', ' ', text) #normalizing whitespace (if sequence of whitespaces, make it justa a whitespace)

        return text.strip()
    return series.apply(_replace_punctuations_unit)


def split_hashtags(series: pd.Series) -> pd.Series:

    def _split_tag(tag: str) -> str:
        word = tag.lstrip("#")
        if not word:
            return ""
        if re.fullmatch(r"[A-Z]+", word):
            return word
            
        # numbers or lowercase followed by uppercase
        tokens = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", word)
        # acronyms followed by CamelCase (like USAUpdates to USA Updates)
        tokens = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", tokens)
        # words followed by numbers
        tokens = re.sub(r"(?<=[A-Za-z])(?=[0-9])", " ", tokens)
        return tokens

    def _process_text(text):
        if not isinstance(text, str):
            return text
        return re.sub(r"#\w+", lambda m: _split_tag(m.group()), text)

    return series.apply(_process_text)


def replace_emojis(series:pd.Series)->pd.Series:
    return series.apply(lambda text:emoji.demojize(text, delimiters=(" ", " ")))


def remove_stopwords(token_series: pd.Series, 
                     stopwords_list: Iterable[str]=STOPWORDS) -> pd.Series:      
    return token_series.apply(lambda tokens: [word for word in tokens 
                                              if (word not in set(stopwords_list)) and 
                                              (not(len(word)==1) or word.isdigit())])


def lemmatize_series(series, tagger=pos_tag, lemmatizer=_wordnet_lem):
    def _get_wordnet_pos(tag):
        if tag.startswith('J'):
            return 'a'
        elif tag.startswith('V'):
            return 'v'
        elif tag.startswith('N'):
            return 'n'
        elif tag.startswith('R'):
            return 'r'
        else:
            return 's'
    
    def _pos_tag_series(series):
        return series.apply(lambda lt: tagger(lt))

    intermediate_sries = _pos_tag_series(series)
    return intermediate_sries.apply(lambda lt: [lemmatizer.lemmatize(token, _get_wordnet_pos(pos)) for (token, pos) in lt])


def text_cleaner(series: pd.Series)->pd.Series:
    series_changed = series.fillna("")
    series_changed = replace_emojis(series_changed)
    series_changed = split_hashtags(series_changed)
    series_changed = series_changed.str.lower()
    series_changed = replace_urls(series_changed)
    series_changed = replace_mentions(series_changed)
    series_changed = stock_mentions(series_changed)
    series_changed = replace_punctuations(series_changed)
    return series_changed


def pipeline(series: pd.Series, 
             to_clean:bool="True",
             stopwords_list: Iterable[str]=STOPWORDS)->pd.Series:
    if to_clean:
        series_changed = text_cleaner(series)
    else:
        series_changed = series
    series_changed = series_changed.str.split()
    series_changed = remove_stopwords(series_changed, stopwords_list)
    series_changed = lemmatize_series(series_changed)
    return series_changed