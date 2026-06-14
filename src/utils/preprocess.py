import re
import emoji
import contractions

import pandas as pd

from typing import Iterable

from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import stopwords

STOPWORDS = list(stopwords.words("English"))
_wordnet_lem = WordNetLemmatizer()


def replace_urls(series: pd.Series,
                 replace_with:str="URL") -> pd.Series:
    """
    Returns a series with all links replaced in original Pandas Series to `replace_with`.

    Parameters
    ----------
    series: pandas.Series
        The series to be cleaned.

    replace_with: str, default="URL"
        A string that will replace all links in Series.

    Returns
    -------
    pandas.Series
        A Series with altered values.
    """
    return series.str.replace(
        r"(?i)(https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,63}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*))",
        replace_with,
        regex=True,
    )


def replace_mentions(series:pd.Series,
                     replace_with:str="REF") -> pd.Series:
    """
    Returns a series with all references (i.e. '@numpy') replaced in original Pandas Series to `replace_with`.

    Parameters
    ----------
    series: pandas.Series
        The series to be cleaned.

    replace_with: str, default="REF"
        A string that will replace all references in Series.

    Returns
    -------
    pandas.Series
        A Series with altered values.
    """
    return series.str.replace(
        r'@\w+', 
        replace_with, 
        regex=True
        )


def stock_mentions(series:pd.Series,
                   replace_with:str="STOCK") -> pd.Series:
    """
    Returns a series with all stock mentions (e.g. '$NVDA') replaced in original Pandas Series to `replace_with`.

    Parameters
    ----------
    series: pandas.Series
        The series to be cleaned.

    replace_with: str, default="STOCK"
        A string that will replace all stock mentions in Series.

    Returns
    -------
    pandas.Series
        A Series with altered values.
    """
    return series.str.replace(
        r'\$[A-Za-z]+', 
        replace_with, 
        regex=True
        )


def replace_punctuations(series:pd.Series)->pd.Series:
    """
    Returns a series without punctuation.

    Parameters
    ----------
    series: pandas.Series
        The series to be cleaned.

    Returns
    -------
    pandas.Series
        A Series with altered values.
    """

    def _replace_punctuations_unit(text):
        """
        Returns a str without punctuation.

        Parameters
        ----------
        text: str or None
            Text to be cleaned.

        Returns
        -------
        str
            Altered text.
        """
        text = str(text)

        text = contractions.fix(text) #fixes apostophres, like You're into You are

        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text) #regex to remove these characters, encoding issues or old ASCII formating

        text = re.sub(r"[–—−]", "-", text) #regex to put every dash type to one single type
        
        text = re.sub(r'(\d)\s*-\s*(\d)', r'\1 to \2', text) #if "-" between 2 numbers change it to "to", but remove this if causing problems
        
        text = re.sub(r'\s-(?=\d)', ' minus ', text) #regex to convert a hyphen to a minus sign token if followed by a number

        text = re.sub(r"'s", " ", text) #getting rid of possessive nouns

        text = re.sub(r"[_ƒº½]", " ", text) #getting rid of unhandled characters
        
        text = re.sub(r'%', ' percent', text) #regex for % to appear as percent

        text = re.sub(r"[^\w\s]", " ", text) #removing punctuation but apostophres already taken care by contractions

        text = re.sub(r'\s+', ' ', text) #normalizing whitespace (if sequence of whitespaces, make it justa a whitespace)

        return text.strip()
    return series.apply(_replace_punctuations_unit)


def split_hashtags(series: pd.Series) -> pd.Series:
    """
    Returns a series with hashtags splitted. 
    
    (e.g. '#ILovePython' -> 'I Love Python')

    Parameters
    ----------
    series: pandas.Series
        The series to be cleaned.

    Returns
    -------
    pandas.Series
        A Series with altered values.
    """
    def _split_tag(tag: str) -> str:
        """
        Returns a tag with hashtags splitted.

        Works only with string that are fully a hashtag.
        (e.g. '#ILovePython' -> 'I Love Python')

        Parameters
        ----------
        tag: str
            The text to be cleaned.

        Returns
        -------
        str
            A text with altered values.
        """
        # get rid of hashtag
        word = tag.lstrip("#")
        if not word:
            return ""
        # if word is all upper case return
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
        """
        Returns a tag with hashtags splitted.

        Finds hashtag pattern and chages it (e.g. '#ILovePython' -> 'I Love Python').

        Parameters
        ----------
        text: str
            The text to be cleaned.

        Returns
        -------
        str
            A text with altered values.
        """
        if not isinstance(text, str):
            return text
        return re.sub(r"#\w+", lambda m: _split_tag(m.group()), text)

    return series.apply(_process_text)


def replace_emojis(series:pd.Series)->pd.Series:
    """
    Returns a series without emojis.

    Parameters
    ----------
    series: pandas.Series
        The series to be cleaned.

    Returns
    -------
    pandas.Series
        A series with altered values.
    """
    return series.apply(lambda text:emoji.demojize(text, delimiters=(" ", " ")))


def remove_stopwords(token_series: pd.Series, 
                     stopwords_list: Iterable[str]=STOPWORDS) -> pd.Series:  
    """
    Returns a series without stopwords.

    Iterates over series of lists of tokens and eliminated provided stopwords.

    Parameters
    ----------
    token_series: pandas.Series
        The series to be cleaned.
    stopwords_list: Iterable[str], default=list(nltk.corpus.stopwords.words("English"))
        Stopwords to remove. Dafault value is english stopwords provided by nltk.
    
    Returns
    -------
    pandas.Series
        A series with altered values.
    """
    return token_series.apply(lambda tokens: [word for word in tokens  # each word
                                              if (word not in set(stopwords_list)) and  # that is not in stopwords
                                              (not(len(word)==1) or word.isdigit())]) # and loger than one, unless it's a digit


def lemmatize_series(series:pd.Series, 
                     tagger=pos_tag, 
                     lemmatizer=_wordnet_lem)->pd.Series:
    """
    Returns a series with lemmatized tokens.

    Iterates over series of lists of tokens and lemmatizes them.

    Parameters
    ----------
    series: pandas.Series
        The series to be cleaned.

    tagger: default=nltk.tag.pos_tag
        POS Tagger to improve lemmetizing.

    lemmatizer: default=nltk.stem.WordNetLemmatizer
        A model to perform lemmatization
    
    Returns
    -------
    pandas.Series
        A series with altered values.
    """
    def _get_wordnet_pos(tag):
        """
        Changes from nltk.tag format to Wordnet POS tag format.

        Parameters
        ----------
        tag: str
            A POS tag.
        
        Returns
        -------
        str
            Modified POS tag 
        """
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
        """
        Gets POS tags for every list of tokens in series.

        Parameters
        ----------
        series: pandas.Series
            Series to be POS tagged.
        
        Returns
        -------
        pandas.Series
            Modified Series. 
        """
        return series.apply(lambda lt: tagger(lt))

    intermediate_sries = _pos_tag_series(series)
    return intermediate_sries.apply(lambda lt: [lemmatizer.lemmatize(token, _get_wordnet_pos(pos)) for (token, pos) in lt])


def text_cleaner(series: pd.Series)->pd.Series:
    """
    Returns a series with fully cleaned text.

    Replaces emojis, urls, stock mentions, references punctuation and splits hashtags.

    Parameters
    ----------
    series: pandas.Series
        The series to be cleaned.

    Returns
    -------
    series_changed: pandas.Series
        A series with altered values."""
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
             to_clean:bool=True,
             stopwords_list: Iterable[str]=STOPWORDS)->pd.Series:
    """
    Returns a series with fully cleaned tokens.

    Replaces emojis, urls, stock mentions, references punctuation and splits hashtags if `to_clean`.
    Then tokenizes series, removes stopwords from `stopwords_list` and lemmatizes tokens.

    Parameters
    ----------
    series: pandas.Series
        The series to be cleaned.

    to_clean: bool, default=True
        Whether to clean the data.

    stopwords_list: Iterable[str], default=list(nltk.corpus.stopwords.words("English"))
        An iterable of stopwords to remove.
        
    Returns
    -------
    series_changed: pandas.Series
        A series with altered values.
    """
    if to_clean:
        series_changed = text_cleaner(series)
    else:
        series_changed = series
    series_changed = series_changed.str.split()
    series_changed = remove_stopwords(series_changed, stopwords_list)
    series_changed = lemmatize_series(series_changed)
    return series_changed