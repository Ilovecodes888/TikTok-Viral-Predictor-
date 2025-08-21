
from __future__ import annotations
from sklearn.base import BaseEstimator, TransformerMixin
from .utils import count_emojis

class TextPrep(BaseEstimator, TransformerMixin):
    def __init__(self, text_cols=('title','script','hashtags')):
        self.text_cols = text_cols
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        df = X.copy()
        df['text'] = df[list(self.text_cols)].fillna('').agg(' '.join, axis=1)
        df['n_chars'] = df['text'].str.len()
        df['n_words'] = df['text'].str.split().str.len()
        df['n_hashtags'] = df['text'].str.count(r'#\w+')
        df['n_emojis'] = df['text'].apply(count_emojis)
        df['n_exclaim'] = df['text'].str.count('!')
        df['n_question'] = df['text'].str.count('\?')
        df['sent_proxy'] = df['text'].str.lower().str.contains("good|great|wow|amazing|glow|smooth|clear").astype(int) -                            df['text'].str.lower().str.contains("bad|hate|slow|fail|irritation").astype(int)
        return df[['text','n_chars','n_words','n_hashtags','n_emojis','n_exclaim','n_question','sent_proxy']]
