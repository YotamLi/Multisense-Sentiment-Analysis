"""
Text preprocessing pipeline for sentiment analysis.

Handles the idiosyncrasies of user-generated text: HTML artifacts, URLs,
social media conventions (@mentions, #hashtags), emoji, informal contractions,
and character repetition common in expressive writing.

The pipeline is intentionally conservative — it normalizes noise without
destroying sentiment-bearing signals (e.g., emoji carry sentiment, repeated
characters indicate emphasis).
"""

from __future__ import annotations

import re
import html

import emoji
import contractions


class TextPreprocessor:
    """Configurable text cleaning pipeline for social media / review text."""

    # Compile patterns once at class level for efficiency
    _URL_RE = re.compile(r"https?://\S+|www\.\S+")
    _HTML_TAG_RE = re.compile(r"<[^>]+>")
    _MENTION_RE = re.compile(r"@\w+")
    _HASHTAG_RE = re.compile(r"#(\w+)")
    _REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")
    _WHITESPACE_RE = re.compile(r"\s+")

    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_html: bool = True,
        normalize_mentions: bool = True,
        expand_hashtags: bool = True,
        demojize: bool = True,
        expand_contractions: bool = True,
        reduce_repeated: bool = True,
        max_repeated: int = 2,
    ):
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_html = remove_html
        self.normalize_mentions = normalize_mentions
        self.expand_hashtags = expand_hashtags
        self.demojize = demojize
        self.expand_contractions = expand_contractions
        self.reduce_repeated = reduce_repeated
        self.max_repeated = max_repeated

    def __call__(self, text: str) -> str:
        return self.clean(text)

    def clean(self, text: str) -> str:
        """Apply the full preprocessing pipeline to a single text."""
        if not text or not isinstance(text, str):
            return ""

        text = html.unescape(text)

        if self.remove_html:
            text = self._HTML_TAG_RE.sub(" ", text)

        if self.remove_urls:
            text = self._URL_RE.sub(" ", text)

        if self.normalize_mentions:
            text = self._MENTION_RE.sub("USER", text)

        if self.expand_hashtags:
            text = self._HASHTAG_RE.sub(r"\1", text)

        if self.demojize:
            text = emoji.demojize(text, delimiters=(" ", " "))

        if self.expand_contractions:
            text = contractions.fix(text)

        if self.reduce_repeated:
            text = self._REPEATED_CHAR_RE.sub(
                r"\1" * self.max_repeated, text
            )

        if self.lowercase:
            text = text.lower()

        text = self._WHITESPACE_RE.sub(" ", text).strip()

        return text

