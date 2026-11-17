from __future__ import annotations

import re

import jieba


CHAPTER_PATTERN = re.compile(r"^第[一二三四五六七八九十]+节\s*[\u4e00-\u9fff、（）()A-Za-z0-9 ]{2,40}$")
SPACE_PATTERN = re.compile(r"[ \t\u3000]+")


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u3000", " ")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return "\n".join(SPACE_PATTERN.sub(" ", line).strip() for line in text.splitlines())


def tokenize(text: str) -> list[str]:
    """Chinese word tokens plus ASCII words and character bigrams."""
    text = text.lower()
    tokens: list[str] = []
    for token in jieba.lcut(text, cut_all=False):
        token = token.strip()
        if not token:
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            tokens.append(token)
            if len(token) >= 3:
                tokens.extend(token[i : i + 2] for i in range(len(token) - 1))
        elif re.fullmatch(r"[a-z0-9._%-]+", token):
            tokens.append(token)
    return tokens


def infer_chapter(line: str, current: str) -> str:
    compact = re.sub(r"\s+", " ", line).strip()
    if CHAPTER_PATTERN.match(compact):
        return compact
    return current


def split_sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[。！？；])\s*|\n+", text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]

