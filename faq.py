"""よくある質問（FAQ）の読み込みと、入力文との照合

faq.yaml の形式:
    faqs:
      - id: attendance              # 必須（重複不可）
        question: 出席は…？          # 必須
        answer: |                   # 必須（複数行のMarkdown可）
          …
        keywords: [出席, 出欠]       # 任意（ないと自動判定の対象外）
        enabled: true               # 任意（falseで非表示かつ判定対象外）
        order: 10                   # 任意（表示順。未指定の項目は指定ありの後ろに記載順で並ぶ）

このモジュールはStreamlitの画面を描画しない。警告は文字列のリストとして返す。
"""
import os
import unicodedata
from dataclasses import dataclass

import streamlit as st
import yaml

KNOWN_KEYS = {"id", "question", "answer", "keywords", "enabled", "order"}


@dataclass(frozen=True)
class Faq:
    id: str
    question: str
    answer: str
    keywords: tuple       # 表示・確認用（元の表記）
    norm_keywords: tuple  # 照合用（正規化済み）
    order: int | None


def normalize(text: str) -> str:
    """NFKC正規化 → 大文字小文字の統一 → 空白・記号・制御文字の除去"""
    text = unicodedata.normalize("NFKC", text).casefold()
    # P: 句読点・括弧など / S: 記号 / Z: 空白 / C: 改行などの制御文字
    return "".join(ch for ch in text if unicodedata.category(ch)[0] not in "PSZC")


def _scalar_to_str(value):
    """YAMLが数値などに解釈した値も文字列として受け付ける（bool・None・リスト等は不可）"""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (str, int, float)):
        return str(value).strip()
    return None


def _parse_item(raw, index: int, seen_ids: set):
    """1項目を検証する。戻り値は (Faq | None, 警告のリスト)。不正ならFaqはNone"""
    label = f"faq.yaml の{index + 1}番目の項目"
    if not isinstance(raw, dict):
        return None, [f"{label}：「キー: 値」の形式になっていないため、スキップしました。"]

    label = f"{label}（id: {raw.get('id', '未指定')}）"
    warnings = []
    unknown = sorted(str(k) for k in raw if k not in KNOWN_KEYS)
    if unknown:
        warnings.append(f"{label}：未知のキー {', '.join(unknown)} は無視しました（綴りを確認してください）。")

    def invalid(reason):
        return None, warnings + [f"{label}：{reason}ため、スキップしました。"]

    values = {}
    for key in ("id", "question", "answer"):
        value = _scalar_to_str(raw.get(key))
        if not value:
            return invalid(f"必須項目 {key} が空か不正な")
        values[key] = value
    if values["id"] in seen_ids:
        return invalid(f"id「{values['id']}」が重複している")

    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        return invalid("enabled が true / false ではない")

    order = raw.get("order")
    if order is not None and (isinstance(order, bool) or not isinstance(order, int)):
        return invalid("order が整数ではない")

    keywords_raw = raw.get("keywords")
    if keywords_raw is None:
        keywords_raw = []
    elif not isinstance(keywords_raw, list):
        keywords_raw = [keywords_raw]  # 「keywords: 出席」のような1語の書き方も許可する
    keywords, norm_keywords = [], []
    for kw in keywords_raw:
        kw_str = _scalar_to_str(kw)
        if kw_str is None:
            return invalid(f"keywords に文字列でない値（{kw!r}）がある")
        norm = normalize(kw_str)
        if not norm:
            warnings.append(f"{label}：キーワード「{kw_str}」は記号や空白だけのため、照合に使いません。")
            continue
        keywords.append(kw_str)
        norm_keywords.append(norm)

    seen_ids.add(values["id"])
    if not enabled:
        return None, warnings
    faq = Faq(
        id=values["id"],
        question=values["question"],
        answer=values["answer"],
        keywords=tuple(keywords),
        norm_keywords=tuple(norm_keywords),
        order=order,
    )
    return faq, warnings


@st.cache_data(show_spinner=False)
def _load_faqs_cached(path: str, mtime: float):
    """mtime はキャッシュキーとしてだけ使う（ファイルが更新されると再読み込みされる）"""
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as e:
        return [], [f"faq.yaml を読み込めませんでした（よくある質問は表示されません）: {e}"]

    if data is None:
        return [], []
    if not isinstance(data, dict) or "faqs" not in data or not isinstance(data["faqs"] or [], list):
        return [], ["faq.yaml の形式が不正です。先頭を「faqs:」とし、その下に項目を「- id: …」の形で並べてください。"]

    faqs, warnings, seen_ids = [], [], set()
    for index, raw in enumerate(data.get("faqs") or []):
        faq, item_warnings = _parse_item(raw, index, seen_ids)
        warnings.extend(item_warnings)
        if faq is not None:
            faqs.append((index, faq))

    # order 指定ありを先に（order順）、指定なしはその後ろに記載順で並べる
    faqs.sort(key=lambda p: (p[1].order is None, p[1].order or 0, p[0]))
    return [faq for _, faq in faqs], warnings


def load_faqs(path: str):
    """FAQを読み込む。戻り値は (Faqのリスト, 警告のリスト)。ファイルがなければ空"""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return [], []
    faqs, warnings = _load_faqs_cached(path, mtime)
    for w in warnings:
        print(f"[faq] {w}")
    return faqs, warnings


def match_faqs(text: str, faqs, max_chars: int, max_results: int):
    """入力文に keywords のいずれかを含むFAQを、表示順で最大 max_results 件返す。

    長い感想に偶然キーワードが入る誤判定を避けるため、入力文（前後の空白を除く）が
    max_chars 文字を超える場合は判定しない。
    """
    text = (text or "").strip()
    if not text or len(text) > max_chars:
        return []
    norm_text = normalize(text)
    if not norm_text:
        return []
    matched = [faq for faq in faqs if any(kw in norm_text for kw in faq.norm_keywords)]
    return matched[:max_results]


def build_redirect_message(matched, template: str, item_format: str) -> str:
    """誘導メッセージを組み立てる。template の {faq_questions} に該当FAQの一覧が入る"""
    lines = "\n".join(item_format.format(question=faq.question) for faq in matched)
    return template.format(faq_questions=lines)
