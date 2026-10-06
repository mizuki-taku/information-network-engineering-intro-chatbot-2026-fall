"""過去ログ画面（質問とAIの回答のペアを、スクロールできる枠内に並べる）"""
import unicodedata

import streamlit as st

from ui.common import keep_widget_value, markdown_keep_breaks, render_response, save_widget_value

_SEARCH_KEY = "history_search"
_SEARCH_STORE_KEY = "history_search_saved"


def build_pairs(messages):
    """[{"role": "user"}, {"role": "assistant"}, ...] を質問と回答のペアにする（古い順）"""
    pairs = []
    for msg in messages:
        if msg["role"] == "user":
            pairs.append({"query": msg["content"], "response": ""})
        elif msg["role"] == "assistant" and pairs and not pairs[-1]["response"]:
            pairs[-1]["response"] = msg["content"]
    return pairs


def _fold(text: str) -> str:
    """検索用。全角・半角と大文字・小文字の違いを無視する"""
    return unicodedata.normalize("NFKC", text or "").casefold()


def filter_pairs(pairs, keyword: str):
    keyword = _fold(keyword.strip())
    if not keyword:
        return pairs
    return [p for p in pairs if keyword in _fold(p["query"]) or keyword in _fold(p["response"])]


def render(messages, height: int, newest_first: bool, redirect_prefix: str, fetch_limit: int):
    keep_widget_value(_SEARCH_KEY, _SEARCH_STORE_KEY)
    keyword = st.text_input(
        "キーワード検索",
        key=_SEARCH_KEY,
        placeholder="質問・回答に含まれる語句で絞り込み",
        on_change=save_widget_value,
        args=(_SEARCH_KEY, _SEARCH_STORE_KEY),
    )

    pairs = build_pairs(messages)
    hits = filter_pairs(pairs, keyword)
    if keyword.strip():
        st.caption(f"{len(hits)} 件ヒットしました（全 {len(pairs)} 件）")
    else:
        st.caption(f"全 {len(pairs)} 件（以前のやり取りは直近 {fetch_limit} 件まで表示されます）")
    if newest_first:
        hits = hits[::-1]

    # st.chat_message を入れると枠が末尾（新しい順では最も古いもの）へ自動スクロールするため使わない
    with st.container(height=height):
        if not pairs:
            st.info("まだやり取りはありません。")
        elif not hits:
            st.info("該当するやり取りはありません。")
        for pair in hits:
            with st.container(border=True):
                st.caption("質問")
                markdown_keep_breaks(pair["query"])
                st.caption("AIの回答")
                render_response(pair["response"], redirect_prefix)
