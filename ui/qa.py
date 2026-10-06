"""Q&A画面（上に入力欄、その直下にAIの回答）

st.chat_input は画面下に固定され回答が見えにくくなるため使わない。
st.form 内の入力は送信まで保持されず、画面を切り替えると入力途中の内容が消えるため、
入力欄は st.text_area にして内容を session_state に退避している。
"""
import time

import streamlit as st

from ui import faq_view
from ui.common import keep_widget_value, markdown_keep_breaks, render_response, save_widget_value

_INPUT_KEY = "qa_input"
_DRAFT_KEY = "qa_draft"
_STATE_KEY = "qa_state"     # 処理待ちの入力・直前のやり取りなどを入れる辞書
_WAIT_TIMEOUT_SECS = 300    # 別の実行が回答を作成中のとき、完了を待つ上限


def _state() -> dict:
    """Q&A画面の状態（通常の辞書）。

    Streamlitは st.session_state を読み書きするたびに、別のボタン操作による再実行で
    処理を打ち切ることがある。回答を受け取ってからSheetsに記録するまでの間に
    打ち切られないよう、st.session_state ではなくこの辞書を直接書き換える。
    """
    return st.session_state.setdefault(_STATE_KEY, {"pending": None, "last_turn": None, "empty_submit": False})


def clear_last_turn():
    _state()["last_turn"] = None


def _submit():
    """送信ボタンの on_click。入力内容を処理待ちに移し、入力欄をクリアする"""
    state = _state()
    query = st.session_state.get(_INPUT_KEY, "")
    if not query.strip():
        state["empty_submit"] = True
        return
    state["pending"] = query
    st.session_state[_INPUT_KEY] = ""
    st.session_state[_DRAFT_KEY] = ""


def process_pending(answer_query):
    """送信済みの入力があれば回答を作成する。どの画面からでも呼べる。

    回答作成中にボタンが押されると、Streamlitは古い実行をAIの応答待ちのまま残して
    新しい実行を並行して始める。同じ入力でAIを二重に呼ばないよう、最初の実行だけが処理し、
    後から始まった実行は完了を待つ。
    """
    state = _state()
    if state["pending"] is None:
        return
    with st.spinner("回答を作成しています…"):
        token = object()
        if state.setdefault("processing", token) is not token:
            deadline = time.monotonic() + _WAIT_TIMEOUT_SECS
            while "processing" in state and time.monotonic() < deadline:
                time.sleep(0.2)
            return
        # ここから記録までは、画面要素を出さず st.session_state にも触れない（打ち切られないように）
        try:
            state["last_turn"] = answer_query(state["pending"])
        finally:
            # エラー時も処理待ちを消す（同じ入力を再実行のたびに送り直さないように）
            state["pending"] = None
            state.pop("processing", None)


def render(answer_query, redirect_prefix: str):
    """answer_query(query) は {"query", "response", "faq_ids"} を返す（AI呼び出し・記録は呼び出し側）"""
    state = _state()
    keep_widget_value(_INPUT_KEY, _DRAFT_KEY)
    st.text_area(
        "質問はこちら",
        key=_INPUT_KEY,
        height=120,
        placeholder="講義についての質問や感想を入力してください",
        on_change=save_widget_value,
        args=(_INPUT_KEY, _DRAFT_KEY),
    )
    st.button("送信", key="qa_submit", type="primary", on_click=_submit)
    if state["empty_submit"]:
        state["empty_submit"] = False
        st.warning("質問や感想を入力してから送信してください。")

    st.markdown("#### AIの回答")
    process_pending(answer_query)

    turn = state["last_turn"]
    if not turn:
        st.caption("質問や感想を送信すると、ここに回答が表示されます。")
        return

    with st.container(border=True):
        st.caption("あなたの質問")
        markdown_keep_breaks(turn["query"])
    render_response(turn["response"], redirect_prefix)
    if turn.get("faq_ids"):
        st.button(
            "よくある質問を開く", key="qa_open_faq",
            on_click=faq_view.open_faq, args=(turn["faq_ids"],),
        )
