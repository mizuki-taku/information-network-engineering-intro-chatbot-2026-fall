"""「よくある質問」の一覧表示

st.dialog（1.37以降。1.34〜1.36は st.experimental_dialog）が使えればダイアログで、
使えなければ画面内のパネルで表示する。表示のみで、AIの呼び出しやSheetsへの記録は行わない。
"""
import streamlit as st

_OPEN_KEY = "faq_open"
_HIGHLIGHT_KEY = "faq_highlight_ids"

_dialog = getattr(st, "dialog", None) or getattr(st, "experimental_dialog", None)
HAS_DIALOG = _dialog is not None


def open_faq(highlight_ids=()):
    """ボタンの on_click に渡す。highlight_ids のFAQは開いた状態で表示する"""
    st.session_state[_OPEN_KEY] = True
    st.session_state[_HIGHLIGHT_KEY] = tuple(highlight_ids)


def _close_faq():
    st.session_state[_OPEN_KEY] = False


def render_faq_list(faqs, warnings, highlight_ids=()):
    for w in warnings:
        st.warning(w)
    if not faqs:
        st.info("現在、よくある質問はありません。")
        return
    for faq in faqs:
        with st.expander(faq.question, expanded=faq.id in highlight_ids):
            st.markdown(faq.answer)


if HAS_DIALOG:
    @_dialog("よくある質問", width="large")
    def _faq_dialog(faqs, warnings, highlight_ids):
        render_faq_list(faqs, warnings, highlight_ids)


def render(faqs, warnings):
    """ヘッダーの直後に呼ぶ。open_faq が押されていれば一覧を表示する"""
    if not st.session_state.get(_OPEN_KEY):
        return
    highlight_ids = st.session_state.get(_HIGHLIGHT_KEY, ())

    if HAS_DIALOG:
        # ダイアログは×で閉じても再実行されないため、1回表示したらフラグを戻す
        st.session_state[_OPEN_KEY] = False
        _faq_dialog(faqs, warnings, highlight_ids)
        return

    with st.container(border=True):
        title_col, close_col = st.columns([4, 1])
        title_col.markdown("#### よくある質問")
        close_col.button("閉じる", key="faq_close", on_click=_close_faq, use_container_width=True)
        render_faq_list(faqs, warnings, highlight_ids)
