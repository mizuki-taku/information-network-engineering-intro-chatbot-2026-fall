"""画面部品で共通に使う処理"""
import streamlit as st


def keep_widget_value(widget_key: str, store_key: str, default=""):
    """画面切り替えで入力欄が描画されないと値が消えるため、別キーに退避した値から復元する。

    入力欄を描画する直前に呼び、入力欄の on_change で save_widget_value を呼ぶ。
    """
    if widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state.get(store_key, default)


def save_widget_value(widget_key: str, store_key: str):
    st.session_state[store_key] = st.session_state[widget_key]


def markdown_keep_breaks(text: str):
    """改行をそのまま表示する（Markdownでは1つの改行が無視されるため、行末に改行記号を補う）"""
    st.markdown((text or "").replace("\n", "  \n"))


def render_response(text: str, redirect_prefix: str):
    """応答を表示する。FAQへの誘導メッセージは改行を保って表示し、AIの回答は従来どおりMarkdownで表示する"""
    if redirect_prefix and (text or "").startswith(redirect_prefix):
        markdown_keep_breaks(text)
    else:
        st.markdown(text)
