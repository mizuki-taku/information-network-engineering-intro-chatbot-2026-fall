"""共通ヘッダー（科目名・受講者名と、画面切り替え・よくある質問ボタン）"""
import streamlit as st

from ui import faq_view

VIEW_QA = "qa"
VIEW_HISTORY = "history"
_VIEW_KEY = "current_view"


def current_view() -> str:
    return st.session_state.setdefault(_VIEW_KEY, VIEW_QA)


def _set_view(view: str):
    st.session_state[_VIEW_KEY] = view


def render(subject_name: str, student_name: str):
    """受講者は氏名のみ表示する（学籍番号などの識別子は表示しない）"""
    view = current_view()
    info_col, qa_col, log_col, faq_col = st.columns([5, 2, 2, 3])
    info_col.markdown(f"**科目名**：{subject_name}　**受講者**：{student_name}")
    qa_col.button(
        "Q&A", key="nav_qa", use_container_width=True,
        type="primary" if view == VIEW_QA else "secondary",
        on_click=_set_view, args=(VIEW_QA,),
    )
    log_col.button(
        "過去ログ", key="nav_history", use_container_width=True,
        type="primary" if view == VIEW_HISTORY else "secondary",
        on_click=_set_view, args=(VIEW_HISTORY,),
    )
    faq_col.button("よくある質問", key="nav_faq", use_container_width=True, on_click=faq_view.open_faq)
    st.divider()
