import os
import gspread
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime  
from zoneinfo import ZoneInfo
from oauth2client.service_account import ServiceAccountCredentials
from file_loader import load_pdf
from file_loader import load_text
from file_loader import load_docx
from text_splitter import split_text
from faiss_indexer import load_and_index_folder, search_index, create_faiss_index
from faq import load_faqs, match_faqs, build_redirect_message
from ui import header, qa, history, faq_view

# ===== UI設定（科目ごとに変更するのはここだけ） =====
SUBJECT_NAME = "情報ネットワーク工学入門"
FAQ_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faq.yaml")
HISTORY_HEIGHT = 520          # 過去ログ枠の高さ（px）
HISTORY_NEWEST_FIRST = True   # 過去ログを新しい順に並べる（Falseで古い順）
HISTORY_FETCH_LIMIT = 10      # 起動時にGoogle Sheetsから読み込む過去のやり取りの件数
FAQ_MATCH_MAX_CHARS = 60      # この文字数以下の入力だけをFAQ判定の対象にする
FAQ_MATCH_MAX_ITEMS = 3       # 誘導メッセージで案内するFAQの最大件数
FAQ_REDIRECT_ITEM_FORMAT = "・{question}"
FAQ_REDIRECT_TEMPLATE = (     # {faq_questions} に該当したFAQの質問（上の形式で1行ずつ）が入る
    "この内容は「よくある質問」に回答があります。\n"
    "{faq_questions}\n"
    "\n"
    "画面右上の「よくある質問」ボタンから確認できます。\n"
    "ほかにも質問や感想があれば、ぜひ入力してください。"
)
# 過去ログの表示で誘導メッセージを見分けるための先頭部分
FAQ_REDIRECT_PREFIX = FAQ_REDIRECT_TEMPLATE.split("{")[0].splitlines()[0]

st.set_page_config(page_title=f"質問応答チャットボット（{SUBJECT_NAME}）")

# 環境変数のロード（必要に応じて）
load_dotenv()

# インデックスの作成
@st.cache_resource
def load_and_index_multiple_folders(folders):
    all_texts = []
    for folder in folders:
        texts = load_and_index_folder(folder, return_documents=True)
        all_texts.extend(texts)
    return create_faiss_index(all_texts)
    
# Google Sheets に接続（未設定・認証失敗時は None を返す）
def get_gsheet():
    import json
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_dict = json.loads(st.secrets["GSPREAD_SERVICE_ACCOUNT"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sheet = client.open(st.secrets["SHEET_NAME"]).sheet1
        return sheet
    except Exception as e:
        st.warning(f"Google Sheetsへの接続に失敗しました（会話ログは保存・取得されません）: {e}")
        return None

# 会話履歴を1行だけGoogle Sheetsに保存（student_idは常にanonymous）
def save_single_turn_to_sheet(user_query, assistant_response, student_id, student_name):
    sheet = get_gsheet()
    if sheet is None:
        return
    try:
        timestamp = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([timestamp, student_id, student_name, user_query, assistant_response])
    except Exception as e:
        st.warning(f"会話ログの保存に失敗しました: {e}")

def fetch_recent_history_text(student_id: str, limit: int = 10) -> list:
    """指定 student_id の履歴をリスト形式で取得（改行対策済み）"""
    if not student_id:
        return []

    sheet = get_gsheet()
    if sheet is None:
        return []

    try:
        rows = sheet.get_all_values()
    except Exception as e:
        st.warning(f"会話履歴の取得に失敗しました: {e}")
        return []

    if len(rows) <= 1:
        return []

    header = rows[0]
    # 列番号の特定（query と response に対応）
    col_sid = header.index("student_id") if "student_id" in header else 1
    col_q = header.index("user_query") if "user_query" in header else 3 # 保存時の名前に合わせる
    col_r = header.index("assistant_response") if "assistant_response" in header else 4

    pairs = []
    # 新しい順にスキャン
    for r in reversed(rows[1:]):
        if len(r) > col_r and r[col_sid].strip() == (student_id or "").strip():
            # QとAをセットにして保存（ここではまだ整形しない）
            pairs.append({"query": r[col_q], "response": r[col_r]})
        if len(pairs) >= limit:
            break

    # 表示用に古い順に戻してリストで返す
    return pairs[::-1]

# 指定 student_id の最後の1行をGoogle Sheetsから削除する（取り消し用）
def delete_last_turn_from_sheet(student_id: str):
    if not student_id:
        return
    try:
        sheet = get_gsheet()
        rows = sheet.get_all_values()
        if len(rows) <= 1:
            return

        header = rows[0]
        col_sid = header.index("student_id") if "student_id" in header else 1

        # 末尾（最新）側から自分の行を探して削除
        for i in range(len(rows) - 1, 0, -1):  # rows[0] はヘッダー行
            if len(rows[i]) > col_sid and rows[i][col_sid].strip() == (student_id or "").strip():
                sheet.delete_rows(i + 1)  # gspreadの行番号は1始まり
                break
    except Exception as e:
        st.warning(f"Google Sheets上のログ削除に失敗しました: {e}")

# --- Moodleからパラメータ受け取り ---
params = st.query_params
student_id   = st.query_params.get("student_id",   "anonymous")
student_name = st.query_params.get("student_name", "不明")

# セッションステートでメッセージの履歴を保持
if "messages" not in st.session_state:
    st.session_state.messages = []

    # ▼ 過去10件の会話履歴を取得（修正版：リストが返ってくる）
    history_data = fetch_recent_history_text(student_id, limit=HISTORY_FETCH_LIMIT)

    # 文字列分割をやめ、リストから直接 session_state に入れる
    for item in history_data:
        st.session_state.messages.append({
            "role": "user",
            "content": item["query"]
        })
        st.session_state.messages.append({
            "role": "assistant",
            "content": item["response"]
        })

# --- フォルダの読み込み処理 ---
lecture_folder = "./information-network-engineering-intro"
example_folder = "./information-network-engineering-intro_example"
log_folder = "./logs"             # 会話ログ保存フォルダ

folders_to_load = [lecture_folder]
if os.path.exists(example_folder):
    folders_to_load.append(example_folder)

# リスト → タプルに変換してから渡す
combined_index = load_and_index_multiple_folders(tuple(folders_to_load))

# --- よくある質問（faq.yaml）の読み込み（更新時刻が変わると再読み込み） ---
faqs, faq_warnings = load_faqs(FAQ_FILE_PATH)

# --- 応答処理（送信後、qa.process_pending からスピナー表示の内側で呼ばれる） ---
def answer_query(query):
    # 回答生成中に別のボタンが押されても、回答の受け取りからSheetsへの記録までが
    # 打ち切られないよう、st.session_state には最初の1回だけ触れて履歴リストを直接更新する
    messages = st.session_state.messages

    # FAQに該当する短い入力はAIを呼ばず、「よくある質問」へ誘導する（判定はキーワード照合のみ）
    matched_faqs = match_faqs(query, faqs, FAQ_MATCH_MAX_CHARS, FAQ_MATCH_MAX_ITEMS)

    messages.append({"role": "user", "content": query})

    if matched_faqs:
        response = build_redirect_message(matched_faqs, FAQ_REDIRECT_TEMPLATE, FAQ_REDIRECT_ITEM_FORMAT)
    else:
        response = search_index(
            combined_index,
            query,
            history_pairs=messages[-20:]
        )

    messages.append({"role": "assistant", "content": response})
    # FAQへ誘導した入力も、出席の提出として通常どおり記録する
    save_single_turn_to_sheet(query, response, student_id, student_name)
    return {"query": query, "response": response, "faq_ids": [faq.id for faq in matched_faqs]}

# --- 画面 ---
header.render(SUBJECT_NAME, student_name)
faq_view.render(faqs, faq_warnings)

if header.current_view() == header.VIEW_QA:
    qa.render(answer_query, FAQ_REDIRECT_PREFIX)
else:
    # 回答の作成中に過去ログへ切り替えた場合も、回答と記録が済んでから表示する
    qa.process_pending(answer_query)
    history.render(
        st.session_state.messages,
        HISTORY_HEIGHT,
        HISTORY_NEWEST_FIRST,
        FAQ_REDIRECT_PREFIX,
        HISTORY_FETCH_LIMIT,
    )

# --- 直前のやり取りを取り消す（表示上の履歴と、Google Sheets上の最後の1行を削除） ---
# 送信後の状態を反映させるため、本文の描画後にサイドバーを描画する
with st.sidebar:
    if st.button("↩️ 直前のやり取りを取り消す", disabled=len(st.session_state.messages) < 2):
        delete_last_turn_from_sheet(student_id)
        st.session_state.messages = st.session_state.messages[:-2]
        qa.clear_last_turn()
        st.rerun()
