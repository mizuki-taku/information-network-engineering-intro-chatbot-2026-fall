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
    
# Google Sheets に接続
def get_gsheet():
    import json
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds_dict = json.loads(st.secrets["GSPREAD_SERVICE_ACCOUNT"])
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    sheet = client.open(st.secrets["SHEET_NAME"]).sheet1
    return sheet
    
# 会話履歴を1行だけGoogle Sheetsに保存（student_idは常にanonymous）
def save_single_turn_to_sheet(user_query, assistant_response, student_id, student_name):
    sheet = get_gsheet()
    timestamp = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d %H:%M:%S")
    sheet.append_row([timestamp, student_id, student_name, user_query, assistant_response])
    
def fetch_recent_history_text(student_id: str, limit: int = 10) -> list:
    """指定 student_id の履歴をリスト形式で取得（改行対策済み）"""
    if not student_id:
        return []

    sheet = get_gsheet()
    rows = sheet.get_all_values()
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

# Streamlitのヘッダー
st.title("質問応答チャットボット（情報技術者キャリアデザイン入門）")

# --- Moodleからパラメータ受け取り ---
params = st.query_params
student_id   = st.query_params.get("student_id",   "anonymous")
student_name = st.query_params.get("student_name", "不明")

# セッションステートでメッセージの履歴を保持
if "messages" not in st.session_state:
    st.session_state.messages = []

    # ▼ 過去10件の会話履歴を取得（修正版：リストが返ってくる）
    history_data = fetch_recent_history_text(student_id, limit=10)

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

# --- フォルダの読み込み処理（変更なし） ---
lecture_folder = "./it-career-design"  # 講義資料フォルダ
example_folder = "./it-career-design_example"     # 回答例フォルダ
log_folder = "./logs"             # 会話ログ保存フォルダ

folders_to_load = [lecture_folder]
if os.path.exists(example_folder):
    folders_to_load.append(example_folder)

# リスト → タプルに変換してから渡す
combined_index = load_and_index_multiple_folders(tuple(folders_to_load))

# セッション状態でバナーの表示・非表示を管理するフラグを初期化
if "welcome_hidden" not in st.session_state:
    st.session_state.welcome_hidden = False

# --- 1. 過去のチャット履歴を画面に表示する ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 2. ユーザー情報（送信ボタンが押されるまで表示） ---
if not st.session_state.welcome_hidden:
    st.info(f"ようこそ {student_name} さん (学籍番号: {student_id})")

# --- 3. ユーザー入力エリア（フォーム） ---
with st.form(key='chat_form', clear_on_submit=True):
    query = st.text_area("質問を入力してください（Ctrl + Enterで送信）:", key="user_input_area")
    submit_button = st.form_submit_button("送信")

# --- 4. 応答処理 ---
if submit_button and query:
    # 送信された瞬間にフラグをTrueにする
    st.session_state.welcome_hidden = True
    
    st.session_state.messages.append({"role": "user", "content": query})
    
    response = search_index(
        combined_index,
        query,
        history_pairs=st.session_state.messages[-20:]
    )

    st.session_state.messages.append({"role": "assistant", "content": response})
    save_single_turn_to_sheet(query, response, student_id, student_name)
    
    # 画面を更新してバナーを消去
    st.rerun()
