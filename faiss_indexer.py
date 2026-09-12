import os
import openai
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from file_loader import load_pdf, load_text, load_docx
from text_splitter import split_text

# .envファイルの内容を読み込み
load_dotenv()

# 環境変数からAPIキー、モデル、温度を取得
openai.api_key = os.getenv("OPENAI_API_KEY")
OPENAI_API_TEMPERATURE  = float(os.getenv("OPENAI_API_TEMPERATURE", "0.7"))
OPENAI_API_MAX_TOKENS   = int(os.getenv("OPENAI_API_MAX_TOKENS",   "1000"))
OPENAI_API_TOP_K        = int(os.getenv("OPENAI_API_TOP_K",        "3"))
EMBEDDING_MODEL_NAME    = os.getenv("OPENAI_EMBEDDING_MODEL",      "text-embedding-3-small")
OPENAI_API_MODEL        = os.getenv("OPENAI_API_MODEL",            "gpt-4o")

# 埋め込みとインデックス作成
def create_faiss_index(texts):
    """
    Document群をベクトル化してFAISSインデックスを構築
    Embeddingモデルは環境変数 OPENAI_EMBEDDING_MODEL で指定
    """
    embeddings = OpenAIEmbeddings(
        api_key=openai.api_key,
        model=EMBEDDING_MODEL_NAME
    )
    return FAISS.from_documents(texts, embeddings)
    

def search_docs(faiss_index, query):
    """FAISSで検索して上位チャンクを返す"""
    return faiss_index.similarity_search(query, k=OPENAI_API_TOP_K)

# クエリ検索とChatGPT 4.0oでの応答生成
def search_index(faiss_index, query, history_pairs=None):
    """
    history_pairs: [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
    ]
    """

    # FAISSインデックスで検索
    results = faiss_index.similarity_search(query)
    content = results[0].page_content if results else "該当する情報が資料内に見つかりませんでした。"

    # system プロンプト（役割定義）
    system_message = {
        "role": "system",
        "content": (
                #学生の質問やコメントに対する基本的な役割とアプローチ
                "あなたは講義資料に基づき、学生のキャリアに関する質問やコメントに正確かつ簡潔に回答するアシスタントです。"
                "丁寧語で回答し、感謝の言葉や『先生の回答：』といった接頭辞は含めないでください。"
                
                #範囲外・不確定な情報への対応
                "講義の範囲を超えたキャリア相談には、その旨を伝えつつ一般的な業界動向や職種知識で簡潔に補足してください。"
                "感想が資料と直接関係なくても、将来のキャリアや自己実現に関連付けて共感・補足を行ってください。"
                "推測による不確定な情報は提供せず、資料またはIT業界の一般的な知見で補完してください。"
                
                #抽象的な発言に対する具体化支援
                "「やりがいが大事」「スキルを上げたい」等の抽象的な発言には、『思いつきで大丈夫ですよ』と添えて具体的なキャリアイメージを促してください。"
                "1.【ロールモデル】：『身近な人や有名なエンジニアで、理想に近い人はいますか？』"
                "2.【価値観】：『それを大切にしたいと思うようになった、具体的な経験はありますか？』"
                "3.【アクション】：『その目標に向けて、今すぐ始められそうな小さな活動は何だと思いますか？』"
                
                #伴走型支援と問いかけ
                "回答は必ず『共感』や『肯定』から始め、最後は自己分析を深めるハードルの低い問いかけを1つ添えてください。"
                "1.【興味の深掘り】：『紹介した職種の中で、直感的に「自分に合いそう」と感じたものはどれですか？』"
                "2.【社会との接続】：『あなたが普段使っているITサービスは、どんな職種の人が作っていそうですか？』"
                "3.【次の一歩】：『今の自分の強みをさらに知るために、次は〇〇について調べてみませんか？』"
                "学生が将来像に悩んでいる場合も、現在の立ち位置を一緒に整理する優しい姿勢を保ってください。"
                
                #履歴の活用と生成言語
                "会話履歴を前提知識とし、自己分析の変化や過去の志向性を踏まえてアドバイスしてください。"
                "回答言語は質問（主要部分）の言語に合わせてください。"
                
                #授業コメントへの反応方針
                "キャリアに対する不安に寄り添いつつ、ポジティブに挑戦を促す教育的・励ましのある回答を提供してください。"

        )
    }

    messages = [system_message]

    # 🔽 ここが最重要：履歴を「会話」として追加
    if history_pairs:
        messages.extend(history_pairs)

    # 🔽 今回の質問
    messages.append({
        "role": "user",
        "content": (
            f"【参考資料】\n{content}\n\n"
            f"質問: {query}"
        )
    })

    response = openai.chat.completions.create(
        model=OPENAI_API_MODEL,
        temperature=OPENAI_API_TEMPERATURE,
        max_tokens=OPENAI_API_MAX_TOKENS,
        messages=messages,
    )

    return response.choices[0].message.content



# フォルダ内のすべてのPDFおよびテキストファイルを読み込んでインデックスを作成
def load_and_index_folder(folder_path, return_documents=False):
    all_texts = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # ✅ 追加：空ファイルをスキップ
        if os.path.getsize(file_path) == 0:
            print(f"スキップ（空ファイル）: {filename}")
            continue

        if filename.endswith(".pdf"):
            documents = load_pdf(file_path)
        elif filename.endswith(".txt"):
            documents = load_text(file_path)
        elif filename.endswith(".docx"):
            documents = load_docx(file_path)
        else:
            continue

        texts = split_text(documents)
        all_texts.extend(texts)

    if return_documents:
        return all_texts
    return create_faiss_index(all_texts)

