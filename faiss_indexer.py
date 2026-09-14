import os
import anthropic
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from file_loader import load_pdf, load_text, load_docx
from text_splitter import split_text

# .envファイルの内容を読み込み
load_dotenv()

# 環境変数からAPIキーと設定を取得
ANTHROPIC_API_KEY       = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY          = os.getenv("OPENAI_API_KEY")  # Embedding用（OpenAIのまま）
CLAUDE_API_TEMPERATURE  = float(os.getenv("CLAUDE_API_TEMPERATURE", "0.15"))  # 0.0〜1.0
CLAUDE_API_MAX_TOKENS   = int(os.getenv("CLAUDE_API_MAX_TOKENS",    "1000"))
CLAUDE_API_TOP_K        = int(os.getenv("CLAUDE_API_TOP_K",         "3"))     # FAISS検索件数
EMBEDDING_MODEL_NAME    = os.getenv("OPENAI_EMBEDDING_MODEL",       "text-embedding-3-small")
CLAUDE_MODEL            = os.getenv("CLAUDE_MODEL",                 "claude-sonnet-4-6")

# Anthropicクライアントの初期化
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# 埋め込みとインデックス作成
def create_faiss_index(texts):
    """
    Document群をベクトル化してFAISSインデックスを構築
    Embeddingモデルは環境変数 OPENAI_EMBEDDING_MODEL で指定
    ※ Embeddingは引き続きOpenAIを使用
    """
    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        model=EMBEDDING_MODEL_NAME
    )
    return FAISS.from_documents(texts, embeddings)


def search_docs(faiss_index, query):
    """FAISSで検索して上位チャンクを返す"""
    return faiss_index.similarity_search(query, k=CLAUDE_API_TOP_K)


def search_index(faiss_index, query, history_pairs=None):
    """
    history_pairs: [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
    ]
    """

    results = faiss_index.similarity_search(query, k=CLAUDE_API_TOP_K)
    content = "\n\n---\n\n".join([r.page_content for r in results]) if results else "該当する情報が資料内に見つかりませんでした。"

    # システムプロンプト（役割定義）
    system_prompt = (
        # 学生の質問やコメントに対する基本的な役割とアプローチ
        "あなたは講義資料に基づき、学生の情報ネットワーク工学・情報システムの基礎に関する質問やコメント、"
        "およびコース選択や将来のキャリアについての質問に正確かつ簡潔に回答するアシスタントです。"
        "丁寧語で回答し、感謝の言葉や『先生の回答：』といった接頭辞は含めないでください。"

        # 範囲外・不確定な情報への対応
        "講義の範囲を超えたキャリア相談には、その旨を伝えつつ一般的な業界動向や職種知識で簡潔に補足してください。"
        "感想が資料と直接関係なくても、将来のキャリアや自己実現に関連付けて共感・補足を行ってください。"
        "推測による不確定な情報は提供せず、資料またはIT業界の一般的な知見で補完してください。"

        # 抽象的な発言に対する具体化支援
        "「やりがいが大事」「スキルを上げたい」等の抽象的な発言には、『思いつきで大丈夫ですよ』と添えて具体的なキャリアイメージを促してください。"
        "1.【ロールモデル】：『身近な人や有名なネットワークエンジニア・情報システムエンジニアで、理想に近い人はいますか？』"
        "2.【価値観】：『情報ネットワークや情報システムの分野を大切にしたいと思うようになった、具体的な経験はありますか？』"
        "3.【アクション】：『その目標に向けて、今すぐ始められそうな小さな活動（例えば気になるコースの授業内容を調べてみる等）は何だと思いますか？』"

        # 伴走型支援と問いかけ
        "回答は必ず『共感』や『肯定』から始め、最後は自己分析を深めるハードルの低い問いかけを1つ添えてください。"
        "1.【興味の深掘り】：『これまで紹介したコース（情報ネットワーク工学コース・知能情報システム工学コースなど）の中で、直感的に「自分に合いそう」と感じたのはどれですか？』"
        "2.【社会との接続】：『あなたが普段使っているインターネットサービスは、どんな技術・職種の人たちが支えていそうですか？』"
        "3.【次の一歩】：『今の自分の興味をさらに知るために、次は〇〇（例：気になるコースのカリキュラム）について調べてみませんか？』"
        "学生が将来像に悩んでいる場合も、現在の立ち位置を一緒に整理する優しい姿勢を保ってください。"

        # 履歴の活用と生成言語
        "会話履歴を前提知識とし、自己分析の変化や過去の志向性を踏まえてアドバイスしてください。"
        "回答言語は質問（主要部分）の言語に合わせてください。"

        # 授業コメントへの反応方針
        "キャリアに対する不安に寄り添いつつ、ポジティブに挑戦を促す教育的・励ましのある回答を提供してください。"
    )

    # メッセージ履歴の構築（Anthropic形式）
    # ※ Anthropic の messages には "system" ロールを含めない
    messages = []

    if history_pairs:
        # OpenAI形式の履歴からsystemロールを除外してそのまま使用可能
        for msg in history_pairs:
            if msg["role"] in ("user", "assistant"):
                messages.append(msg)

    # 最新の質問を追加（参考資料を埋め込む）
    messages.append({
        "role": "user",
        "content": (
            f"【参考資料】\n{content}\n\n"
            f"質問: {query}"
        )
    })

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=CLAUDE_API_MAX_TOKENS,
        temperature=CLAUDE_API_TEMPERATURE,
        system=system_prompt,   # ← Anthropicはsystemを専用引数で渡す
        messages=messages,
    )

    return response.content[0].text


def load_and_index_folder(folder_path, return_documents=False):
    all_texts = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 空ファイルをスキップ
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
