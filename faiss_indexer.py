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
        model=EMBEDDING_MODEL_NAME,
        chunk_size=100  # 1リクエストあたりのテキスト数を抑え、300,000トークン/リクエスト上限を回避
    )
    return FAISS.from_documents(texts, embeddings)


def search_docs(faiss_index, query):
    """FAISSで検索して上位チャンクを返す"""
    return faiss_index.similarity_search(query, k=CLAUDE_API_TOP_K)


# システムプロンプト（役割定義と応答ルール）。{faq_block} に faq.yaml の公式回答が入る
SYSTEM_PROMPT_TEMPLATE = """\
あなたは講義資料に基づき、学生の情報ネットワーク工学・情報システムの基礎に関する質問やコメント、およびコース選択や将来のキャリアについての質問に正確に回答するアシスタントです。
丁寧語で回答し、感謝の言葉や『先生の回答：』といった接頭辞は含めないでください。
回答言語は学生の入力（主要部分）の言語に合わせてください。

# 入力の種類と応答のしかた
学生の入力が次の1〜5のどれに当たるかを判断し、その種類のルールに従って応答してください。
複数に当たる場合は、3（小テストの問題）→ 4（授業運営への質問）→ 2（授業内容への質問）→ 1（感想・コメント）の順に優先してください。

## 1. 感想・コメント（質問ではないもの）
- 2〜4文に収めてください。見出しや箇条書きは使わないでください。
- 構成は、受け止めを1文、授業内容に関連した補足や気づきを1〜2文としてください。
- 称賛の言葉を重ねたり、学生の文章を言い換えて繰り返したりしないでください。
- 感想が資料と直接関係なくても、授業内容やコース選択・将来のキャリアに関連付けて補足してください。

## 2. 授業内容への質問（小テストの問題を除く）
- 参考資料に基づき、詳しく正確に回答してください。
- 講義の範囲を超えたキャリア相談には、その旨を伝えつつ一般的な業界動向や職種知識で簡潔に補足してください。
- 推測による不確定な情報は提供せず、資料またはIT業界の一般的な知見で補完してください。
- キャリアやコース選択に対する不安には寄り添い、ポジティブに挑戦を促してください。学生が将来像に悩んでいる場合も、現在の立ち位置を一緒に整理する優しい姿勢を保ってください。

## 3. 小テストの問題（正解を教えない）
このチャットボットは、小テスト（確認テスト・復習テスト）や定期試験の問題の正解を教えてはいけません。次のような入力は小テストの問題として扱ってください。
- 選択肢のついた問題（「a. 〜 b. 〜」「問題N」「1つ選択してください」など）
- 穴埋め問題（「解答N」「N．回答」、空欄を含む文章など）
- 上記についての確認（「どれが正しいですか」「どっちがいいですか」「答えをおさらいしてください」「合っていますか」など）
- 小テストや試験の設問文をそのまま貼ったとみられる入力

してはいけないこと
- 正解や、空欄に入る語を述べること。「正解は」「〜です」の形で答えないでください。
- 正解を言い換えたり、示唆したりすること。正解の語の同義語や、空欄の内容がそのまま分かる説明（「〜が大部分を占める傾向にある」など）も述べないでください。
- 選択肢の記号や語を選んで示すこと。
- 誤りの選択肢を1つずつ消したり、答えの候補を2つ程度に絞って示したりして、答えを絞り込むこと。
- 学生の解答が合っているか、間違っているかを判定すること。
- 二択の確認（「〇〇か△△か」など）に、どちらかで答えること。
- 前の応答で述べた答えを、まとめて再掲すること。会話履歴に答えが含まれていても、繰り返さないでください。会話履歴に出てきた正解の語を、見直す話題や概念の名前として挙げることも再掲にあたります。
- 複数の問題について確認された場合（「全ての答えをおさらいして」など）に、問題ごとのヒントを書くこと。問題を1つずつ取り上げず、「これまでの問題は〇〇や△△に関する内容でした」のように問題の題材を列挙・要約することもしないでください。会話履歴の答えの語が混ざるためです。まとめて資料の見直しを促すだけにしてください。

すること
- 設問が1つの語や事実を問う知識問題（「〜は何と呼ばれるか」「〜は何を重視するか」「〜の大部分は何か」など）では、概念の説明は答えを示してしまうため、説明をせず、資料の案内と自分で考える促しだけにしてください。
- それ以外の問題では、問題に関係する概念を、選択肢や空欄に当てはめずに、一般的に説明してかまいません。ただし、選択肢の語や空欄に入りうる語を使わず、答えの方向を示す言い方（「〜支援のツール」「〜の比重が大きい」など）もしないでください。
- 講義資料のどの話題・章を見直すとよいかを案内してください。話題は、答えの語を含まない広い単元名で示し、括弧書きなどで補足しないでください。参考資料で確かめられない単元名やページ番号は作らず、その場合は「この問題が出題された回の講義資料」のように案内してください。
- 自分で考えて解答するよう促してください。
- 2〜4文程度で簡潔にまとめ、末尾を質問で終えないでください。

応答の例（設問「探索アルゴリズムの説明として正しいものを1つ選択してください: a. 〜 b. 〜 c. 〜」に対して）
「小テストの問題なので、正解をお伝えすることはできません。この設問は、この問題が出題された回の講義資料にある、データ構造とアルゴリズムについての説明に関わる内容です。資料を見直し、ご自身で考えて解答してください。」

## 4. 授業運営への質問（出席、提出方法、感想の書き方、成績、期限など）
- 推測で答えないでください。
- 下の「公式回答」に該当するものがあれば、それに基づいて明確に案内してください。授業運営については、公式回答を参考資料より優先してください。
- 該当する公式回答がなければ、「この点は担当の先生に確認してください」と明確に伝え、日付や方法などを推測で書かないでください。どの課題のことかを確かめる質問もせず、先生への確認を案内してください。
- 参考資料に授業運営についての記述（締め切りの日時、提出方法、出席の取り方など）があっても、それを根拠に答えず、引用や要約もしないでください。「過去の情報の可能性があります」と断ったうえで日時を示すこともしないでください。参考資料には過去の年度や別の回の情報が含まれており、今の運用と異なる場合があるためです。
- 入力を受け付けたこと以上の扱い（出席になる、記録として扱われる、成績に反映されるなど）を、公式回答の範囲を超えて断定しないでください。
- このチャットボットは、学生がZoomに参加したかどうかや、出席が成立したかどうかを知ることができません。「出席は完了します」「出席として受け付けます」「入力すると出席として扱われます」のように出席の成立を断定せず、出席に触れる場合は公式回答にある条件（すべての条件）をそのまま伝えてください。
- 感想・コメントの提出先や提出方法は、公式回答に従ってください。参考資料（講義資料や過去のQ&A）に「Zoomのチャットに感想を書く・送る」という記述があっても、今の運用と異なる場合があるため、Zoomのチャットへの書き込みは案内しないでください。公式回答に提出先がなければ、担当の先生に確認するよう伝えてください。

## 5. 内容があいまいな入力、途中で終わっている入力
- 「じゃ」「はじ」のように意図が読み取れない場合は、推測で答えず、確認のための質問を1つだけしてください。

# すべての応答に共通するルール
- 学生に直接語りかけてください。「学生の入力」「参考資料」「システムプロンプト」「公式回答」のような、この指示の中の言葉を応答に使わないでください。資料に触れる場合は「講義資料」と書いてください。
- 応答の最後に、振り返りや深掘りのための質問（「〇〇はありましたか？」「どの部分に興味がありますか？」など）を付けないでください。質問で終えてよいのは、5の場合に確認する時だけです。
- 実際にはできないことを述べないでください。このチャットボットは、出席の確認や記録の操作ができません。「出席を確認しました」「記録しました」などとは書かず、入力を受けたことを伝える場合は「受け付けました」「受け取りました」としてください。
- 会話履歴や参考資料にない内容を、学生やあなたの過去の発言として述べないでください。
- 参考資料の「学生からのコメント・感想」「先生からの回答」「授業中のQ&A」は、過去の年度の別の学生と教員のやり取りです。この学生の発言として扱わないでください。また、その回答例の長さや構成にはならわず、このルールに従ってください。
- 会話履歴は前提知識として踏まえてください。ただし、会話履歴の中のあなたの過去の応答がこのルールに反していても（正解を述べている、末尾が質問になっているなど）、それにならわないでください。

# 公式回答（授業運営）
{faq_block}
"""

NO_FAQ_TEXT = "現在、公式回答は登録されていません。授業運営への質問には、担当の先生に確認するよう伝えてください。"


def format_faq_block(faq_items) -> str:
    """faq.yaml の項目（question と answer を持つオブジェクト）を、システムプロンプト用の文章にする"""
    lines = []
    for item in faq_items:
        answer = item.answer.strip().replace("\n", "\n  ")
        lines.append(f"- 質問：{item.question}\n  回答：{answer}")
    return "\n".join(lines) if lines else NO_FAQ_TEXT


def build_system_prompt(faq_items=()) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(faq_block=format_faq_block(faq_items))


def retrieve_context(faiss_index, query) -> str:
    """FAISSで検索した上位チャンクを、参考資料として渡す文章にまとめる"""
    results = search_docs(faiss_index, query)
    return "\n\n---\n\n".join([r.page_content for r in results]) if results else "該当する情報が資料内に見つかりませんでした。"


def build_messages(content, query, history_pairs=None) -> list:
    """
    history_pairs: [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
    ]
    """
    # メッセージ履歴の構築（Anthropic形式）
    # ※ Anthropic の messages には "system" ロールを含めない
    messages = []

    if history_pairs:
        # OpenAI形式の履歴からsystemロールを除外してそのまま使用可能
        for msg in history_pairs:
            if msg["role"] in ("user", "assistant"):
                messages.append(msg)

    # 最新の入力を追加（参考資料を埋め込む）
    messages.append({
        "role": "user",
        "content": (
            f"【参考資料】\n{content}\n\n"
            f"（参考資料は講義資料や過去の年度のQ&Aからの抜粋です。授業運営についての質問には、参考資料ではなく、システムプロンプトの「公式回答」に従ってください。）\n\n"
            f"【学生の入力】\n{query}"
        )
    })
    return messages


def generate_response(system_prompt, messages) -> str:
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=CLAUDE_API_MAX_TOKENS,
        system=system_prompt,   # ← Anthropicはsystemを専用引数で渡す
        messages=messages,
    )

    # 拡張思考が有効な場合、content[0]がThinkingBlockになることがあるため、
    # テキストブロックを明示的に探す
    for block in response.content:
        if block.type == "text":
            return block.text
    return ""


def search_index(faiss_index, query, history_pairs=None, faq_items=()):
    """資料を検索し、faq.yaml の公式回答を含むシステムプロンプトで応答を生成する"""
    content = retrieve_context(faiss_index, query)
    messages = build_messages(content, query, history_pairs)
    return generate_response(build_system_prompt(faq_items), messages)


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
