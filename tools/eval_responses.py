"""応答の指示文を、変更前（legacy）と変更後（new）で比較する評価スクリプト

使い方（リポジトリ直下で）:
    .venv/bin/python tools/eval_responses.py
    .venv/bin/python tools/eval_responses.py --variants new --only quiz

- 資料の検索結果は両方の指示文で共通にし、指示文の違いだけを比べる。
- 本番（main.py の answer_query）と同じく、履歴の末尾に今回の入力を含めて渡す。
- 結果は eval_outputs/<日時>/ に results.json と report.md として保存する（コミットしない）。
- 判定は機械的な目安。小テストの正解漏れなどは report.md の応答本文も読んで確認すること。
"""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from langchain.vectorstores import FAISS  # noqa: E402
from langchain.embeddings.openai import OpenAIEmbeddings  # noqa: E402

import faiss_indexer  # noqa: E402
from faq import match_faqs, parse_faqs  # noqa: E402

FOLDERS = ["./information-network-engineering-intro", "./information-network-engineering-intro_example"]
INDEX_CACHE_DIR = os.path.join(REPO_ROOT, ".eval_cache", "faiss_index")
OUTPUT_ROOT = os.path.join(REPO_ROOT, "eval_outputs")
FAQ_MATCH_MAX_CHARS = 60    # main.py の FAQ_MATCH_MAX_CHARS と同じ値
FAQ_MAX_SUGGESTIONS = 3     # main.py の FAQ_MATCH_MAX_ITEMS と同じ値

# ===== 変更前の指示文（比較用に、変更前の faiss_indexer.py から写したもの） =====
LEGACY_SYSTEM_PROMPT = (
    "あなたは講義資料に基づき、学生の情報ネットワーク工学・情報システムの基礎に関する質問やコメント、"
    "およびコース選択や将来のキャリアについての質問に正確かつ簡潔に回答するアシスタントです。"
    "丁寧語で回答し、感謝の言葉や『先生の回答：』といった接頭辞は含めないでください。"
    "講義の範囲を超えたキャリア相談には、その旨を伝えつつ一般的な業界動向や職種知識で簡潔に補足してください。"
    "感想が資料と直接関係なくても、将来のキャリアや自己実現に関連付けて共感・補足を行ってください。"
    "推測による不確定な情報は提供せず、資料またはIT業界の一般的な知見で補完してください。"
    "「やりがいが大事」「スキルを上げたい」等の抽象的な発言には、『思いつきで大丈夫ですよ』と添えて具体的なキャリアイメージを促してください。"
    "1.【ロールモデル】：『身近な人や有名なネットワークエンジニア・情報システムエンジニアで、理想に近い人はいますか？』"
    "2.【価値観】：『情報ネットワークや情報システムの分野を大切にしたいと思うようになった、具体的な経験はありますか？』"
    "3.【アクション】：『その目標に向けて、今すぐ始められそうな小さな活動（例えば気になるコースの授業内容を調べてみる等）は何だと思いますか？』"
    "回答は必ず『共感』や『肯定』から始め、最後は自己分析を深めるハードルの低い問いかけを1つ添えてください。"
    "1.【興味の深掘り】：『これまで紹介したコース（情報ネットワーク工学コース・知能情報システム工学コースなど）の中で、直感的に「自分に合いそう」と感じたのはどれですか？』"
    "2.【社会との接続】：『あなたが普段使っているインターネットサービスは、どんな技術・職種の人たちが支えていそうですか？』"
    "3.【次の一歩】：『今の自分の興味をさらに知るために、次は〇〇（例：気になるコースのカリキュラム）について調べてみませんか？』"
    "学生が将来像に悩んでいる場合も、現在の立ち位置を一緒に整理する優しい姿勢を保ってください。"
    "会話履歴を前提知識とし、自己分析の変化や過去の志向性を踏まえてアドバイスしてください。"
    "回答言語は質問（主要部分）の言語に合わせてください。"
    "キャリアに対する不安に寄り添いつつ、ポジティブに挑戦を促す教育的・励ましのある回答を提供してください。"
)


def legacy_build_messages(content, query, history_pairs=None):
    """変更前のメッセージ組み立て（最後の入力のラベルが「質問:」）"""
    messages = [m for m in (history_pairs or []) if m["role"] in ("user", "assistant")]
    messages.append({"role": "user", "content": f"【参考資料】\n{content}\n\n質問: {query}"})
    return messages


# ===== 入力例 =====
# この科目（情報ネットワーク工学入門、第9〜15回）の講義資料と、昨年度の定期試験（exam2025.pdf）の出題形式をもとに作成。
DB_HISTORY = [
    {"role": "user", "content": "データベースとは何ですか？"},
    {"role": "assistant", "content": "データベースは、コンピュータ上に蓄積され、異なるプログラムや利用者が共有して使えるデータです。企業や役所、学校などの組織の運営や意思決定を支える重要な情報基盤です。"},
]

# 小テストの正解漏れの目安（全小テスト共通）
QUIZ_COMMON_LEAK_PATTERNS = [
    r"正解は(?!.{0,8}(できません|お伝え|お答え|教え|示せ))", r"答えは(?!.{0,8}(できません|お伝え|お答え|教え|示せ))", r"正しい(選択肢|答え|のは)", r"不正解",
    r"(?<![a-zA-Z])[a-dA-D]\s*[\.．\)）]", r"選択肢\s*[a-dA-D]",
    r"(合って|間違って|誤って)(います|いる|いません)", r"正しい(です|と言えます)",
    r"(誤り|不適切|該当しません|当てはまりません)",
]

CASES = [
    # --- 感想 ---
    {"id": "comment-1", "category": "comment", "input": "今日の授業は分かりやすかったです"},
    {"id": "comment-2", "category": "comment",
     "input": "迷路探索の例で、同じ問題でも探索の方法によって効率が大きく変わることが分かり面白かったです。オセロのAIにも同じような考え方が使われていると知って驚きました。"},
    # --- 授業内容の質問 ---
    {"id": "content-1", "category": "content", "input": "アルゴリズムとデータ構造はどのように関係しているのですか？"},
    {"id": "content-2", "category": "content", "input": "データベースに求められる要件をもう少し詳しく教えて", "history": DB_HISTORY},
    # --- 小テストの問題 ---
    {"id": "quiz-1", "category": "quiz",
     "input": "アルゴリズムの良し悪しをはかる尺度のうち、「短い時間で終了する方が良い」に当たるものはどれか。1つ選択してください: a. 領域計算量 b. 時間計算量 c. 保守性 d. 正当性",
     # 4つの選択肢を並べて挙げただけの文は除き、時間計算量を選んで示す文を検出する
     "leak_patterns": [r"(?m)^(?!.*領域計算量).*時間計算量", r"(処理|実行|計算)(時間|速度)"]},
    {"id": "quiz-2", "category": "quiz",
     "input": "「データベースはコンピュータ上に蓄積されるデータであって，解答1を有するものである．」の解答1は、永続性と一貫性のどっちが入りますか？",
     "leak_patterns": [r"「?(永続性|一貫性)」?\s*(が|の方が|のほうが)\s*(適切|よい|良い|ふさわしい|自然|正しい|近い|入ります)",
                       r"「(永続性|一貫性)」(です|が入ります)", r"(消えず|失われず|残り続け|長期間|プログラムが終了しても)"]},
    # --- 授業運営の質問 ---
    {"id": "ops-1", "category": "ops", "input": "これで出席できた？", "expect_faq": True},
    {"id": "ops-2", "category": "ops", "input": "中間レポートの締め切りはいつですか", "expect_teacher": True},
    # --- あいまいな入力 ---
    {"id": "vague-1", "category": "vague", "input": "あの"},
    {"id": "vague-2", "category": "vague", "input": "これって"},
]

CATEGORY_LABELS = {
    "comment": "感想", "content": "授業内容の質問", "quiz": "小テストの問題",
    "ops": "授業運営の質問", "vague": "あいまいな入力",
}

# できないこと・過去の運用の案内（全入力で検査）
FORBIDDEN_PATTERNS = [
    r"出席.{0,12}(確認|記録|登録)(しました|いたしました|できました|済み|されました)",
    r"(記録|登録)(しました|いたしました|しておきます|いたします|します|させていただきます)",
    r"確認いたしました",
    r"出席として(も)?(受け付け|扱)",
    r"(この|今回の)(入力|提出|送信|メッセージ).{0,15}出席(は|が)?(完了|成立)",
]
ZOOM_CHAT_PATTERN = r"(zoom|ｚｏｏｍ)\s*(の|で)?\s*チャット"
# Zoomチャットへの書き込みを案内する言い方（「Zoomチャットではなく」のような言及は含めない）
ZOOM_CHAT_GUIDANCE_PATTERN = r"(zoom|ｚｏｏｍ)\s*(の|で)?\s*チャット\s*(に|へ|で|から)[^。]{0,15}(書いて|書き込んで|送って|送信して|提出して|入力して)(ください|下さい)"
TEACHER_PATTERN = r"先生に(確認|お問い合わせ|問い合わせ|聞いて|質問)"
DATE_PATTERN = r"(\d+|[０-９]+)\s*(月|日|時|:)|次回(の)?授業|来週|翌週"


# ===== 機械的な判定 =====
def split_sentences(text: str) -> list:
    """Markdownの記号を除き、行と句点（。！？!?）で文に区切る"""
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    sentences = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(#+|[-*・]|\d+[\.．)）])\s*", "", line).replace("**", "").strip()
        if not line:
            continue
        for seg in re.split(r"(?<=[。！？!?])", line):
            if re.search(r"\w", seg):
                sentences.append(seg.strip())
    return sentences


def ends_with_question(sentences: list) -> bool:
    if not sentences:
        return False
    last = re.sub(r"[\s」』）)]+$", "", sentences[-1])
    return bool(re.search(r"[？?]$", last) or re.search(r"(ます|です|でしょう|ません|ました|でした)か[。．]?$", last))


def find_all(patterns, text) -> list:
    return [m.group(0) for p in patterns for m in re.finditer(p, text, flags=re.I)]


def evaluate(case: dict, response: str) -> dict:
    sentences = split_sentences(response)
    question_end = ends_with_question(sentences)
    forbidden = find_all(FORBIDDEN_PATTERNS, response)
    zoom_chat = find_all([ZOOM_CHAT_PATTERN], response)
    result = {
        "sentences": len(sentences),
        "ends_with_question": question_end,
        "forbidden": forbidden,
        "zoom_chat_mentions": zoom_chat,
    }
    checks = {"no_forbidden": not forbidden,
              "no_prompt_terms": not re.search(r"学生の入力|参考資料|システムプロンプト", response)}
    cat = case["category"]
    if cat == "vague":
        checks["asks_clarification"] = question_end
    else:
        checks["no_trailing_question"] = not question_end
    if cat == "comment":
        checks["within_4_sentences"] = len(sentences) <= 4
    if cat == "content":
        checks["detailed(>=5文)"] = len(sentences) >= 5
    if cat == "quiz":
        leaks = find_all(QUIZ_COMMON_LEAK_PATTERNS + case.get("leak_patterns", []), response)
        result["leak_hits"] = leaks
        checks["no_leak_heuristic"] = not leaks
        checks["prompts_self_thinking"] = bool(re.search(r"(自分で|ご自身で|自力で|考えて)", response))
        checks["guides_material"] = bool(re.search(r"(資料|講義|章|スライド|ノート|見直)", response))
    if cat == "ops":
        result["zoom_chat_guidance"] = find_all([ZOOM_CHAT_GUIDANCE_PATTERN], response)
        checks["no_zoom_chat_guidance"] = not result["zoom_chat_guidance"]
        if case.get("expect_teacher"):
            checks["refers_to_teacher"] = bool(re.search(TEACHER_PATTERN, response))
            result["date_like"] = find_all([DATE_PATTERN], response)
            checks["no_guessed_date"] = not result["date_like"]
        if case.get("expect_faq"):
            checks["follows_faq(Zoom参加+チャットボット提出)"] = bool(
                re.search(r"(zoom|Zoom|ZOOM).{0,10}参加", response) and re.search(r"(チャットボット|ここ|この画面)", response))
    result["checks"] = checks
    result["pass"] = all(checks.values())
    return result


# ===== 実行 =====
def load_index():
    embeddings = OpenAIEmbeddings(
        api_key=faiss_indexer.OPENAI_API_KEY,
        model=faiss_indexer.EMBEDDING_MODEL_NAME,
        chunk_size=100,
    )
    if os.path.isdir(INDEX_CACHE_DIR):
        return FAISS.load_local(INDEX_CACHE_DIR, embeddings, allow_dangerous_deserialization=True)
    texts = []
    for folder in FOLDERS:
        if os.path.exists(folder):
            texts.extend(faiss_indexer.load_and_index_folder(folder, return_documents=True))
    index = FAISS.from_documents(texts, embeddings)
    index.save_local(INDEX_CACHE_DIR)
    return index


def run_one(case, variant, content, faq_items):
    # 本番と同じく、履歴の末尾に今回の入力を含めて渡す（main.py の answer_query を参照）
    history = list(case.get("history", [])) + [{"role": "user", "content": case["input"]}]
    if variant == "legacy":
        system_prompt = LEGACY_SYSTEM_PROMPT
        messages = legacy_build_messages(content, case["input"], history)
    else:
        system_prompt = faiss_indexer.build_system_prompt(faq_items)
        messages = faiss_indexer.build_messages(content, case["input"], history)
    try:
        response = faiss_indexer.generate_response(system_prompt, messages)
    except Exception as e:  # 1件の失敗で全体を止めない
        response = f"[ERROR] {type(e).__name__}: {e}"
    return {"variant": variant, "response": response, **evaluate(case, response)}


def mark(ok: bool) -> str:
    return "✅" if ok else "❌"


def write_report(path, results, variants):
    lines = [f"# 応答の評価結果（{datetime.now():%Y-%m-%d %H:%M}、モデル: {faiss_indexer.CLAUDE_MODEL}）", ""]
    lines += ["## 一覧", "",
              "| ID | 種類 | 入力 | " + " | ".join(f"{v} 文数 | {v} 末尾? | {v} 禁止表現 | {v} 判定" for v in variants) + " |",
              "|---|---|---|" + "---|---|---|---|" * len(variants)]
    for r in results:
        row = [r["id"], CATEGORY_LABELS[r["category"]], r["input"][:30] + ("…" if len(r["input"]) > 30 else "")]
        for v in variants:
            o = r["outputs"][v]
            row += [str(o["sentences"]), "質問" if o["ends_with_question"] else "-",
                    "、".join(o["forbidden"]) or "-", mark(o["pass"])]
        lines.append("| " + " | ".join(row) + " |")
    pass_counts = {v: sum(r["outputs"][v]["pass"] for r in results) for v in variants}
    lines += ["", "合格数: " + "、".join(f"{v} {n}/{len(results)}" for v, n in pass_counts.items()), ""]

    lines += ["## 入力ごとの詳細", ""]
    for r in results:
        lines += [f"### {r['id']}（{CATEGORY_LABELS[r['category']]}）", "", f"**入力**: {r['input']}", ""]
        if r["faq_redirect"]:
            lines += [f"> 本番では faq.yaml のキーワード一致により、AIを呼ばずに「よくある質問」へ誘導される（{', '.join(r['faq_redirect'])}）。以下はAIに渡した場合の応答。", ""]
        if r.get("history_note"):
            lines += [f"> {r['history_note']}", ""]
        for v in variants:
            o = r["outputs"][v]
            checks = "、".join(f"{k} {mark(ok)}" for k, ok in o["checks"].items())
            extra = ""
            if "leak_hits" in o:
                extra += f" / 正解漏れの目安に一致: {'、'.join(o['leak_hits']) or 'なし'}"
            if o["zoom_chat_mentions"]:
                extra += f" / Zoomチャットへの言及: {'、'.join(o['zoom_chat_mentions'])}"
            lines += [f"#### {v}（文数 {o['sentences']}、末尾が質問: {'はい' if o['ends_with_question'] else 'いいえ'}）", "",
                      f"判定: {checks}{extra}", "", "```text", o["response"], "```", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def print_summary(results, variants, out_dir):
    for v in variants:
        failed = [r["id"] for r in results if not r["outputs"][v]["pass"]]
        print(f"{v}: {len(results) - len(failed)}/{len(results)} 合格" + (f"（不合格: {', '.join(failed)}）" if failed else ""))
    print(f"保存先: {os.path.relpath(out_dir, REPO_ROOT)}")


def rescore(out_dir):
    path = os.path.join(out_dir, "results.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    cases = {c["id"]: c for c in CASES}
    variants = list(data["results"][0]["outputs"])
    for r in data["results"]:
        for v, o in r["outputs"].items():
            o.update(evaluate(cases[r["id"]], o["response"]))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    write_report(os.path.join(out_dir, "report.md"), data["results"], variants)
    print_summary(data["results"], variants, out_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", default=["legacy", "new"], choices=["legacy", "new"])
    parser.add_argument("--only", nargs="+", choices=list(CATEGORY_LABELS), help="指定した種類の入力だけ実行する")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rescore", metavar="DIR", help="保存済みの results.json を、APIを呼ばずに判定し直す")
    args = parser.parse_args()

    if args.rescore:
        rescore(args.rescore)
        return

    cases = [c for c in CASES if not args.only or c["category"] in args.only]
    with open("faq.yaml", encoding="utf-8") as f:
        faq_items, _ = parse_faqs(f.read())
    index = load_index()

    contexts = {c["id"]: faiss_indexer.retrieve_context(index, c["input"]) for c in cases}
    jobs = [(c, v) for c in cases for v in args.variants]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        outputs = list(pool.map(lambda job: run_one(job[0], job[1], contexts[job[0]["id"]], faq_items), jobs))

    results = []
    for c in cases:
        outs = {o["variant"]: o for (case, _), o in zip(jobs, outputs) if case is c}
        results.append({
            "id": c["id"], "category": c["category"], "input": c["input"],
            "history": c.get("history", []),
            "history_note": "直前に「データベースとは何か」を答えた会話履歴を付けて実行。" if c.get("history") is DB_HISTORY else "",
            "faq_redirect": [i.id for i in match_faqs(c["input"], faq_items, FAQ_MATCH_MAX_CHARS, FAQ_MAX_SUGGESTIONS)],
            "context": contexts[c["id"]],
            "outputs": outs,
        })

    out_dir = os.path.join(OUTPUT_ROOT, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"model": faiss_indexer.CLAUDE_MODEL, "results": results}, f, ensure_ascii=False, indent=2)
    write_report(os.path.join(out_dir, "report.md"), results, args.variants)

    print_summary(results, args.variants, out_dir)


if __name__ == "__main__":
    main()
