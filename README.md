# software-engineering-chatbot

ローカル環境で動かす手順

## セットアップ

python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


##　動作
streamlit run main.py

## 既知の問題と対処法（Claude API連携チャットボット共通）

Anthropic Claude APIを使ったチャットボットで繰り返し発生しやすい問題と対処法をまとめる。
他コース向けにチャットボットを新規構築・移植する際は、まずここを確認する。

### 1. `BadRequestError: 'temperature' is deprecated for this model`

- **原因**: 新しいClaudeモデル（例: `claude-sonnet-5`系）では `messages.create()` に `temperature` パラメータを渡すとエラーになる場合がある。
- **対処法**: `temperature` 引数を渡さず、モデルのデフォルト挙動に任せる。
- **該当箇所**: `faiss_indexer.py` の `client.messages.create()` 呼び出し。

### 2. `AttributeError: 'ThinkingBlock' object has no attribute 'text'`

- **原因**: 拡張思考（extended thinking）に対応したモデルでは、レスポンスの `response.content` に `ThinkingBlock`（思考過程）が含まれることがあり、必ずしも `content[0]` がテキスト本体とは限らない。`response.content[0].text` と決め打ちで取得すると発生する。
- **対処法**: `response.content` をループし、`block.type == "text"` のブロックを探して使う。

```python
for block in response.content:
    if block.type == "text":
        return block.text
```

### 3. チャット履歴が増えると入力欄や回答が画面の下に流れてしまう（Streamlit）

- **原因**: 会話履歴を1画面に並べると、履歴が増えるほど入力欄や最新の回答が下がり、毎回スクロールが必要になる。`st.chat_input()` で入力欄を画面下に固定しても、今度は回答が見えにくくなる。
- **対処法**: 画面を「Q&A」（上に入力欄、直下に最新の回答）と「過去ログ」（高さ固定の枠内でスクロール）に分けた（`ui/` フォルダ）。
  - `st.form` の中の入力は送信するまでサーバーに送られないため、画面を切り替えると入力途中の内容が消える。入力欄はフォームにせず、内容を `st.session_state` に退避している。
  - 高さ固定の `st.container` に `st.chat_message` を入れると、枠が末尾へ自動スクロールする。新しい順に並べる過去ログでは使わない。

### 4. 回答の作成中にボタンを押すと、回答やSheetsへの記録が抜ける・AIが二重に呼ばれる（Streamlit）

- **原因**: Streamlit（`runner.fastReruns` が既定で有効）は、実行中にボタンが押されると、古い実行がAIの応答を待っている間に新しい実行を並行して始める。古い実行は、次に画面要素を出すときや `st.session_state` を読み書きするときに打ち切られる。
- **対処法**: `ui/qa.py` の `process_pending()` で、最初の実行だけが回答を作成し、後から始まった実行は完了を待つ。AIの応答を受け取ってからSheetsに記録するまでは、画面要素を出さず `st.session_state` にも触れない（`main.py` の `answer_query()`）。

## よくある質問（FAQ）の追加方法

画面右上の「よくある質問」に表示する内容は、リポジトリ直下の `faq.yaml` に書く。コードの変更は不要で、項目を追記して GitHub にプッシュすれば、Streamlit Community Cloud に反映される（ページを再読み込みすると表示される）。

```yaml
faqs:
  - id: attendance                     # 必須。項目ごとに別の英数字（重複不可）
    question: 出席はどうすれば完了しますか？   # 必須。一覧に表示する質問
    answer: |                          # 必須。複数行のMarkdownを書ける
      Zoomへの参加と、チャットボットへの質問・感想の提出で完了します。
    keywords: [出席, 出欠, 出席できた]     # 任意。自動判定に使う語句
    enabled: true                      # 任意。false で非表示かつ判定対象外（既定 true）
    order: 1                           # 任意。表示順（小さい順）。指定のない項目は、指定ありの後ろに記載順で並ぶ
```

- **keywords の考え方**: 学生の入力に、いずれかの語句が含まれていれば、AIに回答させず「よくある質問」へ誘導する（このときもSheetsには通常どおり記録される）。
  - 全角／半角、大文字／小文字、空白・記号の違いは無視して照合する。
  - 誤判定を避けるため、判定するのは60文字以下の短い入力だけ（`main.py` 冒頭の `FAQ_MATCH_MAX_CHARS`）。長い感想に語句が入っていても、AIが回答する。
  - 「出席」のような短い語は広く当たる。講義内容の質問にも出てくる語（例:「仕様」「プロトコル」）は入れない。迷ったら「どう書け」「出席できた」のように、言い回しごと登録する。
  - keywords のない項目は、一覧に表示されるだけで自動判定には使われない。
- 書き方を間違えた項目は、その項目だけ読み飛ばし、「よくある質問」の一覧に警告を表示する。
- 誘導メッセージの文言、判定の上限文字数、科目名などは `main.py` 冒頭の「UI設定」で変更できる。他科目に展開するときは、`ui/` フォルダと `faq.py` をそのままコピーし、`main.py` の「UI設定」と `faq.yaml` を科目に合わせる。
