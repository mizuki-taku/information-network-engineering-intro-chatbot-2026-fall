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

### 3. チャット履歴が増えると入力欄がどんどん下に流れてしまう（Streamlit）

- **原因**: `st.form` + `st.text_area` + `st.form_submit_button` で入力欄を作ると、ページの一部として通常のレイアウトに従って配置されるため、会話履歴が増えるほど入力欄の位置が下がっていき、毎回スクロールが必要になる。
- **対処法**: `st.chat_input()` を使う。Streamlit標準のチャット入力欄で、常に画面下部に固定表示される。
  - 注意点: `st.chat_input()` は単一行入力（Enterキーで即送信）。複数行入力やCtrl+Enter送信が必須の場合は、Streamlitのバージョンアップ（複数行対応は概ね1.41以降）を検討するか、CSSで既存フォームを固定表示する方法を検討する。