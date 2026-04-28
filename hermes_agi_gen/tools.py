"""汎用ツール定義。Planner がプロンプトに注入するツール仕様。"""

TOOL_DESCRIPTIONS = """\
【使えるツール】
- PLAN: step1 || step2         複雑なゴールを複数ステップに分解して一括計画する (|| で区切る)
- ANSWER: <text>               質問・相談に直接テキストで回答する (会話・説明・調査結果など)
- SEARCH: <query>              DuckDuckGo でウェブ検索する (最新情報・外部知識の取得)
- FETCH: <url>                 指定URLのコンテンツを取得する (API・HTMLページ・JSONなど直接アクセス)
- CALC: <expression>           数式・統計式を即座に計算する (sqrt/log/sin/pi など math 関数も使用可)
- CMD: <bash command>          シェルコマンドを実行する (ls, find, grep など。インタプリタ起動は不可)
- READ: <filepath>             ファイルの内容を読む
- WRITE: <filepath>            ファイルに書き込む (次の行から内容を記述、末尾は EOF)
- PYTHON: <code>               Python コードを実行する (1行 or 複数行)
- SCHEDULE_AT: <trigger> <goal>  時刻指定でゴールをスケジュールする (デーモンが自動実行)
- DONE: <summary>              タスク完了を宣言し結果を要約する\
"""

TOOL_EXAMPLES = """\
【ツール使用例】
PLAN: SEARCH: 最新の量子コンピュータ動向 || SEARCH: 量子コンピュータ 商用化 2024 || ANSWER: 調査結果のまとめ
PLAN: CMD: find . -name "*.py" | head -20 || READ: main.py || ANSWER: コードの概要
PLAN: FETCH: https://hacker-news.firebaseio.com/v0/topstories.json || ANSWER: 取得したJSONから必要な記事IDを要約する
ANSWER: はい、一般的な質問に答えることができます。コーディング・調査・文章作成など幅広いトピックに対応しています。
SEARCH: Python 3.12 新機能
SEARCH: 東京の今日の天気
FETCH: https://hacker-news.firebaseio.com/v0/topstories.json
FETCH: https://hacker-news.firebaseio.com/v0/item/12345678.json
FETCH: https://news.ycombinator.com/
CALC: sqrt(2) * pi
CALC: sum([1,2,3,4,5]) / 5
CALC: 2 ** 32
CMD: ls -la
CMD: find . -name "*.py" | head -20
CMD: grep -r "class Agent" . --include="*.py" | head -10
READ: README.md
READ: src/config.py
PYTHON: import sys; print(sys.version)
PYTHON:
data = [1, 2, 3, 4, 5]
print(f"合計: {sum(data)}, 平均: {sum(data)/len(data)}")
WRITE: output/result.txt
ここに書き込む内容
複数行も OK
EOF
SCHEDULE_AT: once:2026-04-07T02:30 Hacker Newsのトップニュースを取得して日本語で要約し表示する
SCHEDULE_AT: daily:09:00 毎朝のニュースをチェックして要約する
SCHEDULE_AT: every:30m システム状態を確認する
DONE: README を確認し、プロジェクトが Python 3.11 で動作することを検証しました。

# Hacker News から記事を取得・要約・保存する場合の推奨パターン:
PLAN: FETCH: https://hacker-news.firebaseio.com/v0/topstories.json || FETCH: https://hacker-news.firebaseio.com/v0/item/<id>.json || WRITE: output/hn_summary.txt
ここに要約を書く
EOF\
"""

TOOL_CONSTRAINTS = """\
【制約】
- SEARCH: はキーワード検索専用 (DuckDuckGo)。特定URLのコンテンツ取得には FETCH: を使う
- FETCH: で任意のURLに直接アクセスできる (REST API, HackerNews API, JSONエンドポイントなど)
- SCHEDULE_AT: のゴールは「やること」をプレーンテキストで書く。ツールコマンド (FETCH:/PLAN: など) を埋め込まない
  例: SCHEDULE_AT: once:2026-04-07T02:30 HackerNewsのトップAIニュースを日本語100字に要約して~/Desktop/AI_News/に保存する
- SCHEDULE_AT: 登録に成功したら、直後に DONE: で終了する (別途実行は不要)
- SCHEDULE_AT: トリガー形式: once:<ISO8601> | every:<N>m | every:<N>h | daily:<HH:MM> | weekly:<day>:<HH:MM>
- WRITE: は原則リポジトリ配下のみOK。外部出力先は HERMES_WRITE_ALLOW_DIRS で明示許可されたディレクトリのみ書ける
- PYTHON: はファイル書き込み不可。ファイル変更は WRITE: を使う
- ブラウザ・GUI アプリケーションは使えない
- 失敗済みの同一コマンドを繰り返さない
- tree コマンドは存在しない → find を使う
- locate コマンドは使えない → find を使う\
"""
