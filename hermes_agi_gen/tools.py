"""汎用ツール定義。Planner がプロンプトに注入するツール仕様。"""

TOOL_DESCRIPTIONS = """\
【使えるツール】
- PLAN: step1 || step2  複雑なゴールを複数ステップに分解して一括計画する (|| で区切る)
- ANSWER: <text>        質問・相談に直接テキストで回答する (会話・説明・調査結果など)
- SEARCH: <query>       DuckDuckGo でウェブ検索する (最新情報・外部知識の取得)
- CALC: <expression>    数式・統計式を即座に計算する (sqrt/log/sin/pi など math 関数も使用可)
- CMD: <bash command>   シェルコマンドを実行する (ls, find, grep, python など)
- READ: <filepath>      ファイルの内容を読む
- WRITE: <filepath>     ファイルに書き込む (次の行から内容を記述、末尾は EOF)
- PYTHON: <code>        Python コードを実行する (1行 or 複数行)
- DONE: <summary>       タスク完了を宣言し結果を要約する\
"""

TOOL_EXAMPLES = """\
【ツール使用例】
PLAN: SEARCH: 最新の量子コンピュータ動向 || SEARCH: 量子コンピュータ 商用化 2024 || ANSWER: 調査結果のまとめ
PLAN: CMD: find . -name "*.py" | head -20 || READ: main.py || ANSWER: コードの概要
ANSWER: はい、一般的な質問に答えることができます。コーディング・調査・文章作成など幅広いトピックに対応しています。
SEARCH: Python 3.12 新機能
SEARCH: 東京の今日の天気
SEARCH: LLM agent architecture 2024
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
DONE: README を確認し、プロジェクトが Python 3.11 で動作することを検証しました。\
"""

TOOL_CONSTRAINTS = """\
【制約】
- SEARCH: でウェブ検索が可能。ただし URL の直接アクセスやブラウザ操作は不可
- ブラウザ・GUI アプリケーションは使えない
- 失敗済みの同一コマンドを繰り返さない
- WRITE はリポジトリ内のファイルのみ (システムファイル・上位ディレクトリは禁止)
- tree コマンドは存在しない → find を使う
- locate コマンドは使えない → find を使う\
"""
