"""Hermes AGI Gen 共有定数。"""

# Mistral / Ollama
OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
MISTRAL_API_BASE_URL = "https://api.mistral.ai/v1"
DEFAULT_MISTRAL_MODEL = "mistral"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_URL = f"{OPENROUTER_BASE_URL}/models"
OPENROUTER_CHAT_URL = f"{OPENROUTER_BASE_URL}/chat/completions"

AI_GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v1"
AI_GATEWAY_MODELS_URL = f"{AI_GATEWAY_BASE_URL}/models"
AI_GATEWAY_CHAT_URL = f"{AI_GATEWAY_BASE_URL}/chat/completions"

NOUS_API_BASE_URL = "https://inference-api.nousresearch.com/v1"
NOUS_API_CHAT_URL = f"{NOUS_API_BASE_URL}/chat/completions"

# Groq (無料ティア・OpenAI互換・高速)
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"   # スマートモデル（計画・回答生成）
GROQ_FAST_MODEL = "llama-3.1-8b-instant"          # 高速モデル（分類・軽量タスク）

# ドメイン別エージェント設定
DOMAIN_CONFIG: dict[str, dict] = {
    "general": {
        "success_criteria": ["目標を達成した", "結果を日本語で説明できる"],
        "constraints": ["まず現状を把握する"],
    },
    "coding": {
        "success_criteria": ["コードが動作する", "テストがパスする", "結果を日本語で説明できる"],
        "constraints": ["破壊的操作はしない", "まず現状を把握する"],
    },
    "research": {
        "success_criteria": ["情報を収集・整理できた", "信頼性を確認した", "結果を日本語でまとめられる"],
        "constraints": ["情報源を明記する", "推測と事実を区別する"],
    },
    "writing": {
        "success_criteria": ["文章を完成させた", "目的に合った表現になっている", "結果を日本語で説明できる"],
        "constraints": ["ユーザーの意図を尊重する", "簡潔かつ明確に書く"],
    },
    "data": {
        "success_criteria": ["データを分析できた", "洞察を抽出した", "結果を日本語で説明できる"],
        "constraints": ["データの整合性を保つ", "まず現状を把握する"],
    },
    "ops": {
        "success_criteria": ["タスクを実行できた", "結果を確認した", "結果を日本語で説明できる"],
        "constraints": ["破壊的操作はしない", "まず現状を確認する"],
    },
}
