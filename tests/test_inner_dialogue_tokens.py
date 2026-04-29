"""Gen 10.2 #2 — InnerDialogue × 離散トークン通信の統合テスト。

CognitiveRole 同士が PeerChannel + TokenCodebook で内部対話する経路を検証する。
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from hermes_agi_gen.config import TOKEN_CODEBOOK_INNER_VOCAB
from hermes_agi_gen.inner_dialogue import InnerDialogue
from hermes_agi_gen.peer_channel import PeerChannel
from hermes_agi_gen.token_codebook import TokenCodebook


def _make_dialogue_llm(
    critic_stance="oppose",
    innovator_stance="extend",
    ethicist_stance="qualify",
    consensus=0.8,
    should_proceed=True,
) -> MagicMock:
    """Phase1 (3 ロール) → Phase2 (戦略家) の chat_json を順に返すモック。"""
    llm = MagicMock()
    llm.chat_json.side_effect = [
        {
            "criticism": "リスクが高い",
            "risks": ["risk1", "risk2"],
            "stance": critic_stance,
            "confidence": 0.7,
        },
        {
            "innovation": "alternative案を出す",
            "alternatives": ["alt1"],
            "stance": innovator_stance,
            "confidence": 0.6,
        },
        {
            "assessment": "倫理面で懸念",
            "concerns": ["公正性"],
            "stance": ethicist_stance,
            "confidence": 0.5,
        },
        {
            "strategy": "段階的に進める",
            "refined_goal": "refined",
            "consensus": consensus,
            "should_proceed": should_proceed,
        },
    ]
    return llm


class TestInnerDialogueTokens:
    def test_deliberate_emits_three_tokens_to_strategist_channel(self):
        ch = PeerChannel()
        cb = TokenCodebook(vocab=TOKEN_CODEBOOK_INNER_VOCAB)
        dialogue = InnerDialogue(
            llm=_make_dialogue_llm(),
            peer_channel=ch,
            codebook=cb,
        )
        # strategist プロンプト構築前に受信箱を覗くため history で確認する
        result = dialogue.deliberate("テスト目標", "コンテキスト")
        # 3 ロール分のメッセージが strategist に届いた履歴があるはず
        msgs = ch.history()
        receivers = {m.receiver for m in msgs}
        assert receivers == {"strategist"}
        assert len(msgs) == 3
        # トークンが付随している
        for m in msgs:
            toks = PeerChannel.tokens_of(m)
            assert len(toks) == 1
            assert cb.get(toks[0]) is not None
        # 最終結果が想定通り
        assert result.refined_goal == "refined"

    def test_critic_oppose_maps_to_oppose_or_risk_token(self):
        ch = PeerChannel()
        cb = TokenCodebook(vocab=TOKEN_CODEBOOK_INNER_VOCAB)
        dialogue = InnerDialogue(
            llm=_make_dialogue_llm(critic_stance="oppose"),
            peer_channel=ch,
            codebook=cb,
        )
        dialogue.deliberate("X", "")
        critic_msgs = [m for m in ch.history() if m.sender == "critic"]
        assert len(critic_msgs) == 1
        tok = PeerChannel.tokens_of(critic_msgs[0])[0]
        # critic は "risk" シードがあるので T_RISK か stance 由来 T_OPPOSE
        assert tok in {"T_RISK", "T_OPPOSE"}

    def test_innovator_extend_maps_to_extend_or_creative(self):
        ch = PeerChannel()
        cb = TokenCodebook(vocab=TOKEN_CODEBOOK_INNER_VOCAB)
        dialogue = InnerDialogue(
            llm=_make_dialogue_llm(innovator_stance="extend"),
            peer_channel=ch,
            codebook=cb,
        )
        dialogue.deliberate("X", "")
        inn_msgs = [m for m in ch.history() if m.sender == "innovator"]
        tok = PeerChannel.tokens_of(inn_msgs[0])[0]
        assert tok in {"T_EXTEND", "T_CREATIVE"}

    def test_strategist_prompt_carries_actual_token_information(self):
        """strategist プロンプトに実際のトークン id とラベルが含まれる。

        ハードコードされた見出し文字列の存在ではなく、3 ロールが発火した
        トークンの id (T_*) かラベルがプロンプトに渡っていることを検証する。
        """
        ch = PeerChannel()
        cb = TokenCodebook(vocab=TOKEN_CODEBOOK_INNER_VOCAB)
        llm = _make_dialogue_llm(
            critic_stance="oppose",
            innovator_stance="extend",
            ethicist_stance="qualify",
        )
        dialogue = InnerDialogue(llm=llm, peer_channel=ch, codebook=cb)
        dialogue.deliberate("X", "")

        last_call = llm.chat_json.call_args_list[-1]
        prompt = last_call.args[0][0]["content"]

        # 発火したトークンの id がプロンプトに混入しているはず
        # (deliberate で発話前に emit() され、strategist 受信箱を消費して context へ)
        # _pending_tokens は record_outcome 前なのでまだ残っている
        emitted_ids = {tid for tid, _ in dialogue._pending_tokens}
        assert emitted_ids, "no token was emitted"
        # 各トークン id がプロンプトに含まれる
        for tid in emitted_ids:
            assert tid in prompt, f"token id {tid} not in prompt:\n{prompt}"

    def test_record_outcome_rewards_support_stance_on_success(self):
        """success=True 時、support/extend スタンスのトークンが強化される。"""
        ch = PeerChannel()
        cb = TokenCodebook(vocab=TOKEN_CODEBOOK_INNER_VOCAB)
        # 全員 support にして「賛成派が正しかった」状況を作る
        dialogue = InnerDialogue(
            llm=_make_dialogue_llm(
                critic_stance="support",
                innovator_stance="support",
                ethicist_stance="support",
            ),
            peer_channel=ch,
            codebook=cb,
        )
        dialogue.deliberate("X", "")
        pending = list(dialogue._pending_tokens)
        before = {tid: cb.get(tid).avg_reward for tid, _ in pending}
        dialogue.record_outcome("X", deliberation_was_used=True, success=True)
        after = {tid: cb.get(tid).avg_reward for tid, _ in pending}
        # support スタンスは success で reward=1.0 → 上がる
        for tid, _stance in pending:
            assert after[tid] > before[tid]
        assert dialogue._pending_tokens == []

    def test_record_outcome_rewards_oppose_stance_on_failure(self):
        """success=False 時、oppose/qualify スタンスのトークンが強化される。"""
        ch = PeerChannel()
        cb = TokenCodebook(vocab=TOKEN_CODEBOOK_INNER_VOCAB)
        dialogue = InnerDialogue(
            llm=_make_dialogue_llm(
                critic_stance="oppose",
                innovator_stance="qualify",
                ethicist_stance="oppose",
            ),
            peer_channel=ch,
            codebook=cb,
        )
        dialogue.deliberate("X", "")
        pending = list(dialogue._pending_tokens)
        before = {tid: cb.get(tid).avg_reward for tid, _ in pending}
        dialogue.record_outcome("X", deliberation_was_used=True, success=False)
        after = {tid: cb.get(tid).avg_reward for tid, _ in pending}
        # oppose/qualify は failure で reward=1.0 → 上がる
        for tid, _stance in pending:
            assert after[tid] > before[tid]

    def test_credit_assignment_differentiates_aligned_and_misaligned(self):
        """同じ deliberation で stance が異なるトークンは別方向に動く。"""
        ch = PeerChannel()
        cb = TokenCodebook(vocab=TOKEN_CODEBOOK_INNER_VOCAB)
        dialogue = InnerDialogue(
            llm=_make_dialogue_llm(
                critic_stance="oppose",     # success だと不正解 → reward 0
                innovator_stance="support", # success だと正解 → reward 1
                ethicist_stance="qualify",  # success だと不正解 → reward 0
            ),
            peer_channel=ch,
            codebook=cb,
        )
        dialogue.deliberate("X", "")
        pending = list(dialogue._pending_tokens)
        before = {tid: cb.get(tid).avg_reward for tid, _ in pending}
        dialogue.record_outcome("X", deliberation_was_used=True, success=True)
        after = {tid: cb.get(tid).avg_reward for tid, _ in pending}
        # support の token は上がる、oppose/qualify の token は下がる
        for tid, stance in pending:
            if stance == "support":
                assert after[tid] > before[tid]
            else:
                assert after[tid] < before[tid]

    def test_no_codebook_keeps_legacy_behavior(self):
        """codebook を渡さなければ従来の deliberate と同等で動く。"""
        dialogue = InnerDialogue(llm=_make_dialogue_llm())
        result = dialogue.deliberate("X", "")
        assert result.refined_goal == "refined"
        # トークンも pending も発生しない
        assert dialogue._pending_tokens == []

    def test_reward_only_when_deliberation_was_used(self):
        ch = PeerChannel()
        cb = TokenCodebook(vocab=TOKEN_CODEBOOK_INNER_VOCAB)
        dialogue = InnerDialogue(
            llm=_make_dialogue_llm(), peer_channel=ch, codebook=cb,
        )
        dialogue.deliberate("X", "")
        used = list(dialogue._pending_tokens)
        before = [cb.get(tid).avg_reward for tid, _ in used]
        # deliberation_was_used=False ではトークンに報酬を与えない
        dialogue.record_outcome("X", deliberation_was_used=False, success=True)
        after = [cb.get(tid).avg_reward for tid, _ in used]
        assert before == after
        # pending はそのまま残る (次の真の用法で消費される)
        assert dialogue._pending_tokens == used
