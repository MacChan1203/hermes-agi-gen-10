"""Gen 10.2 — 離散トークン通信レイヤーのテスト。

token_codebook / peer_channel / token_interpreter の単体テストと、
code_agents 経由の Generator↔Reviewer 統合テスト、
BellmanEvaluator の peer_reward_hook 配線テストを含む。
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from hermes_agi_gen.code_agents import CodeGeneratorAgent, CodeReviewerAgent
from hermes_agi_gen.peer_channel import PeerChannel
from hermes_agi_gen.token_codebook import TokenCodebook
from hermes_agi_gen.token_interpreter import TokenInterpreter


# ======================================================================
# TokenCodebook
# ======================================================================
class TestTokenCodebook:
    def test_default_vocab_is_loaded(self):
        cb = TokenCodebook()
        assert cb.vocab_size >= 5
        assert "T_OTHER" in cb.ids()
        assert "T_ALGO" in cb.ids()

    def test_emit_picks_token_by_keyword(self):
        cb = TokenCodebook()
        assert cb.emit("Pythonでクイックソートを実装") == "T_ALGO"
        assert cb.emit("FastAPI で http サーバを作って") == "T_WEB"
        assert cb.emit("pytest で test を書きたい") == "T_TEST"

    def test_emit_falls_back_when_no_keyword_match(self):
        cb = TokenCodebook()
        assert cb.emit("xxx zzz") == "T_OTHER"

    def test_emit_increments_usage_and_records_example(self):
        cb = TokenCodebook()
        cb.emit("テストコードを書く")
        cb.emit("もう一つのテスト依頼")
        stats = cb.get("T_TEST")
        assert stats is not None
        assert stats.n_used == 2
        assert len(stats.examples) == 2

    def test_record_reward_moves_avg_via_ema(self):
        cb = TokenCodebook()
        before = cb.get("T_ALGO").avg_reward
        new_avg = cb.record_reward("T_ALGO", reward=1.0)
        assert new_avg > before
        # 何度も高報酬を入れれば 1 に近づく
        for _ in range(20):
            cb.record_reward("T_ALGO", reward=1.0)
        assert cb.get("T_ALGO").avg_reward > 0.9

    def test_bonus_for_clipped_to_range(self):
        cb = TokenCodebook()
        # 強化前は中立 → 0 付近
        assert abs(cb.bonus_for("T_ALGO")) < 0.1
        # 高報酬を流せば正のボーナス
        for _ in range(30):
            cb.record_reward("T_ALGO", reward=1.0)
        assert cb.bonus_for("T_ALGO") > 0.0
        # 低報酬を流せば負のボーナス
        for _ in range(30):
            cb.record_reward("T_FIX", reward=0.0)
        assert cb.bonus_for("T_FIX") < 0.0

    def test_bonus_for_unknown_token_is_zero(self):
        cb = TokenCodebook()
        assert cb.bonus_for("NOPE") == 0.0

    def test_snapshot_roundtrip(self):
        cb = TokenCodebook()
        cb.emit("ソート アルゴリズム")
        cb.record_reward("T_ALGO", reward=0.9)
        snap = cb.snapshot()

        cb2 = TokenCodebook()
        cb2.load_snapshot(snap)
        assert cb2.get("T_ALGO").n_used == cb.get("T_ALGO").n_used
        assert cb2.get("T_ALGO").avg_reward == pytest.approx(cb.get("T_ALGO").avg_reward)

    def test_snapshot_preserves_examples(self):
        """P1-❹: snapshot/load で examples (使用例) も復元される。"""
        cb = TokenCodebook()
        cb.emit("クイックソートを実装")
        cb.emit("マージソートを書く")
        snap = cb.snapshot()

        cb2 = TokenCodebook()
        cb2.load_snapshot(snap)
        ex2 = cb2.get("T_ALGO").examples
        assert len(ex2) == 2
        assert "クイックソート" in ex2[0]
        assert "マージソート" in ex2[1]

    def test_lookup_is_side_effect_free(self):
        """P0-❷: lookup() は n_used / examples を更新しない。"""
        cb = TokenCodebook()
        before_n = cb.get("T_ALGO").n_used
        before_ex = list(cb.get("T_ALGO").examples)
        tok = cb.lookup("ソート アルゴリズム")
        assert tok == "T_ALGO"
        assert cb.get("T_ALGO").n_used == before_n
        assert cb.get("T_ALGO").examples == before_ex

    def test_emit_score_uses_avg_reward_to_break_ties_in_favor_of_strong(self):
        """P0-❶: hits 同点なら avg_reward が高いトークンが選ばれる (RL ループ閉)。"""
        # T_TEST と T_OTHER で hits=0 にならないように、両方マッチする語彙を作る
        custom_vocab = (
            ("T_OTHER", "fb",  (),                      ()),
            ("T_A",     "A",   ("kw",),                 ()),
            ("T_B",     "B",   ("kw",),                 ()),
        )
        cb = TokenCodebook(vocab=custom_vocab, fallback_id="T_OTHER")
        # 両者とも "kw" で hits=1。avg_reward=0.5 同点。
        # T_B を強化すると、次の emit で T_B が選ばれるはず。
        for _ in range(10):
            cb.record_reward("T_B", reward=1.0)
        chosen = cb.emit("kw")
        assert chosen == "T_B"

    def test_explicit_fallback_id(self):
        """⓫: fallback_id を明示的に指定できる。不正値は ValueError。"""
        custom_vocab = (
            ("X", "x", ("a",), ()),
            ("Y", "y", ("b",), ()),
        )
        cb = TokenCodebook(vocab=custom_vocab, fallback_id="Y")
        # マッチなし → Y が返る
        assert cb.emit("zzz") == "Y"
        with pytest.raises(ValueError):
            TokenCodebook(vocab=custom_vocab, fallback_id="NOTREAL")
        with pytest.raises(ValueError):
            TokenCodebook(vocab=())


# ======================================================================
# PeerChannel
# ======================================================================
class TestPeerChannel:
    def test_send_and_receive(self):
        ch = PeerChannel()
        ch.send(sender="generator", receiver="reviewer", task="x", tokens=["T_ALGO"])
        msgs = ch.receive("reviewer")
        assert len(msgs) == 1
        assert msgs[0].sender == "generator"
        assert PeerChannel.tokens_of(msgs[0]) == ["T_ALGO"]

    def test_receive_drains_inbox(self):
        ch = PeerChannel()
        ch.send("g", "r", "t", tokens=["T1"])
        ch.receive("r")
        assert ch.receive("r") == []

    def test_peek_does_not_drain(self):
        ch = PeerChannel()
        ch.send("g", "r", "t", tokens=["T1"])
        assert len(ch.peek("r")) == 1
        assert len(ch.peek("r")) == 1
        assert len(ch.receive("r")) == 1

    def test_history_accumulates(self):
        ch = PeerChannel()
        ch.send("g", "r", "a")
        ch.send("g", "r", "b")
        assert len(ch.history()) == 2

    def test_max_inbox_evicts_oldest(self):
        ch = PeerChannel(max_inbox=2)
        for i in range(5):
            ch.send("g", "r", f"t{i}")
        msgs = ch.receive("r")
        # deque maxlen=2 なので最後の 2 件のみ残る
        assert len(msgs) == 2
        assert msgs[0].task == "t3"
        assert msgs[1].task == "t4"

    def test_tokens_of_handles_non_dict_context(self):
        from hermes_agi_gen.agent_message import AgentMessage
        msg = AgentMessage(sender="a", receiver="b", task="x", context="just a string")
        assert PeerChannel.tokens_of(msg) == []


# ======================================================================
# TokenInterpreter
# ======================================================================
class TestTokenInterpreter:
    def test_interpret_known_tokens(self):
        cb = TokenCodebook()
        cb.emit("ソート アルゴリズム")
        ti = TokenInterpreter(cb)
        out = ti.interpret(["T_ALGO"])
        assert "T_ALGO" in out
        assert "アルゴリズム" in out

    def test_interpret_unknown_token(self):
        cb = TokenCodebook()
        ti = TokenInterpreter(cb)
        assert "?" in ti.interpret(["NOT_A_TOKEN"])

    def test_explain_one(self):
        cb = TokenCodebook()
        cb.emit("テスト 書く")
        cb.record_reward("T_TEST", reward=1.0)
        ti = TokenInterpreter(cb)
        s = ti.explain_one("T_TEST")
        assert "T_TEST" in s and "テスト" in s

    def test_interpret_empty_list(self):
        cb = TokenCodebook()
        assert TokenInterpreter(cb).interpret([]) == "(empty)"


# ======================================================================
# Integration: CodeGeneratorAgent ↔ CodeReviewerAgent
# ======================================================================
def _mock_llm(reply: str) -> MagicMock:
    llm = MagicMock()
    llm.chat.return_value = reply
    return llm


class TestGeneratorReviewerTokenComms:
    def test_generator_emits_token_to_reviewer(self):
        ch = PeerChannel()
        cb = TokenCodebook()
        gen = CodeGeneratorAgent(
            llm=_mock_llm("```python\ndef sort(): pass\n```"),
            peer_channel=ch,
            codebook=cb,
        )
        gen.generate("クイックソートを実装してください")

        # 受信箱に T_ALGO が届いている
        msgs = ch.peek("reviewer")
        assert len(msgs) == 1
        assert "T_ALGO" in PeerChannel.tokens_of(msgs[0])

    def test_reviewer_consumes_tokens_and_rewards_on_match(self):
        ch = PeerChannel()
        cb = TokenCodebook()
        gen = CodeGeneratorAgent(
            llm=_mock_llm("(unused)"),
            peer_channel=ch,
            codebook=cb,
        )
        rev = CodeReviewerAgent(
            llm=_mock_llm("## 概要\nOK"),
            peer_channel=ch,
            codebook=cb,
        )

        gen.generate("クイックソートを実装してください")
        before = cb.get("T_ALGO").avg_reward
        # T_ALGO の期待語のうち過半 (def/for/while/return/compare/swap/partition)
        # を満たすクイックソート風コード
        quicksort_code = (
            "def quicksort(arr):\n"
            "  if len(arr) <= 1:\n"
            "    return arr\n"
            "  pivot = arr[0]\n"
            "  def partition(xs):\n"
            "    for x in xs:\n"
            "      while x and compare(x, pivot):\n"
            "        swap(xs)\n"
            "    return xs\n"
        )
        rev.review(quicksort_code)
        after = cb.get("T_ALGO").avg_reward
        assert after > before  # 高一致率 → 正の reward
        # 受信箱は消費済み
        assert ch.peek("reviewer") == []

    def test_reviewer_penalizes_token_on_mismatch(self):
        ch = PeerChannel()
        cb = TokenCodebook()
        gen = CodeGeneratorAgent(
            llm=_mock_llm("(x)"), peer_channel=ch, codebook=cb,
        )
        rev = CodeReviewerAgent(
            llm=_mock_llm("ng"), peer_channel=ch, codebook=cb,
        )
        # avg_reward 初期 0.5 から、期待外れ (誤差大) で下がるか
        gen.generate("テストコードを書く")  # → T_TEST (期待: assert / def test / pytest / unittest)
        before = cb.get("T_TEST").avg_reward
        rev.review("print('hello')")  # 期待語ゼロ → 誤差 1.0 → reward 0.0
        after = cb.get("T_TEST").avg_reward
        assert after < before

    def test_reviewer_predict_returns_expected_patterns_from_vocab(self):
        """P3-❽: Reviewer.predict() は受信トークンから期待パターン集合を立てる。"""
        from hermes_agi_gen.code_agents import CodePrediction

        ch = PeerChannel()
        cb = TokenCodebook()
        gen = CodeGeneratorAgent(llm=_mock_llm("x"), peer_channel=ch, codebook=cb)
        rev = CodeReviewerAgent(llm=_mock_llm("x"), peer_channel=ch, codebook=cb)

        gen.generate("クイックソートを実装")
        prediction = rev.predict()
        assert isinstance(prediction, CodePrediction)
        assert prediction.token_ids == ("T_ALGO",)
        # 期待パターンは vocab 由来 (config の expected_patterns) が乗っている
        assert "def " in prediction.expected_patterns
        assert "partition" in prediction.expected_patterns

    def test_reviewer_predict_returns_none_without_received_tokens(self):
        ch = PeerChannel()
        cb = TokenCodebook()
        rev = CodeReviewerAgent(llm=_mock_llm("x"), peer_channel=ch, codebook=cb)
        assert rev.predict() is None

    def test_reviewer_review_appends_interpreter_summary_to_output(self):
        """P2-❻: TokenInterpreter がライブパスから使われ、出力末尾に解釈が付く。"""
        ch = PeerChannel()
        cb = TokenCodebook()
        gen = CodeGeneratorAgent(llm=_mock_llm("x"), peer_channel=ch, codebook=cb)
        rev = CodeReviewerAgent(llm=_mock_llm("## 概要\nOK"), peer_channel=ch, codebook=cb)
        gen.generate("クイックソート")
        out = rev.review("def quicksort(a): return a")
        # 解釈層の出力がレビュー末尾に付与されている
        assert "[内部通信]" in out
        assert "T_ALGO" in out
        assert "予測誤差" in out

    def test_no_codebook_means_no_emission(self):
        """codebook/channel 未指定の従来挙動が壊れていないことを確認。"""
        ch = PeerChannel()
        gen = CodeGeneratorAgent(llm=_mock_llm("ok"))  # peer_channel/codebook 無し
        gen.generate("何でもいい")
        assert ch.peek("reviewer") == []


# ======================================================================
# BellmanEvaluator: peer_reward_hook が即時報酬に乗ること
# ======================================================================
class TestBellmanPeerRewardHook:
    def test_hook_adds_bonus_to_reward(self):
        from hermes_agi_gen.bellman_planner import BellmanEvaluator

        # ValueSystem のモック (utility_score は goal_relevance を返すだけにする)
        vs = MagicMock()
        vs.utility_score.side_effect = lambda action, goal_relevance: goal_relevance

        # フック無し
        ev_plain = BellmanEvaluator(value_system=vs)
        r0 = ev_plain.reward("ANSWER: foo", "foo")

        # フックは常に +0.2 を返す
        ev_hooked = BellmanEvaluator(
            value_system=vs,
            peer_reward_hook=lambda action: 0.2,
        )
        r1 = ev_hooked.reward("ANSWER: foo", "foo")
        assert r1 == pytest.approx(r0 + 0.2)

    def test_agent_runner_wires_peer_reward_hook_when_codebook_given(self):
        """P0-❷: HermesAgentV10(use_bellman=True, codebook=...) が hook を実配線する。"""
        from hermes_agi_gen.agent_runner import HermesAgentV10

        cb = TokenCodebook()
        for _ in range(20):
            cb.record_reward("T_ALGO", reward=1.0)  # T_ALGO を強化

        agent = HermesAgentV10(use_bellman=True, codebook=cb)
        ev = agent.bellman_planner.evaluator
        assert ev.peer_reward_hook is not None

        # ソート系の action は T_ALGO に lookup される → 正のボーナス
        bonus_strong = ev.peer_reward_hook("PYTHON: クイックソート アルゴリズム")
        assert bonus_strong > 0.0

    def test_agent_runner_no_codebook_means_no_hook(self):
        """codebook を渡さない場合は peer_reward_hook が None (従来挙動)。"""
        from hermes_agi_gen.agent_runner import HermesAgentV10
        agent = HermesAgentV10(use_bellman=True)
        assert agent.bellman_planner.evaluator.peer_reward_hook is None

    def test_hook_exception_is_swallowed(self):
        from hermes_agi_gen.bellman_planner import BellmanEvaluator

        vs = MagicMock()
        vs.utility_score.return_value = 0.5

        def bad_hook(action):
            raise RuntimeError("boom")

        ev = BellmanEvaluator(value_system=vs, peer_reward_hook=bad_hook)
        # 例外は握りつぶされて base reward が返るはず
        r = ev.reward("ANSWER: x", "x")
        assert isinstance(r, float)
