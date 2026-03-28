from hermes_agi_gen import AgentState, HermesAgentV9


def test_agent_runs(tmp_path):
    agent = HermesAgentV9(repo_root=tmp_path, max_iterations=2)
    state = AgentState(user_goal="テスト", max_iterations=2)
    final_state = agent.run(state)
    assert final_state.iteration_count >= 1
