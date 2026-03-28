from hermes_agi_gen import AgentOrchestrator, MistralClient
llm = MistralClient()
orch = AgentOrchestrator(llm=llm)
print(orch.run('このプロジェクトの構造を調べてください'))
