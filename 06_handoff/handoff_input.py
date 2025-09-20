from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    handoff,
    RunContextWrapper,
    set_tracing_export_api_key,
)
import os
from pydantic import BaseModel

set_tracing_export_api_key(os.getenv("OPENAI_API_KEY"))

external_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
external_model=OpenAIChatCompletionsModel(
    openai_client=external_client,
    model="gemini-1.5-flash")


class EscalationData(BaseModel):
    reason: str


async def on_handoff(ctx:RunContextWrapper[None], input_data: EscalationData):
    print(f"Escalation agent called with reason: {input_data.reason}")

escalating_agent = Agent(
    name="Escalation Agent",
    instructions="You are an escalating agent your task is to take imediate action on the response you recive from triage agent and handle the situation professionaly.",
    model=external_model,
)

handoff_obj = handoff(
    agent = escalating_agent,
    on_handoff=on_handoff,
    input_type=EscalationData

)
agent = Agent(
    name="triage agent",
    instructions="""
You are a triage agent. If the user's query is urgent, angry, or mentions legal action, refund, or complaint, 
you must handoff to escalating agent with a JSON object: {"reason": "your reason here"}.

Do NOT say anything else if escalating transfer to escalating_agent to handle the situation. If no escalation is needed, answer normally.
""",
    model=external_model,
    handoffs=[handoff_obj]
)

Result = Runner.run_sync(
    agent,
    "i m furious your service is terrible and i want to speak to a manager"
)

print(Result.last_agent.name)
print(Result.final_output)
