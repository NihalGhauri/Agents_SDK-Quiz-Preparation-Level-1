from agents import (
    Agent,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    function_tool,
    handoff,
    enable_verbose_stdout_logging,
    RunContextWrapper,
    Runner,
    set_tracing_export_api_key
)
import os

set_tracing_export_api_key(os.getenv("OPENAI_API_KEY"))

external_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
external_model=OpenAIChatCompletionsModel(
    openai_client=external_client,
    model="gemini-1.5-flash")

def on_hand_off(ctx:RunContextWrapper[None]):
    print("hand off called!")


math_tutor = Agent(
    name="math tutor",
    instructions="you are a math teacher. Answer only math related queries.",
    model=external_model,
    handoff_description="you are a math teacher . Answer only math related queries "

)
history_tutor = Agent(
    name="history tutor",
    instructions="you are a history teacher. Answer only history related queries.",
    model=external_model,
    handoff_description="you are a history teacher . Answer only history related queries "

)
biology_tutor = Agent(
    name="biology tutor",
    instructions="you are a biology teacher. Answer only biology related queries.",
    model=external_model,
    handoff_description="you are a biology teacher . Answer only biology related queries "

)

triage = Agent(
    name="triage tutor",
    instructions="you are a triage agent and your task is to handoff user queries according provided agent description" \
    "if user queries is not related to handoff agents than answer to user through triage agent ",
    model=external_model,
    handoffs=[math_tutor, history_tutor,biology_tutor ]
)

result = Runner.run_sync(triage,"what sum of 2 +2")
print(result.final_output)
print(result.last_agent.name)
