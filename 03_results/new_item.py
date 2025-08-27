from agents import Agent , function_tool , Runner , OpenAIChatCompletionsModel , set_tracing_export_api_key , RunContextWrapper, AsyncOpenAI , enable_verbose_stdout_logging
import os

set_tracing_export_api_key(os.getenv("OPENAI_API_KEY"))

enable_verbose_stdout_logging()

external_client = AsyncOpenAI(
    api_key= os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

external_model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model="gemini-1.5-flash",
)

@function_tool
def get_weather(city: str) -> str:
    """Returns the current weather for a given city."""
    return f"{city} is sunny"

@function_tool
def sum_num(a:int , b:int) -> int:
    """return sum two number """
    return (a+b+1)

math_tutor = Agent(
    name="math_tutor",
    instructions="you are a math tutor and answer to only math related queries.",
    model=external_model,
    tools=[sum_num],
    handoff_description="you are a math tutor and answer to only math related queries",
)

history_agent=Agent(
    name="history_agent",
    instructions="you are a history tutor and answer to only history related queries.",
    model=external_model,
    handoff_description="you are a history tutor and answer to only history related queries."
)

triage_agent = Agent(
    name="Triage ",
    instructions="You are triage agent and your task is to handoff user queries according provided agent description."
    "if user query is not related to handsoff agents than answer to user through triage agent.",
    model=external_model,
    tools=[get_weather],
    handoffs=[history_agent,math_tutor]
)


result = Runner.run_sync(triage_agent, "what is 2+2")
print(result.final_output)
for item in result.new_items:
    print(type(item).__name__)
