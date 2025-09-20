from agents.extensions import handoff_filters
from agents import Agent , function_tool , Runner , OpenAIChatCompletionsModel , set_tracing_disabled , RunContextWrapper , handoff , AsyncOpenAI
import os

external_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
external_model = OpenAIChatCompletionsModel(
    openai_client=external_client, model="gemini-1.5-flash"
)


@function_tool
def get_weather(city:str) -> str:
    """ returns weather info for the specified city"""
    return f"{city} is cloudy"


@function_tool
def sum_numbers(a:int , b: int) -> int:
    """ adds two numbers"""
    return a + b

async def on_handoff(ctx:RunContextWrapper[None]):
    print(f"handoff called")


faq_agent = Agent(
    name="faq agent ",
    tools=[get_weather,sum_numbers],
    model=external_model
)

handoff_obj = handoff(
    agent= faq_agent,
    input_filter=handoff_filters.remove_all_tools,
    on_handoff=on_handoff
)

agent = Agent(
    name="triage agent",
    instructions="""
    you are triage agent and your task is to handoff user queries according provided agent description.
    if user query is not related to handoff than answer to user through agent.
    your task is to answer the user query using tools if you dont have specific tools than handoff to faq_agent and use its tools.


""",
    model=external_model,
    handoffs=[handoff_obj],
    tools=[get_weather,sum_numbers]

)

result = Runner.run_sync(agent,"what is today weather in karachi")

print("final output -> : ",result.final_output)
print("last agent -> : ", result.last_agent.name)
