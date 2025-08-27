from agents import Agent ,AsyncOpenAI, OpenAIChatCompletionsModel, Runner, RunConfig
import os

external_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

external_model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model= "gemini-1.5-flash",
)

config = RunConfig(
    model=external_model,
    model_provider=external_client,
)

English_agent = Agent(
    name="English assistant",
    instructions="A helpful assistant that can answer questions and provide information in English.",
    model=external_model,
)

History_agent = Agent(
    name="History assistant",
    instructions="A helpful assistant that can answer questions and provide information in History.",
    model=external_model,
)

Triage_agent = Agent(
    name="Triage assistant",
    instructions="A helpful assistant that can answer questions and provide information in Triage.",
    handoffs=[English_agent, History_agent],
    model=external_model,
    )

result = Runner.run_sync(Triage_agent, "moen jo daro is a historical place in Pakistan and it is very old. Can you tell me more about it?")

print(f"Result: {result.final_output}")
print({result.last_agent.name})