from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
)
import os


set_tracing_disabled(True)

external_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

external_model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model="gemini-1.5-flash",
)
agent =Agent(
        name="Assistant",
        instructions="you are best Assistant",
        model=external_model,
    )
result =  Runner.run_sync(agent,"2+2")



