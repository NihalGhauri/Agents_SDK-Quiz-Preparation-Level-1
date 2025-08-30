from agents import (
        Agent,
        AsyncOpenAI,
        OpenAIChatCompletionsModel,
        Runner,
        RunConfig,
        set_tracing_export_api_key,
    )
import os
from dotenv import load_dotenv
load_dotenv()

Open_Ai_key = os.getenv("OpenAI_API_Key")
set_tracing_export_api_key(
    Open_Ai_key
)

external_client =AsyncOpenAI(
    api_key= os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

external_model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model="gemini-1.5-flash",
)

config = RunConfig(
    model=external_model,
    model_provider=external_client

)

agent = Agent(
    name="assistant",
    instructions="A helpful assistant that can answer questions and provide information.",
    model=external_model,

)
result = Runner.run_sync(agent, "how many states in pakistan?")


print(f"Result: {result.final_output}")
