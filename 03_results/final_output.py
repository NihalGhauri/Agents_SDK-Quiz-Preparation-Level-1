from agents import Agent , OpenAIChatCompletionsModel, AsyncOpenAI, Runner , set_tracing_export_api_key
import os
from dataclasses import dataclass

set_tracing_export_api_key(os.getenv("OPENAI_API_KEY"))


external_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


external_model= OpenAIChatCompletionsModel(
    openai_client=external_client,
    model= "gemini-1.5-flash",
)

@dataclass
class user:
    name: str
    age:int


agent = Agent(
    name= "assistant",
    instructions="A helpful assistant that can answer questions and provide information.",
    model=external_model,
    output_type=user,
)

question = "my name is Nihal and i am 30 years old"

result = Runner.run_sync(agent, question)
print(result.final_output)
print(type(result.final_output))