from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
    trace,
    set_tracing_export_api_key,
    Runner,
    RunConfig,
)
import asyncio
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
    age: int
    location: str

config = RunConfig(
    model=external_model,
    model_provider=external_client
)


async def main():
    agent = Agent(
        name="Assistant",
        instructions="A helpful assistant that can answer questions and provide information.",
    )
    thread_id = "thread_123"
    with trace(workflow_name="Testing", group_id=thread_id):
        result = await Runner.run(agent,"who is the president of USA?", run_config=config)
        print(result.final_output)

        new_result = result.to_input_list() + [{
            "role": "user",
            "content": "where does he live?"
        }]
        result = await Runner.run(agent, new_result,run_config=config)

        print(result.final_output)
        print(result.last_agent.name)

asyncio.run(main())