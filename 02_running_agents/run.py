from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
)
import os
import asyncio
import rich


set_tracing_disabled(True)

external_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

external_model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model="gemini-1.5-flash",
)


async def main():
    agent =Agent(
        name="Assistant",
        instructions="you are best Assistant",
        model=external_model,
    )
    result = await Runner.run(agent,"2+2")
    rich.print(result.final_output)

asyncio.run(main())