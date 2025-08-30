import asyncio
from openai.types.responses import ResponseTextDeltaEvent
from agents import (
    Agent,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    set_tracing_export_api_key,
    enable_verbose_stdout_logging
)
import os

enable_verbose_stdout_logging()

Open_Ai_key = os.getenv("OPENAI_API_KEY")
set_tracing_export_api_key(Open_Ai_key)


external_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

external_model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model="gemini-1.5-flash",
)

config = RunConfig(model=external_model, model_provider=external_client)

async def main():
    agent = Agent(
        name="Joker",
        instructions="You are a helpful assistant.",
        model=external_model,
    )

    result = Runner.run_streamed(agent, "Please tell me 1 jokes about programming.")
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(
            event.data, ResponseTextDeltaEvent
        ):
            print(event.data.delta, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
