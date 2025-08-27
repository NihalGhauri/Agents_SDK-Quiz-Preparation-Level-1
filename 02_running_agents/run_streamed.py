from agents import Agent , Runner, AsyncOpenAI , OpenAIChatCompletionsModel, set_tracing_disabled
from openai.types.responses import ResponseTextDeltaEvent
import os
import asyncio


set_tracing_disabled(True)

external_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

external_model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model= "gemini-1.5-flash",
)


async def main():
   
    agent = Agent(
        name="assistant",
        instructions="A helpful assistant that can answer questions and provide information.",
        model=external_model,
    )

    result = Runner.run_streamed(agent,"who is full stack developer?")
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data,ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)

asyncio.run(main())


#     print(result.final_output)

# asyncio.run(main())