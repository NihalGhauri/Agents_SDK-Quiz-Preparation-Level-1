from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
    SQLiteSession,
    trace,
)
import os
import asyncio

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
    agent = Agent(
        name="assistant",
        instructions="A helpful assistant that can answer questions and provide information.",
        model=external_model,
    )
    thread_id = "nihal_123"
    session = SQLiteSession("conversion_123")

    # ? ====> first turn <=====

    with trace(workflow_name="Conversation" , group_id=thread_id):
        result = await Runner.run(agent, "who is full stack developer?")
        print("==========================")
        print(result.final_output)
        print("==========================")

        # ? ====> Second turn <=====

        result = await Runner.run(agent,"and skills they have", session=session)
        print("==========================")
        print( result.final_output)
        print("==========================")
        
        result = await Runner.run(agent,"and what is average salary in pakistan?", session=session)
        print("==========================")
        print( result.final_output)
        print("==========================")


asyncio.run(main())
