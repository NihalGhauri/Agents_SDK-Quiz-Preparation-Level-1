from agents import Agent, Runner,AsyncOpenAI, OpenAIChatCompletionsModel,set_tracing_disabled,trace
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
    thread_id = "nihal_123"

    with trace(workflow_name="Conversation ", group_id=thread_id ):

        # ! first turn

        result = await Runner.run(agent, "who is city of lights")
        print( result.final_output)

        # ! second turn

        new_turn = result.to_input_list() + [
            {"role": "user", "content": "what state in it? "}
        ]
        result = await Runner.run(agent, new_turn)
        print( result.final_output)

asyncio.run(main())
