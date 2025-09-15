from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI
import os
import asyncio

external_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

external_model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model="gemini-2.0-flash",
)

spanish_agent = Agent(
    name="spanish agent",
    instructions="you translate the user message to spanish"
)

french_agent = Agent(
    name="french agent",
    instructions="you translate the user message to french"
)

orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions=(
        "You are a translation agent. You use the tools given to you to translate."
        "If asked for multiple translations, you call the relevant tools."
    ),
    tools=[
        spanish_agent.as_tool(
            tool_name="translate_to_spanish",
            tool_description="Translate the user's message to Spanish",
        ),
        french_agent.as_tool(
            tool_name="translate_to_french",
            tool_description="Translate the user's message to French",
        ),
    ],
)

async def main():
    result = await Runner.run(orchestrator_agent,input="say hello how are you in spanish")
    print(result.final_output)

asyncio.run(main())