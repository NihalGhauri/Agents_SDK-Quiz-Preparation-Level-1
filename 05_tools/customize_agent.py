from agents import Agent , AsyncOpenAI , OpenAIChatCompletionsModel, Runner, function_tool , RunConfig
import os
import asyncio


external_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

external_model = OpenAIChatCompletionsModel(
    openai_client=external_client, model="gemini-1.5-flash"
)

external_config = RunConfig(
    model=external_model,
    model_provider=external_client,

)

@function_tool
async def run_my_agent() -> str:
    """ a tool thats run the agent custom configs"""

    agent = Agent(name="My agent",instructions="you are a helpfull assistant")

    result = await Runner.run(
        agent,
        "2+2",
        max_turns=5,
        run_config=external_config
    )
    return str(result.final_output)

if __name__ == "__main__":
    asyncio.run(run_my_agent())