import asyncio
from pydantic import BaseModel
from openai import AsyncOpenAI
import os
from agents import Agent,OpenAIChatCompletionsModel, RunResult, ToolCallOutputItem, set_tracing_disabled, Runner

set_tracing_disabled(True)

external_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


external_model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model="gemini-1.5-flash"
)

async def extract_json_payload(run_result: RunResult) -> str:
    for item in reversed(run_result.new_items):
        if isinstance(item, ToolCallOutputItem) and item.output.strip().startswith("{"):
            return item.output.strip()
    
    return "{}"

data_agent = Agent(
    name="Data agent",
    model=external_model,
    instructions="you are the translate user message to spanish",
)


json_tool = data_agent.as_tool(
    tool_name="get_data_json",
    tool_description="Run the data agent and return only its json payload",
    custom_output_extractor=extract_json_payload,
)

orchestrator_agent = Agent(
    name="orchestrator agent",
    instructions="run the data agent and return only json payload",
    model=external_model,
    tools=[json_tool]
    )

async def main():
    result = await Runner.run(orchestrator_agent,"give me a json payload of prompt for making image")
    print(result.final_output)

asyncio.run(main())