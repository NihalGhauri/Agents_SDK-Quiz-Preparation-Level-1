# from agents import (
#         Agent,
#         AsyncOpenAI,
#         OpenAIChatCompletionsModel,
#         Runner,
#         RunConfig,
#         set_tracing_export_api_key,
#     )
# import os
# from dotenv import load_dotenv
# load_dotenv()

# Open_Ai_key = os.getenv("OPENAI_API_KEY")
# set_tracing_export_api_key(
#     Open_Ai_key
# )

# external_client =AsyncOpenAI(
#     api_key= os.getenv("GEMINI_API_KEY"),
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
# )

# external_model = OpenAIChatCompletionsModel(
#     openai_client=external_client,
#     model="gemini-1.5-flash",
# )

# config = RunConfig(
#     model=external_model,
#     model_provider=external_client

# )

# agent = Agent(
#     name="assistant",
#     instructions="A helpful assistant that can answer questions and provide information.",
#     model=external_model,

# )
# result = Runner.run_sync(agent, "how many states in pakistan?")


# print(f"Result: {result.final_output}")


from typing import Any
from pydantic import BaseModel
from agents import (
    Agent,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    Runner,
    RunConfig,
    FunctionTool,
)
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API key
gemini_api_key = os.getenv("GEMINI_API_KEY")  # Replace with your actual Gemini API key

# Initialize AsyncOpenAI client for Gemini
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


# Define the tool function
def do_some_work(data: str) -> str:
    return f"Processed: {data}"


# Define the schema for function arguments
class FunctionArgs(BaseModel):
    username: str
    age: int


# Async function to run the tool
async def run_function(ctx: Any, args: str) -> str:
    parsed = FunctionArgs.model_validate_json(args)
    return do_some_work(data=f"{parsed.username} is {parsed.age} years old")


# Define the FunctionTool
tool = FunctionTool(
    name="process_user",
    description="Processes extracted user data",
    params_json_schema=FunctionArgs.model_json_schema(),
    on_invoke_tool=run_function,
)

# Configure the model
external_model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model="gemini-1.5-flash",
)

# Configure the run
config = RunConfig(model=external_model, model_provider=external_client)

# Create the agent with the tool
agent = Agent(
    name="assistant",
    instructions="A helpful assistant that can process user data and answer questions.",
    model=external_model,
    tools=[tool],
)

# Run the agent
result = Runner.run_sync(agent, 'Process user data: {"username": "Alice", "age": 25}')

# Print the result
print(f"Result: {result.final_output}")

