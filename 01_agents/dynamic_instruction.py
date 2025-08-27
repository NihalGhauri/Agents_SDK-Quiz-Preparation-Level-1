from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner , RunConfig , RunContextWrapper
import os
from dataclasses import dataclass


external_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

external_model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model="gemini-1.5-flash",
)
@dataclass
class instructions:
    name: str
    topic: str

def dynamic_instruction(
        context: RunContextWrapper[instructions],
        agent: Agent[instructions]
)-> str:
    return f"The name of the instruction is {context.context.name} and the topic is {context.context.topic}."

agent = Agent[instructions](
    name="assistant",
    instructions="A helpful assistant that can answer questions and provide information.",
    model=external_model,
)

result = Runner.run_sync(agent, "instructions(name='Dynamic Instruction', topic='Python Programming')")

print(f"Result: {result.final_output}")
