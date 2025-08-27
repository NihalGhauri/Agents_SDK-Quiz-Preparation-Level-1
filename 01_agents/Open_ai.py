from agents import Agent, OpenAIChatCompletionsModel, Runner
from dotenv import load_dotenv
import os

load_dotenv()

AI = os.getenv("OpenAI_API_Key")

agent = Agent(
    name="assistant",
    instructions="A helpful assistant that can answer questions and provide information.",
    model=OpenAIChatCompletionsModel(model="gpt-3.5-turbo", openai_api_key=AI)
)
result = Runner.run_sync(agent, "who is full stack developer?")
print(result.final_output)