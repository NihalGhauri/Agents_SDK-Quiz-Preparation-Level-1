from pydantic import BaseModel
from agents import (
    Agent,
    GuardrailFunctionOutput,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    output_guardrail,
    set_tracing_disabled,
    enable_verbose_stdout_logging,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
)
import os
import asyncio

enable_verbose_stdout_logging()
external_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
external_model = OpenAIChatCompletionsModel(
    model= 'gemini-2.0-flash',
    openai_client=external_client
)

class MessageOutput(BaseModel):
    response : str

class MathHomeWorkOutput(BaseModel):
    is_math : bool
    reasoning : str


guardrail_agent = Agent(
    name="Guardrail check",
    instructions="check if the output includes any math",
    output_type=MathHomeWorkOutput,
    model=external_model
)


@output_guardrail
async def math_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, output: MessageOutput
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, output.response,context = ctx.content )

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_math
    )

agent = Agent(
    name="customer support agent",
    instructions='you are a customer support agent . you help customers with their questions  ',
    output_type=MessageOutput,
    model=external_model
)

async def main():
    try:
        await Runner.run(agent, "hello can you help me solve for x: 2x + + ")