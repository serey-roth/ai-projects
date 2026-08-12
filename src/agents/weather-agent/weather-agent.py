import argparse
import asyncio
from dataclasses import dataclass

from llama_index.llms.anthropic import Anthropic
from llama_index.core.llms import ChatMessage
from llama_index.core.agent.workflow import FunctionAgent # tool-calling agent from llamaindex

import python_weather

from llama_index.llms.anthropic import Anthropic


@dataclass
class WeatherForecast:
    temp: int
    humidity: int
    precipitation: int
    

async def get_weather_forecast(location: str):
    """
    Get today's weather forecast for a given location.
    
    The weather forecaset includes the current temperature in Celsius,
    humidity and precipitation.
    """
    async with python_weather.Client() as client:
        weather = await client.get(location)
        return WeatherForecast(
            temp=weather.temperature,
            humidity=weather.humidity,
            precipitation=weather.precipitation,
        )
    
async def chat(user_message: str):
    llm = Anthropic(model="claude-haiku-4-5", max_tokens=1024)
    agent = FunctionAgent(
        name="weather-agent",
        tools=[get_weather_forecast],
        llm=llm,
        system_prompt="You are a weather forecast agent that answers users' question regarding the current weather using tools.",
        streaming=True
    )
    response = await agent.run(user_msg=user_message, max_iterations=10)
    print(response)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat agent")
    parser.add_argument("--message")
    args = parser.parse_args()
    
    asyncio.run(chat(args.message))
    
