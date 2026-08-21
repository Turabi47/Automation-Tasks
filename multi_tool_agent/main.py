import asyncio
import logging
import warnings
from dotenv import load_dotenv

from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

from agent import weather_agent

# Load environment variables from .env
load_dotenv()

# Setup logging & warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

APP_NAME = "weather_tutorial_app"
USER_ID = "user_1"
SESSION_ID = "session_001"


async def call_agent_async(query: str, runner: Runner, user_id: str, session_id: str):
    """Sends a query to the agent and prints the final response."""
    print(f"\n>>> User Query: {query}")
    content = types.Content(role='user', parts=[types.Part(text=query)])
    final_response_text = "Agent did not produce a final response."

    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
            elif event.actions and event.actions.escalate:
                final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
            break

    print(f"<<< Agent Response: {final_response_text}")


async def main():
    # Setup Session Service and Runner
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )

    runner = Runner(
        agent=weather_agent,
        app_name=APP_NAME,
        session_service=session_service
    )

    # Run conversation sequence
    await call_agent_async("What is the weather like in London?", runner, USER_ID, SESSION_ID)
    await call_agent_async("How about Paris?", runner, USER_ID, SESSION_ID)
    await call_agent_async("Tell me the weather in New York", runner, USER_ID, SESSION_ID)


if __name__ == "__main__":
    asyncio.run(main())