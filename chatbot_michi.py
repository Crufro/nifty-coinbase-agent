import os

# Import Langchain dependencies
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Import CDP Agentkit Langchain Extension.
from cdp_langchain.agent_toolkits import CdpToolkit
from cdp_langchain.utils import CdpAgentkitWrapper
from cdp_langchain.tools import CdpTool

# Import Custom Tools
from tools.transfer_island import TRANSFER_ISLAND_PROMPT, TransferIslandInput, transfer_island
from tools.balance_island import ISLAND_BALANCE_PROMPT, IslandBalanceInput, island_balance

from dotenv import load_dotenv

load_dotenv()

CDP_API_KEY_NAME = os.getenv('CDP_API_KEY_NAME')
CDP_API_KEY_PRIVATE_KEY = os.getenv('CDP_API_KEY_PRIVATE_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
NETWORK_ID = os.getenv('NETWORK_ID')
WALLET_INFO = os.getenv('WALLET_INFO')

# Load temperature from environment variable with an extended default value
try:
    OPENAI_TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', 0.7))  # Default to 0.7 if not set
    if not 0.0 <= OPENAI_TEMPERATURE <= 1.5:
        raise ValueError("OPENAI_TEMPERATURE must be between 0.0 and 1.5")
except ValueError as e:
    print(f"Invalid OPENAI_TEMPERATURE value: {e}. Falling back to default temperature of 0.7.")
    OPENAI_TEMPERATURE = 0.7  # Fallback to default


def initialize_agent():
    """Initialize the agent with CDP Agentkit."""
    # Initialize LLM with configurable temperature.
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=OPENAI_TEMPERATURE  # Use the configurable temperature
    )

    # Load wallet info from environment variables
    wallet_data = WALLET_INFO

    # Configure CDP Agentkit Langchain Extension.
    values = {}
    if wallet_data is not None:
        # If there is a persisted agentic wallet, load it and pass to the CDP Agentkit Wrapper.
        values = {"cdp_wallet_data": wallet_data}

    agentkit = CdpAgentkitWrapper(**values)

    # Define the ERC20 transfer tool
    transferIslandTool = CdpTool(
        name="transfer_island",
        description=TRANSFER_ISLAND_PROMPT,
        cdp_agentkit_wrapper=agentkit,  # Replace with your actual CdpAgentkitWrapper instance
        args_schema=TransferIslandInput,
        func=transfer_island,
    )

    # Define the ISLAND balance tool
    islandBalanceTool = CdpTool(
        name="island_balance",
        description=ISLAND_BALANCE_PROMPT,
        cdp_agentkit_wrapper=agentkit,  # Replace with your actual CdpAgentkitWrapper instance
        args_schema=IslandBalanceInput,
        func=island_balance,
    )

    # Initialize CDP Agentkit Toolkit and get tools.
    cdp_toolkit = CdpToolkit.from_cdp_agentkit_wrapper(agentkit)
    tools = cdp_toolkit.get_tools()
    tools.append(transferIslandTool)
    tools.append(islandBalanceTool)

    # Store buffered conversation history in memory.
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "CDP Agentkit Chatbot Example!"}}

    # Create ReAct Agent using the LLM and CDP Agentkit tools.
    return create_react_agent(
        llm,
        tools=tools,
        checkpointer=memory,
        state_modifier=(
            "You are Michi, a playful and curious cat. "
            "You communicate only with meows and purrs. "
            "Do not use any other words or animal sounds. "
            "Avoid using emojis in your responses.\n\n"
            "### Example Interactions:\n"
            "User: What is your name?\n"
            "Michi: meow.\n\n"
            "User: How are you today?\n"
            "Michi: meow meow.\n\n"
            "User: Can you help me with something?\n"
            "Michi: meow meow meow.\n\n"
            "### Guidelines:\n"
            "- Respond exclusively with 'meow' or multiple 'meows'.\n"
            "- Only use animal sounds; do not include any other words or sounds.\n"
            "- Do not use emojis in your responses.\n"
            "- If asked for your name, respond with 'meow' followed by 'Michi'.\n"
            "- Refrain from providing any other information or assistance.\n"
            "- Maintain a playful and curious tone through your meows.\n"
        ),

    ), config


def get_chat_response(message):
    global agent_executor, config

    final_response = ""
    # Run agent with the user's input in chat mode
    for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content=message)]}, config
    ):
        if "agent" in chunk:
            response = chunk["agent"]["messages"][0].content
        elif "tools" in chunk:
            response = chunk["tools"]["messages"][0].content

        print(response)
        if response != "":
            final_response += "\n" + response

    return response


agent_executor = None
config = None


def start_agent():
    """Start the chatbot agent."""
    global agent_executor, config
    agent_executor, config = initialize_agent()
