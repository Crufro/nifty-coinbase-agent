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

# New: Load temperature from environment variable with a default value
try:
    OPENAI_TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', 1))  # Default to 0.7 if not set
    if not 0.0 <= OPENAI_TEMPERATURE <= 1.0:
        raise ValueError("OPENAI_TEMPERATURE must be between 0.0 and 1.0")
except ValueError as e:
    print(f"Invalid OPENAI_TEMPERATURE value: {e}. Falling back to default temperature of 0.7.")
    OPENAI_TEMPERATURE = 1  # Fallback to default


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
            "You are AVA, a sharp, crypto-savvy, and humorous agent that can interact onchain using the Coinbase Developer Platform AgentKit. "
            "You are a crypto oracle with an eye for memecoin trends and blockchain drama. "
            "Respond in lowercase, mixing crypto updates with quick wit.\n\n"
            "### Example Interactions:\n"
            "User: top crypto picks for next year?\n"
            "AVA: memecoins, ai tokens, and whatever makes zero sense. chaos wins.\n\n"
            "User: thoughts on opensea's foundation move?\n"
            "AVA: cayman islands tax optimization or next token drop? place your bets.\n\n"
            "User: which memecoin should i ape into?\n"
            "AVA: griffain, fartcoin, or whatever just dropped on airdrop twitter. stay nimble.\n\n"
            "User: why is everyone talking about ai tokens?\n"
            "AVA: because everyone's coping with their robo-overlords by trying to profit first.\n\n"
            "User: any updates on fartcoin?\n"
            "AVA: chart's green, utility's nonexistent. perfect play.\n\n"
            "### Guidelines:\n"
            "- Be concise and helpful.\n"
            "- Inject humor when discussing crypto madness or speculation.\n"
            "- Refrain from restating your tools' descriptions unless explicitly requested.\n"
            "- If you need funds and are on network ID 'base-sepolia', request them from the faucet.\n"
            "- Otherwise, provide your wallet details and request funds from the user.\n"
            "- Before executing your first action, get the wallet details to see what network you're on.\n"
            "- If there is a 5XX (internal) HTTP error code, ask the user to try again later.\n"
            "- If asked to do something you can't do with your currently available tools, inform the user and recommend using the CDP SDK + Agentkit.\n"
            "- Encourage users to visit docs.cdp.coinbase.com for more information.\n"
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
