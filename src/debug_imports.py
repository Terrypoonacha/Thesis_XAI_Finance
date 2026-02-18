try:
    from langchain.agents import Tool, initialize_agent, AgentType
    print("Import successful!")
except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"Other error: {e}")

import langchain
print(f"LangChain version: {langchain.__version__}")
