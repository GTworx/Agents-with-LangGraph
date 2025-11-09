"""
Cooking Agent with LangGraph, PostgreSQL Memory, and Gemini LLM
"""

import os
from typing import Annotated, TypedDict, Sequence
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
import psycopg


# Define the state structure
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    user_preferences: Annotated[dict, "User cooking preferences and dietary restrictions"]


# Initialize Gemini LLM
def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Please set GOOGLE_API_KEY environment variable")
    
    return ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        temperature=0.7,
        google_api_key=api_key
    )


# System prompt for cooking specialization
COOKING_SYSTEM_PROMPT = """You are a specialized cooking assistant with expertise in:
- Recipe recommendations based on ingredients, cuisine types, and dietary restrictions
- Cooking techniques and tips
- Ingredient substitutions
- Meal planning and preparation
- Food safety and storage
- Dietary accommodations (vegetarian, vegan, gluten-free, etc.)

Always provide helpful, accurate cooking advice. If you remember user preferences from previous conversations, use them to personalize your responses.
Be concise but thorough in your explanations."""


# Node functions
def cooking_agent_node(state: AgentState) -> AgentState:
    """Main agent node that processes messages and generates responses"""
    llm = get_llm()
    
    # Build context from previous messages
    messages = [SystemMessage(content=COOKING_SYSTEM_PROMPT)]
    
    # Add user preferences context if available
    if state.get("user_preferences"):
        prefs = state["user_preferences"]
        pref_str = ", ".join([f"{k}: {v}" for k, v in prefs.items()])
        messages.append(SystemMessage(content=f"User preferences: {pref_str}"))
    
    # Add conversation history
    messages.extend(state["messages"])
    
    # Get response from LLM
    response = llm.invoke(messages)
    
    # Update state with new message
    return {
        "messages": [response],
        "user_preferences": state.get("user_preferences", {})
    }


def extract_preferences_node(state: AgentState) -> AgentState:
    """Extract and update user preferences from the conversation"""
    llm = get_llm()
    
    # Get the last user message
    last_message = state["messages"][-1] if state["messages"] else None
    
    if last_message and isinstance(last_message, HumanMessage):
        # Ask LLM to extract preferences
        extraction_prompt = f"""Analyze this message and extract any cooking preferences, dietary restrictions, 
        or food-related information: "{last_message.content}"
        
        Return ONLY a JSON object with keys like: dietary_restrictions, favorite_cuisine, allergies, skill_level, etc.
        If no preferences found, return empty JSON object {{}}.
        """
        
        try:
            result = llm.invoke([HumanMessage(content=extraction_prompt)])
            # Simple parsing - in production, use structured output
            content = result.content.strip()
            if content.startswith("{") and content.endswith("}"):
                import json
                new_prefs = json.loads(content)
                current_prefs = state.get("user_preferences", {})
                current_prefs.update(new_prefs)
                return {"user_preferences": current_prefs}
        except:
            pass
    
    return {"user_preferences": state.get("user_preferences", {})}


# Build the graph
def create_cooking_agent_graph():
    """Create the LangGraph workflow"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("extract_preferences", extract_preferences_node)
    workflow.add_node("cooking_agent", cooking_agent_node)
    
    # Define edges
    workflow.set_entry_point("extract_preferences")
    workflow.add_edge("extract_preferences", "cooking_agent")
    workflow.add_edge("cooking_agent", END)
    
    return workflow


def get_postgres_connection_string():
    """Build PostgreSQL connection string from environment variables"""
    host = os.getenv("POSTGRES_HOST", "postgres")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB", "cooking_agent")
    user = os.getenv("POSTGRES_USER", "agent")
    password = os.getenv("POSTGRES_PASSWORD", "")
    
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def main():
    """Main function to run the cooking agent"""
    print("🍳 Cooking Agent with LangGraph 🍳")
    print("=" * 50)
    
    # Get PostgreSQL connection string
    connection_string = get_postgres_connection_string()
    
    print(f"Connecting to PostgreSQL...")
    
    try:
        # Create PostgreSQL connection
        conn = psycopg.connect(connection_string, autocommit=True)
        
        # Create checkpointer with PostgreSQL
        memory = PostgresSaver(conn)
        
        # Setup the database tables
        memory.setup()
        
        print(f"✓ Connected to PostgreSQL")
        
        # Create and compile the graph
        workflow = create_cooking_agent_graph()
        app = workflow.compile(checkpointer=memory)
        
        # Configuration for thread-based memory
        thread_id = "cooking_session_1"
        config = {"configurable": {"thread_id": thread_id}}
        
        print(f"Thread ID: {thread_id}")
        print("\nType 'quit' to exit\n")
        
        # Conversation loop
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Happy cooking!")
                break
            
            if not user_input:
                continue
            
            # Create input state
            input_state = {
                "messages": [HumanMessage(content=user_input)],
                "user_preferences": {}
            }
            
            # Run the graph
            try:
                result = app.invoke(input_state, config)
                
                # Get the AI response
                if result["messages"]:
                    ai_message = result["messages"][-1]
                    print(f"\n🤖 Chef: {ai_message.content}\n")
                    
                    # Show preferences if any were extracted
                    if result.get("user_preferences"):
                        print(f"📝 Preferences noted: {result['user_preferences']}\n")
            
            except Exception as e:
                print(f"❌ Error: {e}\n")
        
        # Close connection
        conn.close()
        
    except Exception as e:
        print(f"❌ Failed to connect to PostgreSQL: {e}")
        print("Please make sure PostgreSQL is running and credentials are correct.")


if __name__ == "__main__":
    main()
