import os
from dotenv import load_dotenv
from agents import Agent, Runner
# Load environment variables from .env file
load_dotenv()


def main():
    """Main function demonstrating OpenAI Agents SDK usage."""
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set your OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY=sk-...")
        return
    
    # Create a basic agent
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant that provides clear and concise answers."
    )
    
    # Example 1: Simple query
    print("Example 1: Simple query")
    print("-" * 40)
    result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
    print(f"Response: {result.final_output}")
    print()
    
    # Example 2: Interactive conversation
    print("Example 2: Interactive conversation")
    print("-" * 40)
    
    # You can ask multiple questions
    questions = [
        "What is the capital of France?",
        "Explain what Python is in one sentence.",
        "Give me a fun fact about computers."
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"Question {i}: {question}")
        result = Runner.run_sync(agent, question)
        print(f"Answer: {result.final_output}")
        print()


def create_specialized_agent():
    """Example of creating a specialized agent with specific instructions."""
    
    code_agent = Agent(
        name="CodeAssistant",
        instructions="""You are a Python programming expert. 
        Provide clear, well-commented code examples and explanations.
        Always include best practices and explain your reasoning."""
    )
    
    # Example usage of specialized agent
    query = "Show me how to read a CSV file in Python"
    result = Runner.run_sync(code_agent, query)
    print("Specialized Agent Example:")
    print("-" * 40)
    print(f"Query: {query}")
    print(f"Response: {result.final_output}")
    print()


if __name__ == "__main__":
    print("OpenAI Agents SDK Boilerplate")
    print("=" * 50)
    print()
    
    # Run main examples
    main()
    
    # Run specialized agent example
    create_specialized_agent()
    
    print("Done! Check out the OpenAI Agents SDK documentation for more features:")
    print("https://openai.github.io/openai-agents-python/") 