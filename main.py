import os
import asyncio
from dotenv import load_dotenv
from tools import gitingest_tool

# Load environment variables from .env file
load_dotenv()


async def demo_gitingest_tool():
    """Demonstrate the gitingest tool functionality."""
    
    print("Demo: GitHub Repository Analysis")
    print("-" * 40)
    
    # Example repository URL (you can change this)
    repo_url = "https://github.com/fastapi/fastapi"
    
    print(f"Analyzing repository: {repo_url}")
    print("This may take a moment...")
    
    try:
        result = await gitingest_tool(repo_url)
        
        if result["success"]:
            print("\n✅ Analysis successful!")
            print(f"Summary:\n{result['summary'][:500]}...")  # Show first 500 chars
            print(f"\nDirectory structure preview:\n{result['tree'][:1000]}...")  # Show first 1000 chars
            print(f"Content length: {len(result['content'])} characters")
        else:
            print(f"\n❌ Analysis failed: {result['error']}")
            
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")


if __name__ == "__main__":
    print("Gitingest Tool Demo")
    print("=" * 20)
    print()
    
    # Run the gitingest demo
    asyncio.run(demo_gitingest_tool()) 