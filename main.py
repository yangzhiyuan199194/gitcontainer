import os
import asyncio
from dotenv import load_dotenv
from tools import gitingest_tool, clone_repo_tool

# Load environment variables from .env file
load_dotenv()


async def demo_gitingest_tool():
    """Demonstrate the gitingest tool functionality."""
    
    print("Demo 1: GitHub Repository Analysis (Gitingest)")
    print("-" * 50)
    
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


async def demo_git_operations():
    """Demonstrate the git operations tool functionality."""
    
    print("\n\nDemo 2: GitHub Repository Cloning (Git Operations)")
    print("-" * 50)
    
    # Example repository URL (smaller repo for demo)
    repo_url = "https://github.com/octocat/Hello-World"
    
    print(f"Cloning repository: {repo_url}")
    print("This may take a moment...")
    
    try:
        result = await clone_repo_tool(repo_url)
        
        if result["success"]:
            print("\n✅ Clone successful!")
            print(f"Repository: {result['repo_name']}")
            print(f"Local path: {result['local_path']}")
            print(f"Size: {result['repo_size_mb']} MB")
            print(f"Files: {result['file_count']} files")
            print(f"Message: {result['message']}")
        else:
            print(f"\n❌ Clone failed: {result['error']}")
            
    except Exception as e:
        print(f"\n❌ Error during clone: {e}")


if __name__ == "__main__":
    print("GitHub Tools Demo")
    print("=" * 20)
    print()
    
    async def run_demos():
        # Demo gitingest tool
        await demo_gitingest_tool()
        
        # Demo git operations tool
        await demo_git_operations()
    
    # Run all demos
    asyncio.run(run_demos()) 