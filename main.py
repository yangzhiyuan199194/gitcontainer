import os
import asyncio
from dotenv import load_dotenv
from tools import gitingest_tool, clone_repo_tool

# Load environment variables from .env file
load_dotenv()


async def demo_combined_workflow():
    """Demonstrate the combined workflow: clone repository then analyze with gitingest."""
    
    print("Combined Demo: Clone Repository + Gitingest Analysis")
    print("=" * 55)
    
    # Example repository URL (smaller repo for demo)
    repo_url = "https://github.com/octocat/Hello-World"
    
    print(f"Step 1: Cloning repository: {repo_url}")
    print("This may take a moment...")
    
    try:
        # First, clone the repository
        clone_result = await clone_repo_tool(repo_url)
        
        if not clone_result["success"]:
            print(f"\n❌ Clone failed: {clone_result['error']}")
            return
            
        print("\n✅ Clone successful!")
        print(f"Repository: {clone_result['repo_name']}")
        print(f"Local path: {clone_result['local_path']}")
        print(f"Size: {clone_result['repo_size_mb']} MB")
        print(f"Files: {clone_result['file_count']} files")
        
        # Now analyze the cloned repository with gitingest
        print(f"\nStep 2: Analyzing cloned repository at: {clone_result['local_path']}")
        print("This may take a moment...")
        
        ingest_result = await gitingest_tool(clone_result['local_path'])
        
        if ingest_result["success"]:
            print("\n✅ Analysis successful!")
            print(f"Summary:\n{ingest_result['summary'][:500]}...")  # Show first 500 chars
            print(f"\nDirectory structure preview:\n{ingest_result['tree'][:1000]}...")  # Show first 1000 chars
            print(f"Content length: {len(ingest_result['content'])} characters")
        else:
            print(f"\n❌ Analysis failed: {ingest_result['error']}")
            
    except Exception as e:
        print(f"\n❌ Error during workflow: {e}")


if __name__ == "__main__":
    print("GitHub Tools Demo")
    print("=" * 20)
    print()
    
    # Run the combined workflow demo
    asyncio.run(demo_combined_workflow()) 