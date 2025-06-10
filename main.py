import os
import asyncio
from dotenv import load_dotenv
from tools import gitingest_tool, clone_repo_tool, create_container_tool

# Load environment variables from .env file
load_dotenv()


async def demo_combined_workflow():
    """Demonstrate the combined workflow: clone repository, analyze with gitingest, then generate Dockerfile."""
    
    print("Combined Demo: Clone Repository + Gitingest Analysis + Dockerfile Generation")
    print("=" * 75)
    
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
        
        if not ingest_result["success"]:
            print(f"\n❌ Analysis failed: {ingest_result['error']}")
            return
            
        print("\n✅ Analysis successful!")
        print(f"Summary:\n{ingest_result['summary'][:500]}...")  # Show first 500 chars
        print(f"\nDirectory structure preview:\n{ingest_result['tree'][:1000]}...")  # Show first 1000 chars
        print(f"Content length: {len(ingest_result['content'])} characters")
        
        # Generate Dockerfile based on the analysis
        print(f"\nStep 3: Generating Dockerfile based on repository analysis...")
        print("This may take a moment...")
        
        container_result = await create_container_tool(
            gitingest_summary=ingest_result['summary'],
            gitingest_tree=ingest_result['tree'],
            gitingest_content=ingest_result['content'],
            project_name=clone_result['repo_name']
        )
        
        if container_result["success"]:
            print("\n✅ Dockerfile generation successful!")
            print(f"Project: {container_result['project_name']}")
            print(f"Technology Stack: {container_result['technology_stack']}")
            print(f"Port Recommendations: {container_result['port_recommendations']}")
            print(f"Base Image Reasoning: {container_result['base_image_reasoning']}")
            
            if container_result['context_truncated']:
                print(f"\n⚠️  Content was truncated: {container_result['used_content_length']}/{container_result['original_content_length']} characters used")
            
            print(f"\nGenerated Dockerfile:\n{'-' * 40}")
            print(container_result['dockerfile'])
            print('-' * 40)
            
            if container_result['docker_compose_suggestion']:
                print(f"\nSuggested docker-compose.yml:\n{'-' * 40}")
                print(container_result['docker_compose_suggestion'])
                print('-' * 40)
            
            if container_result['additional_notes']:
                print(f"\nAdditional Notes:")
                print(container_result['additional_notes'])
        else:
            print(f"\n❌ Dockerfile generation failed: {container_result['error']}")
            
    except Exception as e:
        print(f"\n❌ Error during workflow: {e}")


if __name__ == "__main__":
    print("GitHub Tools Demo")
    print("=" * 20)
    print()
    
    # Run the combined workflow demo
    asyncio.run(demo_combined_workflow()) 