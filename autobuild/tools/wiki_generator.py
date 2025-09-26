"""
Wiki generation tool for autobuild application.

This module provides functionality to analyze a repository and generate wiki documentation.
"""

import asyncio
import logging
import os
import re
import html
from typing import Dict, Any, Optional, List
from xml.etree import ElementTree as ET

from autobuild.services.llm_client import LLMClient
from autobuild.utils import get_websocket_manager

logger = logging.getLogger(__name__)


async def analyze_repo_structure(local_repo_path: str) -> Dict[str, Any]:
    """
    Analyze repository structure to determine wiki pages.
    
    Args:
        local_repo_path (str): Path to the local repository
        
    Returns:
        Dict containing file tree and README content
    """
    file_tree_lines = []
    readme_content = ""

    for root, dirs, files in os.walk(local_repo_path):
        # Exclude hidden dirs/files and virtual envs
        dirs[:] = [d for d in dirs if
                   not d.startswith('.') and d != '__pycache__' and d != 'node_modules' and d != '.venv']
        for file in files:
            if file.startswith('.') or file == '__init__.py' or file == '.DS_Store':
                continue
            rel_dir = os.path.relpath(root, local_repo_path)
            rel_file = os.path.join(rel_dir, file) if rel_dir != '.' else file
            file_tree_lines.append(rel_file)
            # Find README.md (case-insensitive)
            if file.lower() == 'readme.md' and not readme_content:
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        readme_content = f.read()
                except Exception as e:
                    logger.warning(f"Could not read README.md: {str(e)}")
                    readme_content = ""

    file_tree_str = '\n'.join(sorted(file_tree_lines))
    return {"file_tree": file_tree_str, "readme": readme_content}


async def determine_wiki_structure(file_tree: str, readme_content: str, owner: str, repo: str, model: str,
                                   ws_manager: Optional[Any] = None) -> Dict[str, Any]:
    """
    Determine wiki structure based on repository content.
    
    Args:
        file_tree (str): Repository file tree
        readme_content (str): README content
        owner (str): Repository owner
        repo (str): Repository name
        ws_manager (Optional[Any]): WebSocket manager for streaming output
        
    Returns:
        Dict containing wiki structure
    """
    try:
        llm_client = LLMClient()

        # Create prompt for determining wiki structure
        prompt = f"""
Analyze the following repository structure and README to determine an appropriate wiki structure.

Repository Owner: {owner}
Repository Name: {repo}

File Tree:
{file_tree}

README Content:
{readme_content[:2000]}  # Limit README length

Based on this information, create a structured wiki with:
1. A main title for the wiki
2. A brief description
3. A list of wiki pages with titles and file paths they should cover
4. Group related pages into sections if applicable

Respond ONLY with a valid XML structure like this:
<wiki>
    <title>Repository Wiki Title</title>
    <description>Brief description of the repository</description>
    <pages>
        <page>
            <id>page1</id>
            <title>Page Title</title>
            <filePaths>
                <path>file1.py</path>
                <path>file2.py</path>
            </filePaths>
            <importance>high|medium|low</importance>
        </page>
        <!-- More pages -->
    </pages>
    <sections>
        <section>
            <id>section1</id>
            <title>Section Title</title>
            <pages>
                <pageId>page1</pageId>
                <pageId>page2</pageId>
            </pages>
        </section>
        <!-- More sections -->
    </sections>
</wiki>
"""

        messages = [
            {
                "role": "system",
                "content": "You are an expert technical documentation writer. Analyze repository structures and create comprehensive wiki documentation plans."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        if ws_manager:
            await ws_manager.send_chunk("🧠 Analyzing repository structure for wiki generation...\n")

        result = await llm_client.call_llm(
            model=model,
            messages=messages,
            stream=True
        )

        if result["success"]:
            # Parse XML response
            try:
                # Clean the response content to ensure it's valid XML
                content = result["content"].strip()

                # Remove any markdown code block markers if present
                if content.startswith("```xml"):
                    content = content[6:]  # Remove ```xml
                if content.startswith("```"):
                    content = content[3:]  # Remove ```
                if content.endswith("```"):
                    content = content[:-3]  # Remove ```

                # Remove any text before the first tag
                first_tag_match = re.search(r'<\w+', content)
                if first_tag_match:
                    content = content[first_tag_match.start():]

                # Remove any text after the last tag
                last_tag_match = re.search(r'<\/\w+>\s*$', content)
                if last_tag_match:
                    content = content[:last_tag_match.end()]

                # Try to parse the XML as is first
                try:
                    root = ET.fromstring(content)
                except ET.ParseError:
                    # If parsing fails, try to fix common XML issues
                    # Handle unescaped ampersands in text content
                    content = re.sub(r'&(?![a-zA-Z]+;|#\d+;|#x[0-9a-fA-F]+;)', '&amp;', content)

                    # Try to parse again
                    root = ET.fromstring(content)

                # Extract title and description
                title = root.find("title").text if root.find("title") is not None else f"{owner}/{repo} Wiki"
                description = root.find("description").text if root.find(
                    "description") is not None else "Repository documentation"

                # Extract pages
                pages = []
                pages_elem = root.find("pages")
                if pages_elem is not None:
                    for page_elem in pages_elem.findall("page"):
                        page_id = page_elem.find("id").text if page_elem.find("id") is not None else "page"
                        page_title = page_elem.find("title").text if page_elem.find("title") is not None else "Untitled"

                        file_paths = []
                        file_paths_elem = page_elem.find("filePaths")
                        if file_paths_elem is not None:
                            for path_elem in file_paths_elem.findall("path"):
                                file_paths.append(path_elem.text)

                        importance = page_elem.find("importance").text if page_elem.find(
                            "importance") is not None else "medium"

                        pages.append({
                            "id": page_id,
                            "title": page_title,
                            "filePaths": file_paths,
                            "importance": importance,
                            "relatedPages": []
                        })

                # Extract sections
                sections = []
                root_sections = []
                sections_elem = root.find("sections")
                if sections_elem is not None:
                    for section_elem in sections_elem.findall("section"):
                        section_id = section_elem.find("id").text if section_elem.find("id") is not None else "section"
                        section_title = section_elem.find("title").text if section_elem.find(
                            "title") is not None else "Section"

                        page_ids = []
                        pages_elem = section_elem.find("pages")
                        if pages_elem is not None:
                            for page_id_elem in pages_elem.findall("pageId"):
                                page_ids.append(page_id_elem.text)

                        sections.append({
                            "id": section_id,
                            "title": section_title,
                            "pages": page_ids
                        })
                        root_sections.append(section_id)

                wiki_structure = {
                    "id": "wiki",
                    "title": title,
                    "description": description,
                    "pages": pages,
                    "sections": sections,
                    "rootSections": root_sections
                }

                if ws_manager:
                    await ws_manager.send_chunk(f"✅ Wiki structure determined with {len(pages)} pages\n")

                return wiki_structure
            except ET.ParseError as e:
                error_msg = f"Failed to parse XML response: {str(e)}"
                logger.error(error_msg)
                # Log the problematic content for debugging
                logger.error(f"Problematic XML content: {result['content'][:500]}...")
                if ws_manager:
                    await ws_manager.send_chunk(f"❌ {error_msg}\n")
                return {"error": error_msg}
        else:
            error_msg = f"Failed to determine wiki structure: {result.get('error', 'Unknown error')}"
            logger.error(error_msg)
            if ws_manager:
                await ws_manager.send_chunk(f"❌ {error_msg}\n")
            return {"error": error_msg}

    except Exception as e:
        error_msg = f"Error determining wiki structure: {str(e)}"
        logger.error(error_msg)
        if ws_manager:
            await ws_manager.send_chunk(f"❌ {error_msg}\n")
        return {"error": error_msg}


async def generate_wiki_page_content(page_info: Dict[str, Any], file_tree: str, readme_content: str, owner: str,
                                     repo: str,
                                     local_repo_path: str, model: str, ws_manager: Optional[Any] = None) -> Dict[
    str, Any]:
    """
    Generate content for a wiki page based on repository files.
    
    Args:
        page_info (Dict): Information about the page to generate
        file_tree (str): Repository file tree
        readme_content (str): README content
        owner (str): Repository owner
        repo (str): Repository name
        local_repo_path (str): Local repository path
        ws_manager (Optional[Any]): WebSocket manager for streaming output
        
    Returns:
        Dict containing page content
    """
    try:
        llm_client = LLMClient()

        # Collect content from relevant files
        file_contents = []
        for file_path in page_info.get("filePaths", []):
            full_path = os.path.join(local_repo_path, file_path)
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Limit file content length to avoid context overflow
                        if len(content) > 5000:
                            content = content[:5000] + "\n\n... [Content truncated] ..."
                        file_contents.append(f"File: {file_path}\n{content}\n")
                except Exception as e:
                    logger.warning(f"Could not read file {file_path}: {str(e)}")

        # Create prompt for generating page content
        prompt = f"""
Generate a comprehensive wiki page for the following repository:

Repository Owner: {owner}
Repository Name: {repo}

Page Title: {page_info['title']}

README Content:
{readme_content[:1000]}  # Limit README length

File Tree:
{file_tree}

Relevant File Contents:
{''.join(file_contents)}

Please create a well-structured wiki page that:
1. Provides a clear explanation of what this part of the repository does
2. Documents key components, functions, or files
3. Explains how to use or configure this functionality if applicable
4. Mentions any important considerations or best practices
5. Uses markdown formatting for better readability

Do not include any XML tags or other formatting in your response, just the markdown content.
"""

        messages = [
            {
                "role": "system",
                "content": "You are an expert technical documentation writer creating clear, comprehensive wiki documentation for software repositories."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        if ws_manager:
            await ws_manager.send_chunk(f"📝 Generating content for '{page_info['title']}'...\n")

        result = await llm_client.call_llm(
            model=model,
            messages=messages,
            stream=True,
            ws_manager=ws_manager
        )

        if result["success"]:
            if ws_manager:
                await ws_manager.send_chunk(f"✅ Content generated for '{page_info['title']}'\n\n")
            return {
                "success": True,
                "content": result["content"],
                "title": page_info["title"],
                "id": page_info["id"]
            }
        else:
            error_msg = f"Failed to generate content for '{page_info['title']}': {result.get('error', 'Unknown error')}"
            logger.error(error_msg)
            if ws_manager:
                await ws_manager.send_chunk(f"❌ {error_msg}\n")
            return {"success": False, "error": error_msg}

    except Exception as e:
        error_msg = f"Error generating content for '{page_info['title']}': {str(e)}"
        logger.error(error_msg)
        if ws_manager:
            await ws_manager.send_chunk(f"❌ {error_msg}\n")
        return {"success": False, "error": error_msg}


async def wiki_generator_tool(local_repo_path: str, owner: str, repo: str, model: str,
                              ws_manager: Optional[Any] = None) -> Dict[str, Any]:
    """
    Main wiki generation tool that orchestrates the entire wiki generation process.
    
    Args:
        local_repo_path (str): Path to the local repository
        owner (str): Repository owner
        repo (str): Repository name
        ws_manager (Optional[Any]): WebSocket manager for streaming output
        
    Returns:
        Dict containing generated wiki structure and content
    """
    try:
        if ws_manager:
            await ws_manager.send_chunk("🔍 Starting wiki generation process...\n")

        # Analyze repository structure
        repo_data = await analyze_repo_structure(local_repo_path)
        file_tree = repo_data["file_tree"]
        readme_content = repo_data["readme"]

        # Determine wiki structure
        wiki_structure = await determine_wiki_structure(
            file_tree, readme_content, owner, repo, model, ws_manager
        )

        if "error" in wiki_structure:
            return {"success": False, "error": wiki_structure["error"]}

        # Generate content for each page
        generated_pages = {}
        pages = wiki_structure["pages"]

        if ws_manager:
            await ws_manager.send_chunk(f"📄 Generating content for {len(pages)} wiki pages...\n")

        for i, page in enumerate(pages):
            if ws_manager:
                await ws_manager.send_chunk(f"[{i + 1}/{len(pages)}] ")

            page_content = await generate_wiki_page_content(
                page, file_tree, readme_content, owner, repo, local_repo_path, model, ws_manager
            )

            if page_content["success"]:
                generated_pages[page["id"]] = {
                    "id": page["id"],
                    "title": page_content["title"],
                    "content": page_content["content"],
                    "filePaths": page["filePaths"],
                    "importance": page["importance"],
                    "relatedPages": page.get("relatedPages", [])
                }
            else:
                # If one page fails, we continue with others
                logger.warning(
                    f"Failed to generate content for page {page['id']}: {page_content.get('error', 'Unknown error')}")

        if ws_manager:
            await ws_manager.send_chunk("✅ Wiki generation process completed!\n")

        return {
            "success": True,
            "wiki_structure": wiki_structure,
            "generated_pages": generated_pages
        }

    except Exception as e:
        error_msg = f"Error during wiki generation: {str(e)}"
        logger.error(error_msg)
        if ws_manager:
            await ws_manager.send_chunk(f"❌ {error_msg}\n")
        return {"success": False, "error": error_msg}
