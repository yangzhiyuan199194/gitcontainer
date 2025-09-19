"""
Gitingest prompts for Gitcontainer application.
"""


def create_dockerfile_selection_prompt(
    dockerfile_paths: list,
    summary: str,
    tree: str,
    content: str,
    max_context_tokens: int = 128000
) -> str:
    """
    创建用于选择最佳 Dockerfile 的 prompt
    
    Args:
        dockerfile_paths: Dockerfile 路径列表
        summary: 项目摘要
        tree: 目录结构
        content: 源码内容
        max_context_tokens: 最大上下文 token 数
        
    Returns:
        构建好的 prompt 字符串
    """
    # 构建 Dockerfile 信息列表（仅路径）
    dockerfile_list = []
    for path in dockerfile_paths:
        dockerfile_list.append(f"PATH: {path}")
    
    dockerfiles_section = "\n".join(dockerfile_list)
    
    # 构建基础 prompt（不包含具体内容）
    base_prompt = f"""Based on the repository analysis, select the most appropriate Dockerfile for containerizing this project.

PROJECT SUMMARY:
{summary}

DIRECTORY STRUCTURE:
{tree}

AVAILABLE DOCKERFILES:
{dockerfiles_section}

Please analyze the Dockerfiles and the project context, then select the most appropriate Dockerfile for this project. Consider:
1. Which Dockerfile best matches the project's technology stack
2. Which Dockerfile follows the best practices
3. Which Dockerfile is most likely to successfully build the project
4. Which Dockerfile is the most complete and production-ready

Respond ONLY with the file path of the selected Dockerfile. Do not include any explanations or additional text."""

    # 估算 prompt 长度（英文字符大约 4:1 与 token 的比例）
    prompt_tokens = len(base_prompt) // 4
    max_tokens = 1000  # LLM 返回的最大 token 数
    available_tokens = max_context_tokens - prompt_tokens - max_tokens
    
    # 计算所有 Dockerfile 内容的 token 数
    total_dockerfile_tokens = 0
    for path in dockerfile_paths:
        # 这里我们假设每个路径的长度代表其内容长度，实际应用中应计算实际内容
        total_dockerfile_tokens += len(path) // 4  # 简单估算 token 数
    
    # 计算源代码内容的 token 数
    source_content_tokens = len(content) // 4
    
    # 总需要的 token 数
    total_needed_tokens = prompt_tokens + total_dockerfile_tokens + source_content_tokens + max_tokens
    
    # 如果 token 数在限制范围内，则添加具体 Dockerfile 内容和源代码内容
    detailed_dockerfiles_section = ""
    source_code_section = ""
    if total_needed_tokens <= max_context_tokens:
        dockerfile_details = []
        for path in dockerfile_paths:
            dockerfile_details.append(f"PATH: {path}\nCONTENT:\n[Content would be here]\n" + "="*50)
        detailed_dockerfiles_section = "\n\n".join(dockerfile_details)
        
        # 添加源代码内容
        source_code_section = f"SOURCE CODE CONTEXT:\n{content[:10000]}\n"  # 使用更长的上下文
        
        # 更新 prompt，包含详细内容
        base_prompt = f"""Based on the repository analysis, select the most appropriate Dockerfile for containerizing this project.

PROJECT SUMMARY:
{summary}

DIRECTORY STRUCTURE:
{tree}

{source_code_section}
AVAILABLE DOCKERFILES WITH CONTENT:
{detailed_dockerfiles_section}

Please analyze the Dockerfiles and the project context, then select the most appropriate Dockerfile for this project. Consider:
1. Which Dockerfile best matches the project's technology stack
2. Which Dockerfile follows the best practices
3. Which Dockerfile is most likely to successfully build the project
4. Which Dockerfile is the most complete and production-ready

Respond ONLY with the file path of the selected Dockerfile. Do not include any explanations or additional text."""
    else:
        # 如果超出 token 限制，添加提示信息
        source_code_section = f"SOURCE CODE CONTEXT (first 1000 chars):\n{content[:1000]}\n"  # 使用较短的上下文
        base_prompt += "\n\nNote: Due to context length limitations, the specific content of Dockerfiles is not provided. Please make your selection based on the file paths and project context."

    return base_prompt