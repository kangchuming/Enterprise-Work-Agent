import os
import subprocess
from pathlib import Path

def create_file(filePath, content):
    """
    创建文件，自动创建父目录
    
    Args:
        file_path: 完整文件路径（包含文件名），如 ~/Documents/courage.txt
        content: 文件内容
    """
    # 是否存在目录
    file_path = os.path.expanduser(filePath)

    dir_path = os.path.dirname(filePath)


    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        print(f"创建目录: {dir_path}")
    # 是否存在文件
    try:
        with open(file_path , 'w', encoding='utf-8') as f:
            f.write(content)
        print("写入成功")
        return True
    except FileExistsError as e:
        print(f"写入失败: {e}")
        return False

def read_file(filePath: str, limit:  int = None):
    """
    读取文件
    
    Args:
        file_path: 完整文件路径（包含文件名），如 ~/Documents/courage.txt
    """
    file_path = os.path.expanduser(filePath)

    #读取文件
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            if limit and len(lines) > limit:
                return "".join(lines[:limit]) + f"\n... (还有 {len(lines) - limit} 行未显示)"
            return "".join(lines)
    except FileNotFoundError as e:
        print(f"文件不存在: {file_path} {e}")
        return f"文件不存在: {file_path}"
    except PermissionError as e:
        print(f"文件权限不允许读取 {e}")
        return f"文件权限不允许读取: {file_path}"

# ```
# 任务：给原项目加一个 search_file 工具

# 参数：directory (string), pattern (string)
# 功能：在目录下按 pattern 搜索文件

# 改动清单（不改 prompt）：
# □ TOOLS 加一项
# □ _execute_tool 加一个 elif
# □ 确认 prompt 不需要改（理由：TOOLS 已告诉 LLM）

# 跑通判断：输入"在 /data 目录搜索所有 .txt 文件"，LLM 调 search_file
def search_file(directory: str, pattern: str):
    """
    在目录下按 pattern 搜索文件
    
    Args:
        directory: 完整文件路径（不包含文件名），如 ~/Documents
        pattern: 匹配参数
    """
    dir_path = Path(directory).expanduser().resolve()
    result = []

    for f in dir_path.rglob(pattern):
        result.append(str(f))
    return result

# 修改文档内容
def edit_file(filePath: str, old_text: str, new_text: str) -> str:
    '''精确替换文件中的一段文本'''
    # 拿到目录
    file_path = Path(filePath).expanduser().resolve()

    #拿到原文本
    try:
        content = read_file(file_path)
        print("读取成功, {content}")
    except Exception as e:
        print("读取失败, {e}")
    
    newContent = content.replace(old_text, new_text, 1)

    #写入文档
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(newContent)
        print("写入成功")
        return(f"文本替换成功，{newContent}")
    except FileNotFoundError as e:
        print(f"文件不存在: {file_path} {e}")
        return f"文件不存在: {file_path}"
    except PermissionError as e:
        print(f"文件权限不允许读取 {e}")
        return f"文件权限不允许读取: {file_path}"

def run_bash(command: str, timeout: int):
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]

    if any(d in command for d in dangerous):
        return f"危险命令{command}"
    
    try:
        r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
        out = (r.stderr + r.stdout).strip()
        
        return out[:5000] if out else "not out"
    except subprocess.TimeoutExpired:
        return f"命令超时（{timeout}秒）: {command}"

def tavily_search(query: str, timeout: int, tavilyClient):
    try:
        res = tavilyClient.search(query=query, max_results=3, timeout = timeout, include_raw_content="markdown")
        results = res.get('results', '')
        maxCount = 500

        if not results:
            return "没有搜索到结果"

        trip = []
        for i, obj in enumerate(results, 1):
            raw_text = obj.get("raw_content", '')
            title = obj.get("title", "")
            text = raw_text
            
            if len(raw_text) > maxCount:
                text = raw_text[:maxCount] + "...\n"
            trip.append(f"{i} {title} \n {text}")
        return f'搜索结果：{"\n".join(trip)}'
    except Exception as e:
        print(e)
        return f"搜索报错：{e}"

