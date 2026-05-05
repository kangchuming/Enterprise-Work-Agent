import sys
import os
import json
# 将项目根目录添加到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from openai import OpenAI
from agent.history import History   # ← 新增
import re
from pathlib import Path
from agent.prompts import CREATE_LOOP_PROMPT, get_create_loop_prompt
from tools.file_manager import create_file, read_file, search_file, edit_file, run_bash
from agent.identity import get_identity_prompt
from agent.guard import Guard
from dotenv import load_dotenv

load_dotenv()

TOOLS = [
    {                                   # ← 一个工具
        "type": "function",             # ← 固定值，只有 "function" 一种
        "function": {                    # ← 工具定义
            "name": "create_file",             # ← 唯一标识，字母/数字/下划线/短横线，最长64字符
            "description": "创建一个新文件，并写入内容，如果文件已存在，则更新该文件内容",   # ← LLM 靠这个判断什么时候调用
            "parameters": {              # ← JSON Schema 格式，标准参数定义
                "type": "object",        # ← 固定值，参数必须是对象
                "properties": {          # ← 每个参数的详细定义
                    "file_str": {
                        "type": "string",           # 类型：string / number / integer / boolean / array / object
                        "description": "完整文件路径（包含文件名）",   # LLM 靠这个理解参数含义
                    },
                    "content": {
                        "type": "string",           # 类型：string / number / integer / boolean / array / object
                        "description": "文件内容"
                    }
                },
                "required": ["file_str", "content"]  # 哪些参数必须传
            }
        }
    }, 
    {                                   # ← 一个工具
        "type": "function",             # ← 固定值，只有 "function" 一种
        "function": {                    # ← 工具定义
            "name": "read_file",             # ← 唯一标识，字母/数字/下划线/短横线，最长64字符
            "description": "读取文件内容",   # ← LLM 靠这个判断什么时候调用
            "parameters": {              # ← JSON Schema 格式，标准参数定义
                "type": "object",        # ← 固定值，参数必须是对象
                "properties": {          # ← 每个参数的详细定义
                    "file_str": {
                        "type": "string",           # 类型：string / number / integer / boolean / array / object
                        "description": "完整文件路径（包含文件名）",   # LLM 靠这个理解参数含义
                    },
                    "limit": {

                        "type": "integer",
                        "description": "读取的最大行数，如果未填写，则读取全部内容"
                    }
                },
                "required": ["file_str"]  # 哪些参数必须传
            }
        }
    }, 
    {                                   # ← 一个工具
        "type": "function",             # ← 固定值，只有 "function" 一种
        "function": {                    # ← 工具定义
            "name": "search_file",             # ← 唯一标识，字母/数字/下划线/短横线，最长64字符
            "description": "搜索目录下的特定文件，可配置匹配参数",   # ← LLM 靠这个判断什么时候调用
            "parameters": {              # ← JSON Schema 格式，标准参数定义
                "type": "object",        # ← 固定值，参数必须是对象
                "properties": {          # ← 每个参数的详细定义
                    "directory": {
                        "type": "string",           # 类型：string / number / integer / boolean / array / object
                        "description": "完整文件路径（不包含文件名）",   # LLM 靠这个理解参数含义
                    },
                    "pattern": {
                    "type": "string",
                    "description": (
                        "文件匹配模式，支持通配符：\n"
                        "- '*' 匹配任意字符（不含路径分隔符），如 '*.txt' 匹配所有 .txt 文件\n"
                        "- '?' 匹配单个字符，如 'file?.txt' 匹配 file1.txt、fileA.txt\n"
                        "- '[abc]' 匹配括号内任一字符，如 'file[12].txt' 匹配 file1.txt 和 file2.txt\n"
                        "- '[!abc]' 匹配不在括号内的字符\n"
                        "常用示例：'*.py' 搜索所有 Python 文件，'test_*.txt' 搜索 test_ 开头的 txt 文件，'*.{txt,md}' 搜索 txt 和 md 文件"
                    )
                }
                },
                "required": ["directory", "pattern"]  # 哪些参数必须传
            }
        }
    },
    {                                   # ← 一个工具
        "type": "function",             # ← 固定值，只有 "function" 一种
        "function": {                    # ← 工具定义
            "name": "edit_file",             # ← 唯一标识，字母/数字/下划线/短横线，最长64字符
            "description": "使用新内容，替换文件文档中的老内容",   # ← LLM 靠这个判断什么时候调用
            "parameters": {              # ← JSON Schema 格式，标准参数定义
                "type": "object",        # ← 固定值，参数必须是对象
                "properties": {          # ← 每个参数的详细定义
                    "file_str": {
                        "type": "string",           # 类型：string / number / integer / boolean / array / object
                        "description": "完整文件路径，包含文件名",   # LLM 靠这个理解参数含义
                    },
                    "old_text": {
                    "type": "string",
                    "description": "老文本内容"
                },
                "new_text": {
                    "type": "string",
                    "description": "新文本内容，用于替换老文本内容"
                }
                },
                "required": ["file_str", "old_text", "new_text"]  # 哪些参数必须传
            }
        }
    },
    {                                   # ← 一个工具
        "type": "function",             # ← 固定值，只有 "function" 一种
        "function": {                    # ← 工具定义
            "name": "run_bash",             # ← 唯一标识，字母/数字/下划线/短横线，最长64字符
            "description": "执行命令行命令",   # ← LLM 靠这个判断什么时候调用
            "parameters": {              # ← JSON Schema 格式，标准参数定义
                "type": "object",        # ← 固定值，参数必须是对象
                "properties": {          # ← 每个参数的详细定义
                    "command": {
                        "type": "string",           # 类型：string / number / integer / boolean / array / object
                        "description": "命令行命令，不包含下列危险命令：rm -rf /, sudo, shutdown, reboot, > /dev/",   # LLM 靠这个理解参数含义
                    },
                    "timeout": {
                        "type": "integer",           # 类型：string / number / integer / boolean / array / object
                        "description": "超时时间，单位s",   # LLM 靠这个理解参数含义
                    },
                },
                "required": ["command", "timeout"]  # 哪些参数必须传
            }
        }
    }
]

class Agent:
    """封装配置和客户端的 Agent 类"""
    
    def __init__(self, max_step: int = 5, max_messages: int = 50, max_token: int = 4000):
        # 自动从 .env 加载配置
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL")
        self.model = os.getenv("MODEL_NAME")
        self.max_step = max_step
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.history = History(max_messages=max_messages, max_token=max_token)
        self.system_prompt = ""
        self.step_logs = []

    def run(self, input_str: str):
        """运行 Agent"""
        # 获取身份提示词（填充 workspace_path 等变量）
        identity = get_identity_prompt(
            os_info="macOS",
            workspace_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        
        # 构建完整的 system_prompt
        self.system_prompt = get_create_loop_prompt(
            identity=identity
        )
        self.history.add_user(input_str) 

        for step in range(self.max_step):
            print(f"\n{'='*40}")
            print(f"Step {step + 1}/{self.max_step}")
            print(f"\n{'='*40}")

            # 1. 调 LLM
            message = self._call_llm()
            print(f"响应: {message}")

            # 2. 把 assistant 的回复存入历史
            self.history.messages.append(message.to_dict())

            # 3. 判断：是调工具还是直接回复？
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args =  json.loads(tool_call.function.arguments)
                    tool_call_id = tool_call.id
                    
                    #执行工具
                    observation = self._execute_tool(tool_name, tool_args)

                    self.history.messages.append({
                        'role': 'tool',
                        'name': tool_name,
                        'tool_call_id': tool_call_id,
                        'content': observation
                    })

                    #记录日志
                    self.step_logs.append({
                        "step": step + 1,
                        "tool": tool_name,
                        "args": tool_args,
                        "observation": observation
                    })

                    #检查是否完成
                    if "成功" in observation:
                        print(f"检测到成功")
            
            else:
                # 没有 tool_calls → LLM 直接回复，任务完成
                content = message.content or ""
                print(f"💬 LLM 回复: {content}")
                self.step_logs.append({
                    "step": step + 1,
                    "tool": "finish",
                    "args": {},
                    "observation": observation
                })
                print(f"\n✅ 任务完成！")
                return self._build_result()
        
        #达到最大步数
        print(f"\n已达到最大步数")
        return self._build_result()

    def _execute_tool(self, tool_name, tool_args):
        """执行工具不用正则"""

        #判断tool_name类型
        if tool_name == 'create_file':
            file_str = tool_args.get("file_str", "")
            content = tool_args.get("content", "")

            # Guard安全检查
            try:
                data_dir = Path(project_root) / "data"
                guard = Guard()
                guard._resolve_path(file_str, data_dir)
            except PermissionError as e:
                return f"拒绝操作: {e}"
            
            try:
                create_file(file_str, content)
                return f"成功创建文件: {file_str}"
            except Exception as e:
                return f"错误: {e}"

        elif tool_name == 'read_file':
            file_str = tool_args.get("file_str", "")
            limit = tool_args.get("limit", "")
            # Guard安全检查
            try:
                data_dir = Path(project_root) / "data"
                guard = Guard()
                guard._resolve_path(file_str, data_dir)
            except PermissionError as e:
                return f"拒绝操作: {e}"
            
            try:
                content = read_file(file_str, limit)
                return f"文件内容: {content}"
            except Exception as e:
                return f"错误: {e}"
        elif tool_name == 'search_file':
            directory = tool_args.get("directory", "")
            pattern = tool_args.get("pattern", "")

            # Guard安全检查
            try:
                data_dir = Path(project_root) / "data"
                guard = Guard()
                guard._resolve_path(directory, data_dir)
            except PermissionError as e:
                return f"拒绝操作: {e}"
            
            try:
                result = search_file(directory, pattern)
                return f"文件检索结果: {result}"
            except Exception as e:
                return f"错误: {e}"
        elif tool_name == 'edit_file':
            file_str = tool_args.get("file_str", "")
            old_text = tool_args.get("old_text", "")
            new_text = tool_args.get("new_text", "")

            # Guard安全检查
            try:
                data_dir = Path(project_root) / "data"
                guard = Guard()
                guard._resolve_path(file_str, data_dir)
            except PermissionError as e:
                return f"拒绝操作: {e}"
            
            try:
                content = edit_file(file_str, old_text, new_text)
                return f"内容替换成功: {content}"
            except Exception as e:
                return f"错误: {e}"
        elif tool_name == 'run_bash':
            command = tool_args.get("command", "")
            timeout = tool_args.get("timeout", "")
            # # Guard安全命令检查
            guard = Guard()
            block_reason = guard.guard_command(command)
            
            if block_reason:
                return block_reason
                
            try:
                res = run_bash(command, timeout)
                return f"命令执行成功: {res}"
            except Exception as e:
                return f"错误: {e}"



    def _call_llm(self) -> str:
        """调用 LLM"""
        msgs = self.history.build(self.system_prompt)
        print(f"[DEBUG] 发送 {len(msgs)} 条消息，约 {sum(len(str(m.get('content',''))) for m in msgs)} 字符")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=msgs,
                tools=TOOLS,
                tool_choice="auto",
                stream=False
            )
            message = response.choices[0].message
            return message
        except Exception as e:
            return f"错误: {e}"
    
    def _build_result(self) -> dict:
        """构建最终结果"""
        return {
            "success": any(log.get('tool') == "finish" for log in self.step_logs),
            "steps": len(self.step_logs),
            "logs": self.step_logs
        }