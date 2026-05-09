import sys
import os
import json
# 将项目根目录添加到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from openai import OpenAI
from agent.history import History   # ← 新增
import casbin
from dotenv import load_dotenv
import re
from pathlib import Path
from agent.prompts import CREATE_LOOP_PROMPT, get_create_loop_prompt
from tools.file_manager import create_file, read_file, search_file, edit_file, run_bash, tavily_search
from agent.identity import get_identity_prompt
from agent.guard import Guard
from agent.log_config import setup_logging, get_audit_logger
from tavily import TavilyClient
from e2b_code_interpreter import Sandbox

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
    },
    {                                   # ← 一个工具
        "type": "function",             # ← 固定值，只有 "function" 一种
        "function": {                    # ← 工具定义
            "name": "tavily_search",             # ← 唯一标识，字母/数字/下划线/短横线，最长64字符
            "description": "执行网络搜索，tavili",   # ← LLM 靠这个判断什么时候调用
            "parameters": {              # ← JSON Schema 格式，标准参数定义
                "type": "object",        # ← 固定值，参数必须是对象
                "properties": {          # ← 每个参数的详细定义
                    "query": {
                        "type": "string",           # 类型：string / number / integer / boolean / array / object
                        "description": "搜索的内容",   # LLM 靠这个理解参数含义
                    },
                    "timeout": {
                        "type": "integer",           # 类型：string / number / integer / boolean / array / object
                        "description": "超时时间，单位s",   # LLM 靠这个理解参数含义
                    },
                },
                "required": ["query", "timeout"]  # 哪些参数必须传
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "在云端沙箱中执行 Python 代码，返回输出",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "要执行的 Python 代码"
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "e2b_file",
            "description": "操作沙箱中的文件：读取、写入、列出目录",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["read", "write", "list"],
                        "description": "操作类型"
                    },
                    "path": {
                        "type": "string",
                        "description": "文件路径"
                    },
                    "content": {
                        "type": "string",
                        "description": "写入的内容（action=write 时需要）"
                    }
                },
                "required": ["action", "path"]
            }
        }
    }
]

TOOL_ACTION_MAP = {
    "create_file": "create",
    "read_file": "read",
    "search_file": "search",
    "edit_file": "update",
    "run_bash": "run_bash",
    "tavily_search": "online_search",
    "run_python": "run_python"
}

class Agent:
    """封装配置和客户端的 Agent 类"""
    
    def __init__(self, max_step: int = 5, max_messages: int = 50, max_token: int = 4000):
        setup_logging(os.getenv("OS_ENV", "dev"))
        # 自动从 .env 加载配置
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL")
        self.model = os.getenv("MODEL_NAME")
        self.max_step = max_step
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.history = History(max_messages=max_messages, max_token=max_token)
        self.system_prompt = ""
        self.step_logs = []
        self.tavilyClient = TavilyClient(os.getenv("TAVILY_API_KEY"))
        self.sbx = Sandbox.create(timeout=600)
        self.log = get_audit_logger(agent="Agent", model=self.model)
        self.enforcer = casbin.Enforcer('config/model.conf', 'config/policy.csv')


    def run(self, input_str: str, user: str = 'anonymous', resource: str = 'data'):
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
            self.log.info("执行的步数次序",  current_step=step+1, max_step = self.max_step)
            # 1. 调 LLM
            message = self._call_llm()
            self.log.info("llm返回的消息", message=message)

            # 2. 把 assistant 的回复存入历史
            self.history.messages.append(message.to_dict())

            # 3. 判断：是调工具还是直接回复？
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args =  json.loads(tool_call.function.arguments)
                    tool_call_id = tool_call.id
                    action = TOOL_ACTION_MAP.get(tool_name, 'unknown')

                    if not self.enforcer.enforce(user, resource, action):
                        self.log.warn("权限不足", user=user, resource=resource, action=action)
                        observation = f"权限不足：{user} 无权对 {resource} 执行 {action}"

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
                        continue
                    
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
                        self.log.info("执行成功",observation = observation)
            
            else:
                # 没有 tool_calls → LLM 直接回复，任务完成
                content = message.content or ""
                self.log.info("💬 LLM 回复", content = content)
                self.step_logs.append({
                    "step": step + 1,
                    "tool": "finish",
                    "args": {},
                    "observation": observation
                })
                self.log.info("task_finish", step_logs=self.step_logs[-1])
                return self._build_result()
        
        #达到最大步数
        self.log.error("max_steps_reached", step = step)
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
                self.log.error("安全检查出错", error=e)
                return f"拒绝操作: {e}"
            
            try:
                create_file(file_str, content)
                self.log.info("成功创建文件", file_str=file_str)
                return f"成功创建文件: {file_str}"
            except Exception as e:
                self.log.error("创建文件出错", error = e)
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
                self.log.error("安全检查出错", error=e)
                return f"拒绝操作: {e}"
            
            try:
                content = read_file(file_str, limit)
                self.log.info("成功读取文件", content=content)
                return f"文件内容: {content}"
            except Exception as e:
                self.log.error("读取文件出错", error = e)
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
                self.log.error("拒绝操作", error=e)
                return f"拒绝操作: {e}"
            
            try:
                result = search_file(directory, pattern)
                self.log.info("成功检索文件", result=result)
                return f"文件检索结果: {result}"
            except Exception as e:
                self.log.error("文件检索失败", error=e)
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
                self.log.error("编辑文件安全检查出错", error = e)
                return f"拒绝操作: {e}"
            
            try:
                content = edit_file(file_str, old_text, new_text)
                self.log.info("成功编辑文件", content=content)
                return f"内容替换成功: {content}"
            except Exception as e:
                self.log.error("编辑文件出错", error = e)
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
                self.log.info("成功执行命令", res=res)
                return f"命令执行成功: {res}"
            except Exception as e:
                self.log.error("执行命令出错", error = e)
                return f"错误: {e}"
        elif tool_name == 'tavily_search':
            query = tool_args.get("query", "")
            timeout = tool_args.get("timeout", 30)
                
            try:
                res = tavily_search(query, timeout, self.tavilyClient)
                self.log.info("成功搜索网络", res=res)
                return f"搜索成功，结果为: {res}"
            except Exception as e:
                self.log.error("搜索网络出错", error = e)
                return f"错误: {e}"
        elif tool_name == 'run_python':
            code = tool_args.get("code", "")
            res = self.sbx.run_code(code)

            if res.error:
                self.log.error("执行python失败", res=res.error)
                return f"执行出错： {res.error}"
            
            stdout_str = "".join(res.logs.stdout)
            stderr_str = "".join(res.logs.stderr)

            if stderr_str:
                self.log.error("执行python出错", res=stderr_str)
                return f"{stdout_str}出错: {stderr_str}"
            self.log.info("成功执行python", res=stdout_str)
            return f"成功{stdout_str}"
            
        elif tool_name == 'e2b_file':
            action = tool_args.get("action", "")
            path = tool_args.get("path", "")
            content = tool_args.get("content", "")

            try:
                if action == 'read':
                    res = self.sbx.files.read(path)
                    self.log.info("成功执行e2b读取操作", res=res)
                    return f"读取内容为: {res}"
                elif action == 'write':
                    self.sbx.files.write(path, content)
                    self.log.info("成功执行e2b写入操作", action=action)
                    return f"写入成功"
                elif action == 'list':
                    file_list = self.sbx.files.list()
                    self.log.info("成功执行e2b查看列表操作", res=file_list)
                    return f"sandbox list清单为：{file_list}"
                else:
                    self.log.error("e2b操作不合法", action=action)
                    return "操作不合法"
            except Exception as e:
                self.log.error("执行e2b操作失败", error=e)
                return f"错误: {e}"

    def _call_llm(self) -> str:
        """调用 LLM"""
        msgs = self.history.build(self.system_prompt)
        self.log.info("DEBUG消息统计",  msg_count =len(msgs), str_len = sum(len(str(m.get('content',''))) for m in msgs))
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=msgs,
                tools=TOOLS,
                tool_choice="auto",
                stream=False
            )
            message = response.choices[0].message
            self.log.info("成功llm调用", message=message)
            return message
        except Exception as e:
            self.log.error("执行llm操作失败", error=e)
            return f"错误: {e}"
    
    def _build_result(self) -> dict:
        """构建最终结果"""
        self.sbx.kill()
        self.log.info("成功构建llm调用结果", step_logs=self.step_logs)
        return {
            "success": any(log.get('tool') == "finish" for log in self.step_logs),
            "steps": len(self.step_logs),
            "logs": self.step_logs
        }