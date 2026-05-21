import json
from openai import OpenAI
from agent.identity import get_identity_prompt
from dotenv import load_dotenv

load_dotenv()   # ← 必须放在 os.getenv（第 212 行）之前
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from agent.prompts import CREATE_LOOP_PROMPT, get_create_loop_prompt
from agent.history import History 
from agent.guard import Guard
from pathlib import Path
from tools.file_manager import create_file, read_file, search_file, edit_file, run_bash, tavily_search
from tavily import TavilyClient
from e2b_code_interpreter import Sandbox

import casbin
import sqlite3
import operator
import os

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

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model = os.getenv("MODEL_NAME")
client = OpenAI(api_key=api_key, base_url=base_url)
enforcer = casbin.Enforcer('config/model.conf', 'config/policy.csv')

# 模块级单例
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = Path(project_root) / "data"
tavilyClient = TavilyClient(os.getenv("TAVILY_API_KEY"))
sbx = Sandbox.create(timeout=600)

# ====== 1. State 定义 ======
class AgentState(TypedDict):
    # 输入
    input: str
    user: str
    resource: str
    # 对话
    messages: Annotated[list, operator.add]
    system_prompt: str
    tool_calls: list
    # LLM 输出
    step_logs: Annotated[list, operator.add]
    # 循环控制
    max_step: int
    step: int
    # 结果
    results: list

# ====== 2. 节点函数 ======

# ====== 公共 helper：文件路径安全检查 ======
def _guard_check_path(file_str: str):
    """对文件路径做 Guard 安全检查，不通过抛 PermissionError"""
    Guard()._resolve_path(file_str, data_dir)


# ====== 每个工具一个短函数 ======
def _exec_create_file(args: dict) -> str:
    _guard_check_path(args["file_str"])
    create_file(args["file_str"], args["content"])
    return f"成功创建文件: {args['file_str']}"

def _exec_read_file(args: dict) -> str:
    _guard_check_path(args["file_str"])
    content = read_file(args["file_str"], args.get("limit"))
    return f"文件内容: {content}"

def _exec_search_file(args: dict) -> str:
    _guard_check_path(args["directory"])
    result = search_file(args["directory"], args["pattern"])
    return f"文件检索结果: {result}"

def _exec_edit_file(args: dict) -> str:
    _guard_check_path(args["file_str"])
    content = edit_file(args["file_str"], args["old_text"], args["new_text"])
    return f"内容替换成功: {content}"

def _exec_run_bash(args: dict) -> str:
    block_reason = Guard().guard_command(args["command"])
    if block_reason:
        return block_reason
    res = run_bash(args["command"], args.get("timeout", 30))
    return f"命令执行成功: {res}"

def _exec_tavily_search(args: dict) -> str:
    res = tavily_search(args["query"], args.get("timeout", 30), tavilyClient)
    return f"搜索成功，结果为: {res}"

def _exec_run_python(args: dict) -> str:
    res = sbx.run_code(args["code"])
    if res.error:
        return f"执行出错：{res.error}"
    stdout_str = "".join(res.logs.stdout)
    stderr_str = "".join(res.logs.stderr)
    if stderr_str:
        return f"{stdout_str}出错: {stderr_str}"
    return f"成功{stdout_str}"

def _exec_e2b_file(args: dict) -> str:
    action = args["action"]
    if action == "read":
        return f"读取内容为: {sbx.files.read(args['path'])}"
    elif action == "write":
        sbx.files.write(args["path"], args.get("content", ""))
        return "写入成功"
    elif action == "list":
        return f"sandbox list清单为：{sbx.files.list()}"
    return f"不支持的操作类型: {action}"


# ====== 注册表：工具名 → 执行函数 ======
TOOL_EXECUTORS = {
    "create_file":   _exec_create_file,
    "read_file":     _exec_read_file,
    "search_file":   _exec_search_file,
    "edit_file":     _exec_edit_file,
    "run_bash":      _exec_run_bash,
    "tavily_search": _exec_tavily_search,
    "run_python":    _exec_run_python,
    "e2b_file":      _exec_e2b_file,
}


# ====== 统一入口：一行分发 ======
def _execute_tool(tool_name: str, tool_args: dict) -> str:
    """执行工具，通过注册表分发"""
    executor = TOOL_EXECUTORS.get(tool_name)
    if executor is None:
        return f"未知工具: {tool_name}"
    try:
        return executor(tool_args)
    except PermissionError as e:
        return f"拒绝操作: {e}"
    except Exception as e:
        return f"错误: {e}"

def prepare_prompt(state: AgentState):
    """构建 system prompt + 拼接历史"""
    identity = get_identity_prompt(
            os_info="macOS",
            workspace_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

    system_prompt = get_create_loop_prompt(
        identity = identity
    )

    return {
        "system_prompt": system_prompt
    }

def call_llm(state: AgentState):
    """调用 LLM"""
    msgs = [{"role": "system", "content": state["system_prompt"]}] + state["messages"]
    step = state.get("step", 0) + 1

    try:
        respone = client.chat.completions.create(
            model = model,
            messages = msgs,
            tools = TOOLS,
            tool_choice = "auto",
            stream = False
        )
        
        msg = respone.choices[0].message
        assistant_msg = {"role": "assistant", "content": msg.content}

        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } 
                for tc in msg.tool_calls
            ]
            return {
                "messages": [assistant_msg],
                "tool_calls": assistant_msg["tool_calls"],
                "step": step  
            }
        else:
            return {
                "messages": [assistant_msg],
                "tool_calls": [],
                "step": step
            }

    except Exception as e:
        return {
            "messages": [{"role": "assistant", "content": f"调用 LLM 出错: {e}"}],
            "tool_calls": [],
            "step": step  
        }
    

def guard_check_and_execute(state: AgentState):
    """权限检查 + 工具执行"""
    tool_calls = state["tool_calls"]

    if not tool_calls:
        return {}

    tc = tool_calls[0]
    tool_name = tc["function"]["name"]
    tool_args = json.loads(tc["function"]["arguments"])
    tool_call_id = tc["id"]

    action = TOOL_ACTION_MAP.get(tool_name, 'unknown')
    user = state.get("user", "anonymous")
    resource = state.get("resource", "data")

    if not enforcer.enforce(user, resource, action):
        observation = f"权限不足：{user} 无权对 {resource} 执行 {action}"
        return {
            "messages": [{
                        'role': 'tool',
                        'name': tool_name,
                        'tool_call_id': tool_call_id,
                        'content': observation
                        
            }],
            "step_logs": [{
                "step": state["step"],
                "tool": tool_name,
                "args": tool_args,
                "observation": observation
            }]
        }
    
    observation = _execute_tool(tool_name, tool_args)

    return {
        "messages": [
            {
                'role': 'tool',
                'name': tool_name,
                'tool_call_id': tool_call_id,
                'content': observation
            }
        ],
        "step_logs": [{
                "step": state["step"],
                "tool": tool_name,
                "args": tool_args,
                "observation": observation
        }]
    }

# ====== 3. 条件路由 ======
def should_continue(state: AgentState) -> Literal["execute", "finish"]:
    if state.get("step", 0) >= state.get("max_step", 5):
        return "finish"
    elif state.get("tool_calls", []):
        return "execute"
    else:
        return "finish"

def finish(state: AgentState):
    """提取最终结果"""
    sbx.kill()

    final_answer = ""
    for msg in reversed(state.get("messages", [])):
        if msg.get("role") == "assistant":
            final_answer = msg.get("content", "")
            break

    return {
        "step_logs": [{
            "step": state.get("step", 0),
            "tool": "finish",
            "args": {},
            "observation": final_answer,
        }],
        "results": [{
            "success": True,
            "final_answer": final_answer,
            "steps": state.get("step", 0),
            "logs": state.get("step_logs", []),
        }],
    }


# ====== 4. 组装图 ======
graph = StateGraph(state_schema=AgentState)

graph.add_node("prepare", prepare_prompt)
graph.add_node("llm", call_llm)
graph.add_node("execute", guard_check_and_execute)
graph.add_node("finish", finish)

graph.add_edge(START, 'prepare')
graph.add_edge('prepare', 'llm')

graph.add_conditional_edges(
    'llm',
    should_continue,
    {
        'execute': 'execute',
        "finish": "finish"
    }
)

graph.add_edge("execute", "llm")
graph.add_edge('finish', END)

app = graph.compile(
    checkpointer=SqliteSaver(sqlite3.connect("checkpoints.db"))
)
