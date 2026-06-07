import json
from openai import OpenAI
from langsmith import traceable, get_current_run_tree
from langsmith.wrappers import wrap_openai
from agent.identity import get_identity_prompt
from langgraph.types import interrupt
from dotenv import load_dotenv

load_dotenv()   # ← 必须放在 os.getenv（第 212 行）之前
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from agent.prompts import get_create_loop_prompt
from agent.history import History 
from agent.guard import Guard
from agent.log_config import setup_logging, get_audit_logger
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
    "run_python": "run_python",
    "e2b_file": "justify"
}

max_messages = 50
max_token = 8000
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model = os.getenv("MODEL_NAME")
cheap_model = os.getenv("CHEAP_MODEL_NAME")
expensive_model = os.getenv("EXPENSIVE_MODEL_NAME")
_raw_client = OpenAI(api_key=api_key, base_url=base_url)
client = wrap_openai(_raw_client)   # ← LangSmith 自动拦截所有 API 调用
enforcer = casbin.Enforcer('config/model.conf', 'config/policy.csv')
history = History(max_messages=max_messages, max_token=max_token)
log = get_audit_logger(agent="Agent", model=model)
# 模块级单例
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = Path(project_root) / "data"
tavilyClient = TavilyClient(os.getenv("TAVILY_API_KEY"))
_sbx: Sandbox | None = None
setup_logging(os.getenv("OS_ENV", "dev"))

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

def _get_sandbox() -> Sandbox:
    global _sbx
    if _sbx is None:
        _sbx = Sandbox.create(timeout=600)
    return _sbx

# ====== 每个工具一个短函数 ======
def _exec_create_file(args: dict) -> str:
    _guard_check_path(args["file_str"])
    create_file(args["file_str"], args["content"])
    log.info("成功创建文件", file_str=args["file_str"])
    return f"成功创建文件: {args['file_str']}"

def _exec_read_file(args: dict) -> str:
    _guard_check_path(args["file_str"])
    content = read_file(args["file_str"], args.get("limit"))
    log.info("成功读取文件", content=content[:200])
    return f"文件内容: {content}"

def _exec_search_file(args: dict) -> str:
    _guard_check_path(args["directory"])
    result = search_file(args["directory"], args["pattern"])
    log.info("成功检索文件", result=str(result)[:200])
    return f"文件检索结果: {result}"

def _exec_edit_file(args: dict) -> str:
    _guard_check_path(args["file_str"])
    content = edit_file(args["file_str"], args["old_text"], args["new_text"])
    log.info("成功编辑文件", content=str(content)[:200])
    return f"内容替换成功: {content}"

def _exec_run_bash(args: dict) -> str:
    block_reason = Guard().guard_command(args["command"])
    if block_reason:
        log.warn("命令被拦截", command=args["command"][:100], reason=block_reason)
        return block_reason
    res = run_bash(args["command"], args.get("timeout", 30))
    log.info("成功执行命令", res=str(res)[:200])
    return f"命令执行成功: {res}"

def _exec_tavily_search(args: dict) -> str:
    res = tavily_search(args["query"], args.get("timeout", 30), tavilyClient)
    log.info("成功搜索网络", res=str(res)[:200])
    return f"搜索成功，结果为: {res}"

def _exec_run_python(args: dict) -> str:
    sbx = _get_sandbox()
    res = sbx.run_code(args["code"])
    if res.error:
        log.error("执行python失败", error=res.error)
        return f"执行出错：{res.error}"
    stdout_str = "".join(res.logs.stdout)
    stderr_str = "".join(res.logs.stderr)
    if stderr_str:
        log.error("执行python出错", stderr=stderr_str)
        return f"{stdout_str}出错: {stderr_str}"
    log.info("成功执行python", stdout=stdout_str[:200])
    return f"成功{stdout_str}"

def _exec_e2b_file(args: dict) -> str:
    action = args["action"]
    sbx = _get_sandbox()
    if action == "read":
        res = sbx.files.read(args["path"])
        log.info("成功执行e2b读取操作", res=str(res)[:200])
        return f"读取内容为: {res}"
    elif action == "write":
        sbx.files.write(args["path"], args.get("content", ""))
        log.info("成功执行e2b写入操作", path=args["path"])
        return "写入成功"
    elif action == "list":
        file_list = sbx.files.list()
        log.info("成功执行e2b查看列表操作", res=str(file_list)[:200])
        return f"sandbox list清单为：{file_list}"
    log.error("e2b操作不合法", action=action)
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
def _execute_tool(tool_name: str, tool_args: dict) -> dict:
    """执行工具，通过注册表分发"""
    log.info("开始执行工具", tool=tool_name, args=str(tool_args)[:200])
    executor = TOOL_EXECUTORS.get(tool_name)
    if executor is None:
        log.error("未知工具", tool_name=tool_name)
        return {"success": False, "content": tool_name}
    try:
        return {"success": True, "content": executor(tool_args)}
    except PermissionError as e:
        log.error("安全检查出错", error=e)
        return {"success": False, "content": f"拒绝操作: {e}"}
    except Exception as e:
        log.error("工具执行出错", tool=tool_name, error=e)
        return {"success": False, "content": f"错误: {e}"}

def prepare_prompt(state: AgentState):
    """构建 system prompt + 拼接历史"""
    log.info("prepare_prompt 开始", input=state["input"][:100], user=state.get("user", "admin"))
    identity = get_identity_prompt(
            os_info="macOS",
            workspace_path=project_root
        )

    system_prompt = get_create_loop_prompt(
        identity = identity
    )
    msgs_to_add = []
    if not state["messages"]:
        msgs_to_add = [{"role": "user", "content": state["input"]}]

    log.info("system_prompt 构建完成", prompt_len=len(system_prompt))
    return {
        "system_prompt": system_prompt,
        "messages": msgs_to_add
    }

def _classify_intent(query: str) -> str:
    """
    用便宜模型判断用户意图复杂度。
    返回 "simple" 或 "complex"。
    分类失败时默认返回 "simple"（走便宜模型，安全优先）。
    """
    try:
        resp = client.chat.completions.create(
            model=cheap_model,  # 用最便宜的模型做分类
            messages=[
                {
                    "role": "system",
                    "content": (
                        "判断用户请求的复杂度，只输出一个单词：simple 或 complex。\n\n"
                        "simple：单步操作、简单问答、基础文件读写、定义查询。\n"
                        "  例：'读取 README.md'、'创建 test.txt'、'什么是 Python'\n\n"
                        "complex：多步推理、代码生成、架构分析、方案设计、代码审查。\n"
                        "  例：'分析项目架构'、'写一个登录系统'、'找出代码安全问题'\n\n"
                        "只输出 simple 或 complex，不要其他内容。"
                    )
                },
                {"role": "user", "content": query}
            ],
            max_tokens=500,
            temperature=0,
            langsmith_extra={
                "metadata": {"classifier_model": cheap_model}
            }
        )
        msg = resp.choices[0].message
        assistant_msg = {"role": "assistant", "content": msg.content}

        # DeepSeek 思考模式需要回传 reasoning_content
        if hasattr(msg, "reasoning_content") and msg.reasoning_content:
            assistant_msg["reasoning_content"] = msg.reasoning_content

        content = (resp.choices[0].message.content or '').strip().lower()
        return content if content in ('simple', 'complex') else 'simple'
    except Exception:
        return 'simple' # 分类失败默认走便宜模型

def call_llm(state: AgentState):
    """调用 LLM"""
    history.messages = state["messages"]
    msgs = history.build(state["system_prompt"])
    step = state.get("step", 0) + 1
    log.info("LLM 调用开始", step=step, msg_count=len(msgs), str_len=sum(len(str(m.get("content", ""))) for m in msgs))

    user_input = state.get('input', '')
    intent = _classify_intent(user_input)
    selected_model = cheap_model if intent == 'simple' else expensive_model

    run = get_current_run_tree()

    if run:
        run.add_metadata({
            "intent": intent,
            "model": selected_model
        })

    log.info("模型路由",
             step=step,
             intent=intent,
             selected_model=selected_model,
             query_preview=user_input[:80])

    try:
        response = client.chat.completions.create(
            model = selected_model,
            messages = msgs,
            tools = TOOLS,
            tool_choice = "auto",
            stream = False,
            langsmith_extra={  # ← ✅ 两本指南都推荐的方式
                "metadata": {
                    "intent": intent,
                    "model": selected_model,
                    "step": step,
                }
            }
        )
        
        msg = response.choices[0].message
        assistant_msg = {"role": "assistant", "content": msg.content}

        # DeepSeek 思考模式需要回传 reasoning_content
        if hasattr(msg, "reasoning_content") and msg.reasoning_content:
            assistant_msg["reasoning_content"] = msg.reasoning_content

        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } 
                for tc in msg.tool_calls
            ]
            log.info("LLM 返回工具调用", tool_calls=assistant_msg["tool_calls"])
            return {
                "messages": [assistant_msg],
                "tool_calls": assistant_msg["tool_calls"],
                "step": step  
            }
        else:
            log.info("LLM 直接回复", content=str(msg.content)[:200])
            return {
                "messages": [assistant_msg],
                "tool_calls": [],
                "step": step
            }

    except Exception as e:
        log.error("LLM 调用失败", error=e)
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

    all_messages = []
    all_logs = []

    for tc in tool_calls:
        tool_name = tc["function"]["name"]
        tool_args = json.loads(tc["function"]["arguments"])
        tool_call_id = tc["id"]

        action = TOOL_ACTION_MAP.get(tool_name, 'unknown')
        user = state.get("user", "admin")
        resource = state.get("resource", "data")

        log.info("权限检查", step=state["step"], tool=tool_name, action=action, user=user, resource=resource)

        if not enforcer.enforce(user, resource, action):
            observation = f"权限不足：{user} 无权对 {resource} 执行 {action}"
            log.warn("权限不足", user=user, resource=resource, action=action)

            all_messages.append({
                        'role': 'tool',
                        'name': tool_name,
                        'tool_call_id': tool_call_id,
                        'content': observation              
            })
            all_logs.append({
                "step": state["step"],
                "tool": tool_name,
                "args": tool_args,
                "observation": observation
            })
            continue

        # 新增：高危操作中断审批
        if tool_name in ["tavily_search", "run_bash"]:
            decision = interrupt(f"是否批准 {tool_name}？参数: {tool_args}")
            if decision != 'yes':
                all_messages.append({"role": "tool", "content": "被用户拒绝", "tool_call_id": tool_call_id, 'name': tool_name})
                all_logs.append({
                    "step": state["step"],
                    "tool": tool_name,
                    "args": tool_args,
                    "observation":  "被用户拒绝",
                    'name': tool_name
                })
                continue

        observation = _execute_tool(tool_name, tool_args)
        log.info("工具执行完成", tool=tool_name, observation=observation["content"][:200])
        
        all_messages.append({
            "role": "tool", 
            "content": observation["content"], 
            "tool_call_id": tool_call_id, 
            'name': tool_name
        })

        all_logs.append({
            "step": state["step"],
            "tool": tool_name,
            "args": tool_args,
            "observation": observation
        })
        
    return {
        "messages": all_messages,
        "step_logs": all_logs,
        "tool_calls": []
    }

# ====== 3. 条件路由 ======
def should_continue(state: AgentState) -> Literal["execute", "finish"]:
    step_logs = state.get("step_logs", [])

    if state.get("step", 0) >= state.get("max_step", 5):
        log.warn("达到最大步数限制，结束", step=state["step"], max_step=state.get("max_step", 5))
        return "finish"
    elif state.get("tool_calls", []):
        log.info("继续执行工具", step=state["step"])
        return "execute"
    else:
        log.info("无工具调用，结束对话", step=state["step"])
        return "finish"

def finish(state: AgentState):
    """提取最终结果"""
    global _sbx
    if _sbx is not None:
        _sbx.kill()
        _sbx = None

    final_answer = ""
    for msg in reversed(state.get("messages", [])):
        if msg.get("role") == "assistant":
            final_answer = msg.get("content", "")
            break

    log.info("任务完成", steps=state.get("step", 0), final_answer=final_answer[:200])
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

if __name__ == "__main__":
    _classify_intent("帮我设计一个微服务架构")
