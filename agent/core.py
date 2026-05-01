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
from tools.file_manager import create_file, read_file
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
                },
                "required": ["file_str"]  # 哪些参数必须传
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
            identity=identity,
            input="input_str",
            history=""
        )

        # self.messages = [
        #     {'role': 'system', 'content': system_prompt},
        #     {'role': 'user', 'content': input_str}
        # ]
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

            # Guard安全检查
            try:
                data_dir = Path(project_root) / "data"
                guard = Guard()
                guard._resolve_path(file_str, data_dir)
            except PermissionError as e:
                return f"拒绝操作: {e}"
            
            try:
                content = read_file(file_str)
                return f"文件内容: {content}"
            except Exception as e:
                return f"错误: {e}"
        else:
            return f"未知工具 {tool_name}"

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