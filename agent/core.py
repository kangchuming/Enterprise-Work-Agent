import sys
import os
# 将项目根目录添加到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from openai import OpenAI
import re
from pathlib import Path
from agent.prompts import CREATE_LOOP_PROMPT, get_create_loop_prompt
from tools.file_manager import create_file
from agent.identity import get_identity_prompt
from agent.guard import _resolve_path, guard_command
from dotenv import load_dotenv

load_dotenv()

class OpenAICompatibleClient:
    """
    一个用于调用任何兼容OpenAI接口的LLM服务的客户端。
    """
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, system_prompt: str) -> str:
        """调用LLM API来生成回应。"""
        print("正在调用大语言模型...")
        try:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': '创建一个名字为aus的文档'}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )
            answer = response.choices[0].message.content
            print("大语言模型响应成功。")
            return answer
        except Exception as e:
            print(f"调用LLM API时发生错误: {e}")
            return "错误:调用语言模型服务时出错。"

class Agent:
    """封装配置和客户端的 Agent 类"""
    
    def __init__(self, max_step: int = 5):
        # 自动从 .env 加载配置
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL")
        self.model = os.getenv("MODEL_NAME")
        self.max_step = max_step
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.messages = []
        self.step_logs = []

    def run(self, input_str: str):
        """运行 Agent"""
        # 获取身份提示词（填充 workspace_path 等变量）
        identity = get_identity_prompt(
            os_info="macOS",
            workspace_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        
        # 构建完整的 system_prompt
        system_prompt = get_create_loop_prompt(
            identity=identity,
            input="请创建 test.txt 文件，内容是 Hello",
            history=""
        )

        self.messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': input_str}
        ]

        for step in range(self.max_step):
            print(f"\n{'='*40}")
            print(f"Step {step + 1}/{self.max_step}")
            print(f"\n{'='*40}")

            # 1. 调 LLM
            response = self._call_llm()
            print(f"响应: {response}")
            # 2. 解析 Thought 和 Action
            max_retries = 3
            for retry in range(max_retries):
                try:
                    action, thought = self._parse_response(response)
                    if not action:
                        raise ValueError("Action 为空")
                    break
                except Exception as e:
                    # 解析失败 → 自纠错：把报错信息塞回 messages
                    error_msg = f"格式错误：{e}。请严格按照 Thought: ... Action: ... 格式输出，不要输出其他内容。"
                    print(f"⚠️ 解析失败 (重试 {retry + 1}/{max_retries}): {e}")
                    
                    # 把 LLM 的错误输出和报错信息都塞回 messages
                    self.messages.append({"role": "assistant", "content": response})
                    self.messages.append({"role": "user", "content": error_msg})
                    response = self._call_llm()
                    print(f"response: {response}")
            else:
                print(f"解析最大次数使用完: {max_retries}")
                continue
            print(f"action: {action}, thought: {thought}")

            # 3. 执行 Action
            observation = self._execute_action(thought)
            print(f"Observation: {observation}")
            if "成功" in observation:
                print(f"Observation: {observation}")
                print(f"✅ 检测到成功，自动结束")
                self.step_logs.append({
                    "step": step + 1,
                    "thought": thought,
                    "action": "Finish[任务完成]",
                    "observation": observation
                })
                return self._build_result()
        
            # 4. 记录日志
            self.step_logs.append({
                "step": step + 1,
                "action": action,
                "thought": thought,
                "observation": observation
            })
            # 5. 结果塞回 messagess
            self.messages.append({
                "role": "assistant",
                "content": f"Thought: {thought}\nAction: {action}"
            })
            self.messages.append({
                "role": "user",
                "content": f"Observation: {observation}"
            })
            # 6. 检查是否完成
            # response = self.client.generate(system_prompt= system_prompt)
            # print(f"响应: {response}")
            if action.startswith('Finish'):
                print(f"\n任务完成！")
                return self._build_result()
        print(f"\n已达到最大重试次数")
        return self._build_result()

    def _call_llm(self) -> str:
        """调用 LLM"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                stream=False
            )
            answer = response.choices[0].message.content
            return answer
        except Exception as e:
            return f"错误: {e}"

    def _parse_response(self, response: str) -> tuple:
        """解析 Thought 和 Action"""
        thought_match = re.search(r"Thought:\s*(.*?)(?=Action:|$)", response, re.DOTALL)
        action_match = re.search(r"Action:\s*(.*)", response, re.DOTALL)
        
        thought = thought_match.group(1).strip() if thought_match else ""
        action = action_match.group(1).strip() if action_match else "Finish[未能解析]"
        
        return thought, action
    
    def _execute_action(self, action: str) -> str:
        """执行工具（目前只有 create_file）"""
        
        if action.startswith('create_file'):
            match = re.match(r'create_file\[(.*?),\s*(.*)$$', action, re.DOTALL)
            if match:
                file_str = match.group(1).strip()
                content = match.group(2).strip()

                # ✅ Guard: 路径安全检查
                try:
                    data_dir = Path(project_root) / "kcm"
                    _resolve_path(file_str, data_dir)
                except PermissionError as e:
                    return f"拒绝操作: {e}"

                try:
                    create_file(file_str, content)
                    return "成功操作 create_file "
                except Exception as e:
                    return f"错误: {e}"
            
        elif action.startswith("finish"):
            return "任务完成"
        else:
            return f"位置操作：{action}"

    
    def _build_result(self) -> dict:
        """构建最终结果"""
        return {
            "success": any(log["action"].startswith("Finish") for log in self.step_logs),
            "steps": len(self.step_logs),
            "logs": self.step_logs
        }