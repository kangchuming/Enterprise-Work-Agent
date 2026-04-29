class History:
    """最简历史管理——从 nanobot session/manager.py 掠夺"""

    def __init__(self, max_messages: int = 50, max_token: int = 4000):
        #初始化对话消息
        self.messages:list[dict] = []
        self.max_messages = max_messages
        self.max_token = max_token
    
    def add_user(self, content: str):
        #记录用户提升
        self.messages.append({'role': 'user', 'content': content})
    
    def add_assistant(self, content: str, tool_calls: dict = None):
        msg = {'role': 'assistant', 'content': content}
        if tool_calls:
            msg['tool_calls'] = tool_calls
        #记录模型回复
        self.messages.append(msg)
    
    def add_tool_result(self, tool_call_id, name, result):
        self.messages.append({'role': 'tool', 'name': name, 'tool_call_id': tool_call_id, 'content': result})
    
    def build(self, system_prompt: str) -> list[dict]:
        #条数截断：只保留最近的几条
        msgs = self.messages[-self.max_messages:]
        msgs = self._snip_by_tokens(msgs, system_prompt)
        msgs = self._align_to_user_start(msgs)
        #拼装完整消息列表发给LLM
        return [
            {'role': 'system', 'content': system_prompt},
            *msgs
        ]
    
    def _snip_by_tokens(self, messages, system_prompt: str) -> list[dict]:
        budget = self.max_token - self._count_token(system_prompt) 
        kept = []
        total = 0

        for t in reversed(messages):
            tokens = self._count_token(str(t.get("content", ""))) 
            if kept and total + tokens > budget:
                break
            kept.append(t)
            total += tokens

        kept.reverse()
        return kept
    
    def _align_to_user_start(self, messages: list[dict]) -> list[dict]:
        for i, msg in enumerate(messages):
            if(msg.get('role') == 'user'):
                return messages[i:]
        return messages
        
    @staticmethod
    def _count_token(text: str) -> int:
        return max(1, len(text.encode('utf-8')) // 3)
