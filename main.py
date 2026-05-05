# main.py（项目根目录）
import sys
import os
from agent.core import Agent

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    agent = Agent(max_messages=50, max_token=8000)
    agent.run('show me the first 5 lines of text.txt using a shell command,在1s内完成')
