# main.py（项目根目录）
import sys
import os
from agent.core import Agent

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    agent = Agent(max_messages=50, max_token=8000)
    agent.run('在Sandbox环境下，新建一个a.md 帮我写入文字：夜深忽梦少年事，梦啼妆泪红阑干。醒来风止人亦远，旧我沉入旧时山。')
