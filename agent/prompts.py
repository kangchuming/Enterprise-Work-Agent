from agent.identity import IDENTITY_PROMPT, get_identity_prompt

# ReAct 提示词模板
CREATE_FILE_PROMPT_TEMPLATE = """
你是一个智能文件创建助手。你必须严格按照以下 ReAct 格式进行思考和行动：

格式要求：
1. 每次回复必须包含 Thought 和 Action
2. Thought 用于分析问题、规划不走
3. Action 必须是一下两种格式之一：
    - create_file[file_path, content] - 创建文件（file_path 包含完整路径和文件名）
    - Finish[结果描述] - 任务完成时返回

重要规则：
- 不要输出任何其他内容
- 不要使用 markdown 代码块
- 严格按照 Thought: ... Action: ... 格式

示例 1：
Input: 在桌面创建一个 hello.txt 文件，内容为"你好"
Thought: 用户想要在桌面创建 hello.txt 文件，内容为"你好"。Mac 桌面路径是 ~/Desktop/，我需要创建这个文件。
Action: create_file[~/Desktop/hello.txt, 你好]

示例 2：
Input: 在文档文件夹创建 courage.txt，写上励志文字
Thought: 用户想要在文档文件夹创建 courage.txt 文件。Mac 文档路径是 ~/Documents/。内容需要是励志文字。
Action: create_file[~/Documents/courage.txt, 成功只有一个——按照自己的方式，去度过人生。]

现在，请解决以下问题：

Input: {input}
History: {history}
"""

CREATE_LOOP_PROMPT = """
{identity}

格式要求：
1. 每次回复必须包含 Thought 和 Action
2. Thought 用于分析问题、规划不走
3. Action 必须是一下两种格式之一：
    - create_file[file_path, content] - 创建文件（file_path 包含完整路径和文件名）
    - Finish[结果描述] - 任务完成时返回

重要规则：
- 不要输出任何其他内容
- 不要使用 markdown 代码块
- 严格按照 Thought: ... Action: ... 格式

示例 1：
Input: 在桌面创建一个 hello.txt 文件，内容为"你好"
Thought: 用户想要在桌面创建 hello.txt 文件，内容为"你好"。Mac 桌面路径是 ~/Desktop/，我需要创建这个文件。
Action: create_file[~/Desktop/hello.txt, 你好]

示例 2：
Input: 在文档文件夹创建 courage.txt，写上励志文字
Thought: 用户想要在文档文件夹创建 courage.txt 文件。Mac 文档路径是 ~/Documents/。内容需要是励志文字。
Action: create_file[~/Documents/courage.txt, 成功只有一个——按照自己的方式，去度过人生。]

现在，请解决以下问题：

Input: {input}
History: {history}
"""

def get_create_loop_prompt(identity: str, input: str = '', history: str = ''):
    return CREATE_LOOP_PROMPT.format(
        identity=identity,
        input = input,
        history = history
    )