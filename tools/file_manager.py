import os

def create_file(filePath, content):
    """
    创建文件，自动创建父目录
    
    Args:
        file_path: 完整文件路径（包含文件名），如 ~/Documents/courage.txt
        content: 文件内容
    """
    # 是否存在目录
    file_path = os.path.expanduser(filePath)

    dir_path = os.path.dirname(filePath)


    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        print(f"创建目录: {dir_path}")
    # 是否存在文件
    try:
        with open(file_path , 'w', encoding='utf-8') as f:
            f.write(content)
        print("写入成功")
        return True
    except FileExistsError as e:
        print(f"写入失败: {e}")
        return False

def read_file(filePath):
    """
    读取文件
    
    Args:
        file_path: 完整文件路径（包含文件名），如 ~/Documents/courage.txt
    """
    file_path = os.path.expanduser(filePath)

    #读取文件
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print("读取成功")
            return content
    except FileNotFoundError as e:
        print(f"文件不存在: {file_path} {e}")
        return f"文件不存在: {file_path}"
    except PermissionError as e:
        print(f"文件权限不允许读取 {e}")
        return f"文件权限不允许读取: {file_path}"