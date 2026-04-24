from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()

class Config:
    """配置管理类"""

    def init(self):
        project_root = Path(__file__).parent.parent
        self.workspace_path = str(project_root)
        self.data_path = str(project_root / "data")

        #Key引入
        self.openai_api_key = self._get_env("OPENAI_API_KEY")
        self.openai_base_url = self._get_env("OPENAI_BASE_URL")
        self.model_name = self._get_env("MODEL_NAME")
        self.tavily_api_key = self._get_env("TAVILY_API_KEY")

    def _get_env(self, key) -> str:
        """获取环境变量，如果不存在返回空字符串"""
        return os.getenv(key, "")
