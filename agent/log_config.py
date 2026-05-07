import logging
from pathlib import Path
import structlog

def setup_logging(env: str = "dev"):
    """
        统一日志配置入口
        
        Args:
            env: "dev" → 彩色终端输出；"prod" → JSON 文件输出
    """
    
    if env == 'dev':
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.dev.set_exc_info,
                structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
                structlog.dev.ConsoleRenderer()
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True
        )
    else:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = open(log_dir / "agent.log", "a", encoding="utf-8")

        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.dev.set_exc_info,
                structlog.processors.JSONRenderer()

            ],
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(file=log_file),
            cache_logger_on_first_use=True
        )


def get_audit_logger(**context):
    """获取带审计上下文的 logger，用于绑定 request_id、user_id 等字段"""
    return structlog.get_logger().bind(**context)

