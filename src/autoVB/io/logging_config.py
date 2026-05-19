from __future__ import annotations

import inspect
import logging
import sys
import warnings
from typing import TextIO


INFO_LOG_FORMAT: str = "%(message)s"
LEVEL_LOG_FORMAT: str = "%(levelname)s: %(message)s"
DEBUG_LOG_FORMAT = "[%(levelname)s][%(autovb_scope)s] %(message)s"

_DEFAULT_LOG_RECORD_FACTORY = logging.getLogRecordFactory()
_LOG_RECORD_FACTORY_INSTALLED: bool = False


def _find_caller_class_name(record: logging.LogRecord) -> str:
    """从调用栈中推断 logger 调用所在的类名。"""
    frame = inspect.currentframe()
    try:
        frame = frame.f_back if frame is not None else None
        while frame is not None:
            code = frame.f_code
            if code.co_filename == record.pathname and code.co_name == record.funcName:
                self_obj = frame.f_locals.get("self")
                if self_obj is not None:
                    return self_obj.__class__.__name__
                cls_obj = frame.f_locals.get("cls")
                if isinstance(cls_obj, type):
                    return cls_obj.__name__
                return ""
            frame = frame.f_back
    finally:
        del frame
    return ""


def _autovb_log_record_factory(*args: object, **kwargs: object) -> logging.LogRecord:
    """为日志记录补充作用域字段。"""
    record = _DEFAULT_LOG_RECORD_FACTORY(*args, **kwargs)
    class_name = _find_caller_class_name(record)
    record.autovb_scope = f"{class_name}.{record.funcName}" if class_name else record.funcName
    return record


def _install_log_record_factory() -> None:
    """安装一次 LogRecord factory，避免重复包装。"""
    global _LOG_RECORD_FACTORY_INSTALLED
    if not _LOG_RECORD_FACTORY_INSTALLED:
        logging.setLogRecordFactory(_autovb_log_record_factory)
        _LOG_RECORD_FACTORY_INSTALLED = True


class AutoVBLogFormatter(logging.Formatter):
    """日志格式化器：INFO 保持纯文本，其它级别带级别前缀。"""

    def __init__(self) -> None:
        super().__init__()
        # INFO 作为用户可读运行输出，不额外加 "INFO:" 前缀。
        self.info_formatter: logging.Formatter = logging.Formatter(INFO_LOG_FORMAT)
        # WARNING/ERROR 等级保留前缀，方便用户在 .out 中快速定位问题。
        self.level_formatter: logging.Formatter = logging.Formatter(DEBUG_LOG_FORMAT)

    def format(self, record: logging.LogRecord) -> str:
        """根据日志级别选择输出格式。"""
        if record.levelno == logging.INFO:
            return self.info_formatter.format(record)
        return self.level_formatter.format(record)


def configure_logging(
    level: int = logging.INFO,
    *,
    stream: TextIO | None = None,
    force: bool = False,
) -> None:
    """配置 autoVB 的全局日志输出和 warnings 转发。

    Args:
        level: 根 logger 的日志级别，例如 ``logging.INFO`` 或 ``logging.DEBUG``。
        stream: 日志输出流；如果为 ``None``，默认写入 ``sys.stdout``，方便
            ``autovb input.autovb > input.out`` 捕获用户可读日志。
        force: 是否强制覆盖已有 logging 配置。为 ``False`` 时，如果根 logger
            已经有 handler，则只更新日志级别。

    Returns:
        None。
    """
    _install_log_record_factory()
    root_logger: logging.Logger = logging.getLogger()
    if force or not root_logger.handlers:
        # autoVB 的 .out 是用户可读结果文件，所以默认把日志写到 stdout；
        # 这样 `autovb input.autovb > input.out` 会包含完整运行信息。
        handler: logging.StreamHandler[TextIO] = logging.StreamHandler(stream or sys.stdout)
        handler.setFormatter(AutoVBLogFormatter())
        logging.basicConfig(
            level=level,
            handlers=[handler],
            force=force,
        )
    else:
        root_logger.setLevel(level)

    # 将 warnings.warn(...) 统一转发给 logging，避免 warning 格式分裂。
    logging.captureWarnings(True)
    warnings.simplefilter("default")


def get_logger(name: str) -> logging.Logger:
    """获取使用 autoVB 全局配置的 logger。"""
    return logging.getLogger(name)

def autovb_warning(message: str, category: type[Warning] = RuntimeWarning, stacklevel: int = 2):
    """
    打印警告信息。
    Args:
        message (str): 警告信息内容
    """    
    logger = get_logger(__name__)
    logger.info("!" * 40)
    warnings.warn(message, category=category, stacklevel=stacklevel)
    logger.info("!" * 40)

def autovb_subroutine(message: str):
    """
    打印子程序信息。
    Args:
        message (str): 子程序信息内容
    """
    logger = get_logger(__name__)
    logger.info("=" * 40)
    logger.info(f"{message}")
    logger.info("=" * 40)
