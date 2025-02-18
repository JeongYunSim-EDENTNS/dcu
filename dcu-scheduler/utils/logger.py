import logging
import os
import re
import pytz
from datetime import datetime
from colorlog import ColoredFormatter
from tqdm.std import tqdm as std_tqdm
from contextlib import contextmanager

logger_handlers = []


class MakeFileHandler(logging.FileHandler):
    """로그 파일이 저장될 디렉터리를 생성하며 로그 파일을 생성"""
    def __init__(self, filename, mode="a", encoding=None, delay=0):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        super().__init__(filename, mode, encoding, delay)


class CustomColoredFormatter(ColoredFormatter):
    """컬러 포맷터로 로그 메시지의 시간대를 설정"""
    def __init__(self, *args, timezone=pytz.UTC, **kwargs):
        super().__init__(*args, **kwargs)
        self.timezone = timezone

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, pytz.UTC).astimezone(self.timezone)
        return dt.strftime(datefmt) if datefmt else dt.isoformat()


class RemoveAnsiEscapeFormatter(logging.Formatter):
    """ANSI escape 코드를 제거하는 포맷터"""
    def format(self, record):
        ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
        record.msg = ansi_escape.sub('', record.msg)
        return super().format(record)


def get_logger(name=None, level=None, log_dir=None, timezone=None):
    """로거를 생성 및 반환"""
    logger = logging.getLogger(name)

    # 기본 타임존 설정
    timezone = timezone or pytz.UTC

    # 로그 레벨 설정
    log_level = level or os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(log_level)

    # 스트림 핸들러 추가 (컬러 로깅)
    stream_handler = logging.StreamHandler()
    stream_formatter = CustomColoredFormatter(
        "%(bold_green)s%(asctime)s%(reset)s %(log_color)s%(levelname)-8s%(reset)s %(bold_purple)s%(name)-15s%(reset)s %(white)s%(message)s%(reset)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "blue",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red"
        },
        datefmt="%Y-%m-%d %H:%M:%S",
        reset=True,
        style='%',
        timezone=timezone
    )
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    # 파일 핸들러 추가 (로그를 파일에 기록)
    if log_dir:
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(log_dir, name or "app", f"{today}.log")
        file_handler = MakeFileHandler(log_file, encoding="utf8")
        file_formatter = RemoveAnsiEscapeFormatter("%(asctime)s %(levelname)-8s %(name)-15s %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    logger_handlers.append(logger)
    return logger


def refresh_log_level():
    """환경 변수 LOG_LEVEL에 따라 로깅 수준 갱신"""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    for logger in logger_handlers:
        logger.setLevel(log_level)


class _TqdmLoggingHandler(logging.StreamHandler):
    """tqdm와 호환되는 로깅 핸들러"""
    def __init__(self, tqdm_class=std_tqdm):
        super().__init__()
        self.tqdm_class = tqdm_class

    def emit(self, record):
        try:
            msg = self.format(record)
            self.tqdm_class.write(msg, file=self.stream)
            self.flush()
        except Exception:
            self.handleError(record)


def logging_redirect_tqdm(logger: logging.Logger, tqdm_class=std_tqdm):
    """tqdm 출력과 로깅을 통합"""
    tqdm_handler = _TqdmLoggingHandler(tqdm_class)
    orig_handler = next((h for h in logger.handlers if isinstance(h, logging.StreamHandler)), None)
    if orig_handler:
        tqdm_handler.setFormatter(orig_handler.formatter)
        tqdm_handler.stream = orig_handler.stream
    logger.handlers = [tqdm_handler] + [h for h in logger.handlers if not isinstance(h, logging.StreamHandler)]
    return logger


@contextmanager
def logging_redirect_tqdm_with_context(loggers=None, tqdm_class=std_tqdm):
    """tqdm 출력과 로깅을 컨텍스트 매니저로 통합"""
    loggers = loggers or [logging.root]
    original_handlers = [logger.handlers for logger in loggers]
    try:
        for logger in loggers:
            logging_redirect_tqdm(logger, tqdm_class)
        yield
    finally:
        for logger, handlers in zip(loggers, original_handlers):
            logger.handlers = handlers
