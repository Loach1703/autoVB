#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import sys

from ..utils.constants import VERSION
from ..io.logging_config import configure_logging, get_logger
from ..main import autoVBMain
from ..io.readers import autoVBInputParser

logger = get_logger(__name__)


def build_autovb_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autovb",
        description="Run autoVB from an .autovb, Gaussian, or XMVB input file.",
    )
    parser.add_argument("input_file", type=Path, help="input file")
    parser.add_argument(
        "--mem",
        default=None,
        help="memory for external programs, for example 4GB, 8G, or 4000MB",
    )
    parser.add_argument(
        "--nproc",
        default=None,
        help="number of processors for external programs",
    )
    return parser


def normalize_memory(mem: str) -> str:
    if mem.lower().endswith("g"):
        return mem[:-1] + "GB"
    if mem.lower().endswith("m"):
        return mem[:-1] + "MB"
    if mem.isdigit():
        return mem + "MB"
    return mem


def autovb_main(argv=None):
    arg_parser = build_autovb_parser()
    # 配置输出日志
    configure_logging()
    logger.info("Welcome to autoVB! Version %s", VERSION)
    argv = list(argv) if argv is not None else sys.argv[1:]
    if not argv:
        logger.error("Usage: autovb <input-file> [--mem MEM] [--nproc NPROC]")
        return 2

    args = arg_parser.parse_args(argv)
    input_file: Path = args.input_file
    resolved = input_file.resolve()
    if not resolved.exists():
        logger.error("input file not found: %s", input_file)
        return 2
    # 输入文件解析
    input_parser = autoVBInputParser(input_file)
    debug = input_parser.input_data.debug
    configure_logging(level=logging.DEBUG if debug else logging.INFO)

    # 命令行参数优先级最高，其次是输入文件参数，最后是默认值
    mem = args.mem or input_parser.input_data.mem or "4GB"
    nproc = args.nproc or input_parser.input_data.nproc or "1"
    mem = normalize_memory(mem)
    logger.info("Using memory: %s, nproc: %s", mem, nproc)
    input_parser.input_data.mem = mem
    input_parser.input_data.nproc = nproc

    main_obj = autoVBMain(input_parser.input_data)
    try:
        main_obj.main()
    except Exception as e:
        logger.error("An error occurred: %s", e)
        raise
    return 0
