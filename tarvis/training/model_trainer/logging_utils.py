import logging
import re

import tarvis.utils.distributed as dist_utils


class FormatterWithANSIEscapeSequenceRemoval(logging.Formatter):
    """
    Formatter which removes ANSI escape sequences from the text given to it. This includes e.g. color codes and other
    character sequences which can be parsed on a console window but not by a text editor.
    """
    def __init__(self, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)

        # regex for removing ANSI escape sequences (e.g. color codes)
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def format(self, record):
        record.msg = self.ansi_escape.sub('', record.msg)
        return super().format(record)


def create_console_logger(main_log_level, subprocess_log_level, file_output_path=None):
    logger = logging.getLogger("Trainer")
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    formatter = logging.Formatter("[%(proc_id)d] %(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S")
    ch.setFormatter(formatter)

    if dist_utils.is_main_process():
        ch.setLevel(main_log_level)
    else:
        ch.setLevel(subprocess_log_level)

    logger.addHandler(ch)

    if file_output_path is not None:
        fh = logging.FileHandler(file_output_path, 'w')
        # formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S")
        formatter = FormatterWithANSIEscapeSequenceRemoval("%(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.propagate = False

    extra = {"proc_id": dist_utils.get_rank()}

    logger = logging.LoggerAdapter(logger, extra)
    logger.propagate = False

    return logger
