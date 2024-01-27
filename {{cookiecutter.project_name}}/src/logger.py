import sys
import logging


GREEN_COLOR_CODE = '\x1b[38;5;34m'
GREY_COLOR_CODE = '\x1b[38;20m'
YELLOW_COLOR_CODE = '\x1b[33;20m'
RED_COLOR_CODE = '\x1b[31;20m'
BOLD_RED_COLOR_CODE = '\x1b[31;1m'
RESET_COLOR_CODE = '\x1b[0m'


class CustomFormatter(logging.Formatter):
    format_str = '%(asctime)s %(name)s %(levelname)s %(message)s'  # noqa: WPS323

    formats_dict = {
        logging.DEBUG: GREEN_COLOR_CODE + format_str + RESET_COLOR_CODE,
        logging.INFO: GREY_COLOR_CODE + format_str + RESET_COLOR_CODE,
        logging.WARNING: YELLOW_COLOR_CODE + format_str + RESET_COLOR_CODE,
        logging.ERROR: RED_COLOR_CODE + format_str + RESET_COLOR_CODE,
        logging.CRITICAL: BOLD_RED_COLOR_CODE + format_str + RESET_COLOR_CODE,
    }

    def format(self, record):
        log_fmt = self.formats_dict.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def init_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Configure handlers and formatters
    file_handler = logging.FileHandler('logs/admin_logs.log', mode='a')
    std_handler = logging.StreamHandler(sys.stdout)
    formatter = CustomFormatter()

    file_handler.setFormatter(formatter)
    std_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(std_handler)

    return logger