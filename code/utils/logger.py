import logging
import os
from accelerate import Accelerator

def setup_logger(log_dir: str, name: str, accelerator = None) :

    """
    Create and configure a logger.

    The logger writes messages to the console on every process and, when
    running in a distributed environment, also writes them to a log file
    from the main process only.

    Args:

        log_dir: Directory where the log file will be stored.
        name: Name of the log file (without extension).
        accelerator: accelerator: Hugging Face Accelerator instance used to determine
            whether the current process is the main process. If ``None``,
            logging is configured only for console output.

    Returns:
        A configured ``logging.Logger`` instance.

    """
    os.makedirs(log_dir, exist_ok=True)
    handlers = [logging.StreamHandler()]

    if accelerator is not None and accelerator.is_main_process:
        handlers.insert(0, logging.FileHandler(os.path.join(log_dir, f"{name}.log")))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers
    )

    return logging.getLogger(__name__)


