import os
import logging as py_logging
import argparse
from distutils.util import strtobool


_log_level = {
    None: py_logging.NOTSET,
    "debug": py_logging.DEBUG,
    "info": py_logging.INFO,
    "warning": py_logging.WARNING,
    "error": py_logging.ERROR,
    "critical": py_logging.CRITICAL
}


def get_logger(
    log_file_path=None,
    name="default_log",
    level=None
):
    directory = os.path.dirname(log_file_path)
    if os.path.isdir(directory) and not os.path.exists(directory):
        os.makedirs(directory)

    root_logger = py_logging.getLogger(name)
    handlers = root_logger.handlers

    def _check_file_handler(logger, filepath):
        for handler in logger.handlers:
            if isinstance(handler, py_logging.FileHandler):
                handler.baseFilename
                return handler.baseFilename == os.path.abspath(filepath)
        return False

    if (log_file_path is not None and not
            _check_file_handler(root_logger, log_file_path)):
        log_formatter = py_logging.Formatter(
            "%(asctime)s [%(levelname)-5.5s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S")
        file_handler = py_logging.FileHandler(log_file_path)
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)
    if any([type(h) == py_logging.StreamHandler for h in handlers]):
        return root_logger
    level_format = "\x1b[36m[%(levelname)-5.5s]\x1b[0m"
    log_formatter = py_logging.Formatter(f"{level_format} %(message)s")
    console_handler = py_logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(_log_level[level])
    return root_logger


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--env-name", type=str, default="highway-v0",
        help="env-name of the experiment")
    parser.add_argument("--model-time", type=str, default="",
        help="time of the eval model")
    parser.add_argument("--total-timesteps", type=int, default=1e5,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e4),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batchsize", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=0,
        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=1e-3,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=1e-3,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--alpha", type=float, default=0.02,
            help="Entropy regularization coefficient.")
    parser.add_argument("--adaptive-alpha", type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    args = parser.parse_args()
    # fmt: on
    return args
