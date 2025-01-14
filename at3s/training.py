from network import NetworkEnv
import yaml
import logging
import logging.config
import os
import shutil
from morl_baselines.capql import CAPQL
from morl_baselines.common.evaluation import evaluate
from morl_baselines.common.utils import save_results

# Create work dir
if os.path.exists("./result"):
    shutil.rmtree("./result")
os.mkdir("./result")
os.environ["KERAS_BACKEND"] = "tensorflow"

# Config logging
with open("./at3s/logging_config.yaml", "r") as f:
    config = yaml.safe_load(f.read())
logging.config.dictConfig(config)
logger = logging.getLogger("my_logger")

logger.info("Hello")
env = NetworkEnv("at3s/generator.yaml", "at3s/processor.yaml", False)
action = [0.5, 0.5];
env.step(action)
