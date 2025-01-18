from at3s.network import NetworkEnv
import yaml
import logging
import logging.config
import os
import shutil
from morl_baselines.multi_policy.capql.capql import CAPQL
import time
import random

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time

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

class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, file_to_watch, env):
        self.file_to_watch = file_to_watch
        self.env = env

    def on_modified(self, event):
        if event.src_path.endswith(self.file_to_watch):
            logger.info(f"The file '{self.file_to_watch}' has been modified.")
            action = self.read_action_from_file()
            logger.info("=======NEW ACTION: %s=======", action)
            self.env.step(action)

    def read_action_from_file(self):
        with open(file_to_watch, "r") as file:
            # Read the first line from the file
            line = file.readline()

        # Try to extract the first float from the line
        first_float = None
        for word in line.split():
            try:
                first_float = float(word)
                break  # Exit the loop after finding the first float
            except ValueError:
                continue  # Ignore non-float words

        if first_float is not None:            
            return first_float
        else:            
            return 0.0





def watch_file(file_path, env):
    event_handler = FileChangeHandler(file_to_watch=file_path, env=env)
    observer = Observer()
    observer.schedule(event_handler, path=".", recursive=False)

    try:
        print(f"Watching file: {file_path}")
        observer.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

step_second = 20
env = NetworkEnv("at3s/generator.yaml", "at3s/processor.yaml", False, step_second)
max_action = len(env.get_traffic_classes())
file_to_watch = "action.txt"
watch_file(file_to_watch, env)