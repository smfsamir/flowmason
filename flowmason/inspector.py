from typing import Iterable
import ipdb
import dill
from typing import Tuple, Dict
import json
import os
import loguru

logger = loguru.logger
def load_latest_steps(experiment_name: str):
    # load the latest file. It will be run_####.json under
    # outputs/{experiment_name}
    # the latest one will have the highest number.
    # code:
    fname = max([f for f in os.listdir(os.path.join("outputs", experiment_name)) if f.startswith("run_")])
    with open(os.path.join("outputs", experiment_name, fname), 'r') as f:
        return json.load(f)

def load_artifact(step: Tuple[str, Dict[str, str]]):
    artifact_path = step[1]["cache_path"]
    with open(artifact_path, 'rb') as f:
        return dill.load(f)
    
def load_mr_artifact(step: Tuple[str, Dict[str, str]]):
    artifact_path = step[1][-1]["cache_path"]
    with open(artifact_path, 'rb') as f:
        return dill.load(f)

def load_artifact_with_step_name(metadata, step_name, is_mr_step = False):
    for step in metadata:
        if is_mr_step:
            if step[0][0] == step_name:
                return load_mr_artifact(step)
        else:
            if step[0] == step_name:
                return load_artifact(step)
    logger.error(f"Step {step_name} not found in metadata")
    return -1

def get_all_files_in_metadata(metadata: Iterable):
    cache_paths = []
    for cache_item in metadata:
        if isinstance(cache_item, tuple): # singleton step  
            cache_paths.append(cache_item[1]["cache_path"])
        elif isinstance(cache_item, list): # map reduce step
            for i in range(len(cache_item[1]) - 1): # NOTE: 0 is the name of the map reduce step
                # i is the index over the map items
                cache_paths.append(cache_item[1][i][1]["cache_path"])
            cache_paths.append(cache_item[1][-1]["cache_path"])
    return cache_paths

