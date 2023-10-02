import dill
from typing import Tuple, Dict
import json
import os

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