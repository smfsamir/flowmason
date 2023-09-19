import hashlib
from typing import Any, Tuple, Callable, Dict, OrderedDict, List
import dill
import datetime 
import loguru
import os
import json

# TODO: make the cache dir an argument to the conduct function.
CACHE_DIR = "cache"
logger = loguru.logger

def create_metadata(step_version, 
                 step_kwargs, start_time: str, end_time: str,
                 cache_dir: str, execution_status: str):
    cache_name = _get_step_cache_name(step_kwargs['step_name'], step_version, step_kwargs)
    hash_name = hashlib.sha256(cache_name.encode()).hexdigest()
    hashed_fcache_name = os.path.join(cache_dir, hash_name)
    return {
            "version": step_version,
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "start_time": start_time,
            "end_time": end_time,
            "kwargs": step_kwargs,
            "execution_status": execution_status,
            "cache_path": hashed_fcache_name
    }

def cache_result(cache_dir: str, step_name, step_version, step_kwargs, result: Any):
    # TODO: add cache folder
    cache_name = _get_step_cache_name(step_name, step_version, step_kwargs)

    os.makedirs(cache_dir, exist_ok=True)
    # with open(cache_name, 'wb') as f:
    cache_hashed_name = hashlib.sha256(cache_name.encode()).hexdigest()
    logger.info(f"Caching result of step {step_name} at {cache_hashed_name}")
    with open(os.path.join(cache_dir, str(cache_hashed_name)), 'wb') as f:
        dill.dump(result, f)
    # return the cache path
    return os.path.join(cache_dir, cache_name)

def load_from_cache(cache_dir, step_name, step_version, step_kwargs):
    # TODO: add cache folder.
    cache_name = _get_step_cache_name(step_name, step_version, step_kwargs)
    cache_hashed_name = hashlib.sha256(cache_name.encode()).hexdigest()

    try:
        with open(os.path.join(cache_dir, f"{cache_hashed_name}"), 'rb') as f:
            return dill.load(f)
    except FileNotFoundError: # we started using hashed names later on, so we need to check for both.
        with open(os.path.join(cache_dir, cache_name), 'rb') as f:
            return dill.load(f)

def _get_step_cache_name(step_name, step_version, step_kwargs):
    kwargs = step_kwargs.copy()
    # these should always be in there, unless we're doing step
    if "version" in kwargs:
        kwargs.pop("version")
    if "step_name" in kwargs:
        kwargs.pop("step_name")
    # remove k-v pairs where the key ends in "_ignore"
    for k in list(kwargs.keys()):
        if k.endswith("_ignore"):
            kwargs.pop(k)
    kwarg_str = "-".join([f"{k}={v}" for k, v in sorted(kwargs.items())])
    return f"{step_name}-{step_version}-{kwarg_str}.dill" if kwarg_str else f"{step_name}-{step_version}.dill"

def _get_cacheable_cache_name(cacheable_name, cacheable_version, cacheable_kwargs):
    kwargs = cacheable_kwargs.copy()
    # these should always be in there, unless we're doing step
    if "version" in kwargs:
        kwargs.pop("version")
    if "cacheable_name" in kwargs:
        kwargs.pop("cacheable_name")
    # remove any keys that end in "no_cache". 
    for k in list(kwargs.keys()):
        if k.endswith("_no_cache"):
            kwargs.pop(k)

    kwarg_str = "-".join([f"{k}={v}" for k, v in sorted(kwargs.items())]) # NOTE: in the future, we may want to hash this instead of using the string.
    return f"{cacheable_name}-{cacheable_version}-{kwarg_str}.dill" if kwarg_str else f"{cacheable_name}-{cacheable_version}.dill"

def _check_should_execute(curr_step: str, curr_step_arguments: Dict[str, str], 
                          cache_dir: str,
                          previous_steps_to_execute: List[str]):
    cache_name = _get_step_cache_name(curr_step, curr_step_arguments['version'], curr_step_arguments)
    hash_name = hashlib.sha256(cache_name.encode()).hexdigest()
    hashed_fcache_name = os.path.join(cache_dir, hash_name)
    for step in previous_steps_to_execute:
        if step in curr_step_arguments.values():
            return True 
    if not os.path.exists(hashed_fcache_name):
        return True
    return False

# Steps have different behaviour, they're discrete units with their own metadata 
def meta_step(steps: OrderedDict[str, Tuple[Callable, Dict[str, str]]], 
              cache_dir: str) -> Tuple[Any, str]: 
    steps_to_execute = []
    for step in steps:
        should_execute = _check_should_execute(step, steps[step][1], cache_dir, steps_to_execute)
        if should_execute:
            steps_to_execute.append(step)
    # print which steps are going to be executed and which are cached.
    logger.info(f"Steps that will be executed: {steps_to_execute}")
    logger.info(f"Steps that are cached: {[step for step in steps if step not in steps_to_execute]}")
    
    def step(step_func):
        def wrapper(*args, **kwargs):
            step_name = kwargs["step_name"]
            step_version = kwargs["version"]
            if step_name not in steps_to_execute:
                cache_name = _get_step_cache_name(step_name, step_version, kwargs)
                hash_name = hashlib.sha256(cache_name.encode()).hexdigest()
                hashed_fcache_name = os.path.join(cache_dir, hash_name)
                logger.info(f"Step {step_name} is cached at {hashed_fcache_name}, continuing.")
                return hashed_fcache_name, "cached"
            # TODO: right now, the step does not get re-executed when one of the dependencies are out of date.
            logger.info(f"Running step {step_name}")

            #### DAG input logic goes here. ####:
            all_steps = steps.keys()
            original_kwargs = kwargs.copy()
            for key, value in kwargs.items(): 
                if value == step_name:
                    continue
                if value in all_steps: # substitute the value with the result of the step.
                    value_step_name = value
                    kwargs[key] = load_from_cache(cache_dir, value_step_name, steps[value_step_name][1]['version'], steps[value_step_name][1])
            result = step_func(*args, **kwargs)
            if result is not None:
                cache_path = cache_result(cache_dir, step_name, step_version, original_kwargs, result)
                return cache_path, "executed"
            else:
                logger.info(f"Step {step_name} returned None, not caching.")
                return "no result to cache", "executed"
        return wrapper
    return step

# cacheables can only be used within a step. They cannot have dependencies on previous steps, only the current step.
    # they also are only stored in the cache dir and not sim linked to the experiment folder 
def cacheable(cache_dir: str):
    def cacheable_wrapper(func):
        def wrapper(*args, **kwargs):
            # TODO: we need to know the name of the step, the version of the step, and the kwargs of the step.
            # TODO: check if the function is cached. If it is, load the result and return it.
            assert "cacheable_name" in kwargs, "cacheable_name must be in kwargs"
            assert "version" in kwargs, "version must be in kwargs"
            cacheable_name = kwargs["cacheable_name"]
            logger.info(f"Running cacheable {cacheable_name}")
            cacheable_version = kwargs["version"]
            cache_name = _get_cacheable_cache_name(cacheable_name, cacheable_version, kwargs)
            fcache_name = os.path.join(cache_dir, cache_name)
            if os.path.exists(fcache_name):
                logger.info(f"Cacheable {cacheable_name} is cached, loading.")
                with open(fcache_name, 'rb') as f:
                    return dill.load(f)
            else:
                result = func(*args, **kwargs)
                os.makedirs(cache_dir, exist_ok=True)
                # with open(cache_name, 'wb') as f:
                with open(os.path.join(cache_dir, cache_name), 'wb') as f:
                    dill.dump(result, f)
                return result
        return wrapper
    return cacheable_wrapper
        
def conduct(cache_dir: str, experiment_steps: OrderedDict, experiment_name: str):
    step_wrapper = meta_step(experiment_steps, cache_dir)
    experiment_dir = os.path.join("outputs", experiment_name)
    if not os.path.exists(experiment_dir):
        run_fname = os.path.join(experiment_dir, "run_0000.json")
        os.makedirs(experiment_dir)
    else:
        run_num = len(os.listdir(experiment_dir))
        run_num_str = str(run_num).zfill(4)
        run_fname = os.path.join(experiment_dir, f"run_{run_num_str}.json")
    steps_metadata= []
    for step_name in experiment_steps:
        # step_name, step_fn, step_kwargs = step
        step_fn = step_wrapper(experiment_steps[step_name][0])
        step_kwargs = experiment_steps[step_name][1]
        step_version = step_kwargs["version"]
        start_time = datetime.datetime.now().strftime("%H:%M:%S")
        step_kwargs["step_name"] = step_name
        result_cache_path, execution_status = step_fn(**step_kwargs)
        end_time = datetime.datetime.now().strftime("%H:%M:%S")
        metadata = create_metadata(step_version, step_kwargs, start_time, end_time,
                                   cache_dir, execution_status)
        steps_metadata.append((step_name, metadata)) 
    # write the metadata to a json file.
    with open(run_fname, 'w') as f:
        json.dump(steps_metadata, f, indent=4)