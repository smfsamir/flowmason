# import ipdb
import pdb
from functools import partial
from dataclasses import dataclass
import hashlib
from typing import Any, Tuple, Callable, Dict, OrderedDict, List, Union
import dill
import datetime 
import loguru
import os
import json

@dataclass
class SingletonStep:
    step_fn: Callable
    step_params: Dict[str, Any]

@dataclass
class MapReduceStep:
    step_fns: OrderedDict[str, SingletonStep]
    map_params: Dict[str, List] 
    constant_params: Dict[str, Any]
    reduce_fn: Callable

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
    cache_name = _get_step_cache_name(step_name, step_version, step_kwargs)

    os.makedirs(cache_dir, exist_ok=True)
    # with open(cache_name, 'wb') as f:
    cache_hashed_name = hashlib.sha256(cache_name.encode()).hexdigest()
    logger.info(f"Caching result of step {step_name} at {cache_hashed_name}")
    with open(os.path.join(cache_dir, str(cache_hashed_name)), 'wb') as f:
        dill.dump(result, f)
    # return the cache path
    return os.path.join(cache_dir, cache_hashed_name)

def load_from_cache(cache_dir, step_name, step_version, step_kwargs):
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

# TODO: need to re-write this so it can check for both types of steps.
def _check_should_execute(curr_step: str, curr_step_arguments: Dict[str, Any], 
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

def step_wrapper(step_func, cache_map: Dict[str, str], cache_dir: str):
    def wrapper(*args, **kwargs):
        step_name = kwargs["step_name"]
        step_version = kwargs["version"]
        # TODO: right now, the step does not get re-executed when one of the dependencies are out of date.
        logger.info(f"Running step {step_name}")

        #### DAG input logic goes here. ####:
        original_kwargs = kwargs.copy()
        for key, value in kwargs.items(): 
            if value == step_name:
                continue
            if value in cache_map: # substitute the value with the result of the step.
                kwargs[key] = dill.load(open(cache_map[value], 'rb'))
        result = step_func(*args, **kwargs)
        if result is not None:
            cache_path = cache_result(cache_dir, step_name, step_version, original_kwargs, result)
            return cache_path, "executed"
        else:
            logger.info(f"Step {step_name} returned None, not caching.")
            return "no result to cache", "executed"
    return wrapper

def execute_map_reduce_step(mapreduce_step_name: str, 
                            map_reduce_step: MapReduceStep, 
                            cache_map: Dict[str, str], cache_dir: str):
    # thing to be careful about: ensure that map_param invariant steps are not cached
    ## in order to do that, we should:
    ### create a different cache directory for map reduce steps
    ### or suffix the cache name with the map param values, for all steps in the map reduce step (regarless of whether they are invariant or not)
    map_reduce_mapdata = []
    map_params = map_reduce_step.map_params
    num_map_param_settings = map_params[list(map_params.keys())[0]]
    final_result_paths = []
    for i in range(len(num_map_param_settings)):
        map_param_setting_cache = {}
        map_kwargs = {k: v[i] for k, v in map_params.items()}
        # add the constant params to the map_kwargs
        map_kwargs = {**map_kwargs, **map_reduce_step.constant_params}
        for singleton_step_name, singleton_step_impl in map_reduce_step.step_fns.items():
            # combine cache map with map_param_setting_cache
            step_fn = step_wrapper(singleton_step_impl.step_fn, 
                                   {**cache_map, **map_param_setting_cache}, # NOTE: there will be an overwrite issue here, if one of the map reduce step was also an external singleton step. But that shouldn't be happening anyway, since step names should be unique.
                                   cache_dir)
            step_version = map_kwargs["version"]
            start_time = datetime.datetime.now().strftime("%H:%M:%S")
            map_kwargs["step_name"] = f"{mapreduce_step_name}_{singleton_step_name}_{i}"
            # call step fn on the union of the map kwargs and the singleton step kwargs, overriding the map kwargs when there is a conflict.
            fn_kwargs = {**map_kwargs, **singleton_step_impl.step_params}
            result_cache_path, execution_status = step_fn(**fn_kwargs) 
            end_time = datetime.datetime.now().strftime("%H:%M:%S")
            metadata = create_metadata(step_version, map_kwargs, start_time, end_time,
                                    cache_dir, execution_status)
            map_reduce_mapdata.append([singleton_step_name, metadata])
            map_param_setting_cache[singleton_step_name] = result_cache_path
        final_result_paths.append(result_cache_path) 
    # use the reduce function to combine the results.
    reduce_fn = map_reduce_step.reduce_fn
    # load all of the results from the final_result_paths using dill.load
    final_results = [dill.load(open(path, 'rb')) for path in final_result_paths]
    final_result = reduce_fn(final_results)
    # create all_map_kwargs by combining constant_params and map_kwargs
    all_map_kwargs = {**map_reduce_step.map_params, **map_reduce_step.constant_params, "step_name": mapreduce_step_name}
    map_reduce_result_cache_path = cache_result(cache_dir, mapreduce_step_name, map_kwargs["version"], all_map_kwargs, final_result)
    return map_reduce_result_cache_path, map_reduce_mapdata
        
def conduct(cache_dir: str, experiment_steps: OrderedDict[str, Union[SingletonStep, MapReduceStep]], experiment_name: str):
    experiment_dir = os.path.join("outputs", experiment_name)
    if not os.path.exists(experiment_dir):
        run_fname = os.path.join(experiment_dir, "run_0000.json")
        os.makedirs(experiment_dir)
    else:
        run_num = len(os.listdir(experiment_dir))
        run_num_str = str(run_num).zfill(4)
        run_fname = os.path.join(experiment_dir, f"run_{run_num_str}.json")
    steps_to_execute = []
    for curr_step_name, curr_step_impl in experiment_steps.items():
        if isinstance(curr_step_impl, SingletonStep):
            should_execute = _check_should_execute(curr_step_name, 
                                                curr_step_impl.step_params, 
                                                cache_dir, 
                                                steps_to_execute)
        elif isinstance(curr_step_impl, MapReduceStep):
            # the params are the combination of the map params and the constant params.
            should_execute = _check_should_execute(curr_step_name,
                                                {**curr_step_impl.map_params, 
                                                 **curr_step_impl.constant_params},
                                                cache_dir,
                                                steps_to_execute)
        else:
            raise ValueError(f"Step {curr_step_name} is not a valid step type.")
        if should_execute:
            steps_to_execute.append(curr_step_name) 

    steps_metadata = []
    cache_map = {}
    for exp_step_name, step_impl in experiment_steps.items(): 
        if exp_step_name not in steps_to_execute:
            cache_name = _get_step_cache_name(exp_step_name, step_impl.step_params['version'], step_impl.step_params)
            hash_name = hashlib.sha256(cache_name.encode()).hexdigest()
            hashed_fcache_name = os.path.join(cache_dir, hash_name)
            logger.info(f"Step {exp_step_name} is cached at {hashed_fcache_name}, continuing.")
            step_kwargs = step_impl.step_params
            step_kwargs["step_name"] = exp_step_name
            metadata = create_metadata(step_impl.step_params['version'], step_impl.step_params, "00:00:00", "00:00:00",
                                    cache_dir, "cached")
            steps_metadata.append((exp_step_name, metadata))
            # add to cache map
            cache_map[exp_step_name] = hashed_fcache_name
            continue
        if isinstance(step_impl, SingletonStep):
            step_fn = step_wrapper(step_impl.step_fn, cache_map, cache_dir)
            step_kwargs = step_impl.step_params
            step_version = step_kwargs["version"]
            start_time = datetime.datetime.now().strftime("%H:%M:%S")
            step_kwargs["step_name"] = exp_step_name
            result_cache_path, execution_status = step_fn(**step_kwargs)
            end_time = datetime.datetime.now().strftime("%H:%M:%S")
            metadata = create_metadata(step_version, step_kwargs, start_time, end_time,
                                    cache_dir, execution_status)
            steps_metadata.append((exp_step_name, metadata))
            cache_map[exp_step_name] = result_cache_path
        elif isinstance(step_impl, MapReduceStep):
            start_time = datetime.datetime.now().strftime("%H:%M:%S")
            result_cache_path, map_red_metadata = execute_map_reduce_step(exp_step_name, step_impl, cache_map, cache_dir)
            end_time = datetime.datetime.now().strftime("%H:%M:%S")
            final_metadata = create_metadata(step_impl.constant_params["version"],
                                             {**step_impl.map_params, **step_impl.constant_params, "step_name": exp_step_name},
                                            start_time, end_time, execution_status="executed",
                                            cache_dir=cache_dir)
            map_red_metadata.append(final_metadata)
            steps_metadata.append([exp_step_name, map_red_metadata])
            cache_map[exp_step_name] = result_cache_path
            
    # write the metadata to a json file.
    with open(run_fname, 'w') as f:
        json.dump(steps_metadata, f, indent=4)