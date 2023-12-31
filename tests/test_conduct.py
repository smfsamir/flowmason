import math
import dill
import json
import pdb
import shutil
import ipdb
import os
import pytest
from collections import OrderedDict
from flowmason.dag import conduct, MapReduceStep, SingletonStep

def _step_toy_fn(step_name, version, 
                 arg1: float):
    print(arg1)
    return 3.1 + arg1

def test_conduct():
    assert True

@pytest.fixture
def cache_dir():
    print("Creating cache dir")
    cache_dir = "tests/test_cache"
    os.makedirs(cache_dir, exist_ok=True)
    yield cache_dir
    shutil.rmtree(cache_dir)
    shutil.rmtree("outputs/test_orchestration")

def test_map_reduce_steps(cache_dir):
    step_dict = OrderedDict()
    print(f"Cache dir: {cache_dir}")
    map_reduce_dict = OrderedDict()
    map_reduce_dict['step_toy_fn'] = SingletonStep(_step_toy_fn, {
        'version': '001'
    })
    step_dict['step_map_reduce'] = MapReduceStep(
        map_reduce_dict,
        {"arg1": [2.9, 3.0, 3.1]}, 
        {"version": "001"},
    sum)
    conduct(cache_dir, step_dict, "test_orchestration")
    with open("outputs/test_orchestration/run_0000.json", 'r') as f:
        obj = json.load(f)
        assert dill.load(open(obj[0][1][-1]['cache_path'], 'rb')) == sum([3.1 + 2.9, 3.1 + 3.0, 3.1 + 3.1])

def test_singleton_step_chain(cache_dir):
    step_dict = OrderedDict()
    step_dict['step_singleton'] = SingletonStep(_step_toy_fn, {
        'version': "001", 
        'arg1': 2.9
    })
    step_dict['step_singleton_two'] = SingletonStep(_step_toy_fn, {
        'version': "001", 
        'arg1': 'step_singleton'
    })
    conduct(cache_dir, step_dict, "test_orchestration")
    with open("outputs/test_orchestration/run_0000.json", 'r') as f:
        json_obj = json.load(f)
        # check if dill.load(open(json_obj[1]['cache_path'], 'rb')) == 3.1 + 2.9 + 3.1
        assert dill.load(open(json_obj[1][1]['cache_path'], 'rb')) == 3.1 + 2.9 + 3.1
    
def test_map_reduce_multistep(cache_dir): 
    step_dict = OrderedDict()
    print(f"Cache dir: {cache_dir}")
    map_reduce_dict = OrderedDict()
    map_reduce_dict['step_toy_fn'] = SingletonStep(_step_toy_fn, {
        'version': '001'
    })
    map_reduce_dict['step_toy_fn_two'] = SingletonStep(_step_toy_fn, {
        'version': '001',
        'arg1': 'step_toy_fn'
    })
    step_dict['step_map_reduce'] = MapReduceStep(
        map_reduce_dict,
        {"arg1": [2.9, 3.0, 3.1]}, 
        {"version": "001"},
    sum)
    conduct(cache_dir, step_dict, "test_orchestration")
    # TODO: need to fix the metadata to include assertions
    with open("outputs/test_orchestration/run_0000.json", 'r') as f:
        obj = json.load(f)
        # 0 is the first step, 1 is the metadata, -1 is he metadata specifically for the map reduce step.
        assert abs(dill.load(open(obj[0][1][-1]['cache_path'], 'rb')) - sum([(3.1 + 2.9) + 3.1, (3.1 + 3.0) + 3.1, (3.1 + 3.1) + 3.1])) < 1e-6

def test_singleton_step(cache_dir):
    step_dict = OrderedDict()
    step_dict['step_singleton'] = SingletonStep(_step_toy_fn, {
        'version': "001", 
        'arg1': 2.9
    })
    conduct(cache_dir, step_dict, "test_orchestration")

def test_composite_singleton_map_reduce_steps(cache_dir):
    step_dict = OrderedDict()
    step_dict['step_singleton'] = SingletonStep(_step_toy_fn, {
        'version': "001", 
        'arg1': 2.9
    })
    conduct(cache_dir, step_dict, "test_orchestration")

def test_parameter_passing():
    def my_function(parameter, **kwargs):
        print(parameter)
        return parameter
    
    def my_function_two(parameter):
        return parameter
    assert my_function(parameter="hello") == "hello"
    assert my_function(**{"parameter": "hello"}) == "hello"
    assert my_function("hello", x=3)

# TODO: fill in
def test_singleton_step_caching(cache_dir):
    # create a singleton step. Then, run conduct once. Check that the
    # function execution status is "executed" by checking the metadata.
    # Then, run conduct again. Check that the function execution status
    # is "cached" in "run_0001.json"
    singleton_step = SingletonStep(_step_toy_fn, {
        'version': "001", 
        'arg1': 2.9
    })
    step_dict = OrderedDict()
    step_dict['step_singleton'] = singleton_step
    conduct(cache_dir, step_dict, "test_orchestration")
    with open("outputs/test_orchestration/run_0000.json", 'r') as f:
        obj = json.load(f)
        assert obj[0][1]['execution_status'] == "executed"
    conduct(cache_dir, step_dict, "test_orchestration")
    with open("outputs/test_orchestration/run_0001.json", 'r') as f:
        obj = json.load(f)
        # assert obj[0][1]['execution_status'] == "cached"

# TODO: fill in
def test_map_reduce_step_caching(cache_dir):
    # create a map reduce step. Then, run conduct once. Check that the
    # function execution status is "executed" by checking the metadata.
    # Then, run conduct again. Check that the function execution status
    # is "cached" in "run_0001.json"
    map_reduce_step = MapReduceStep(
        {"step_toy_fn": SingletonStep(_step_toy_fn, {
            'version': '001'
        }), 
        "step_toy_fn_two": SingletonStep(_step_toy_fn, {
            'version': '001',
            'arg1': 'step_toy_fn'
        })},
        {"arg1": [2.9, 3.0, 3.1]}, 
        {"version": "001"}, 
        sum)
    step_dict = OrderedDict()
    step_dict['step_map_reduce'] = map_reduce_step
    conduct(cache_dir, step_dict, "test_orchestration")
    with open("outputs/test_orchestration/run_0000.json", 'r') as f:
        obj = json.load(f)
        assert obj[0][1][-1]['execution_status'] == "executed"
    conduct(cache_dir, step_dict, "test_orchestration")
    with open("outputs/test_orchestration/run_0001.json", 'r') as f:
        obj = json.load(f)
        assert obj[0][1]['execution_status'] == "cached"

def test_map_reduce_should_execute_updated_singletons(cache_dir):
    # create a map reduce step with two singletons. Run conduct once.
    # then, change the second singleton's version. Run conduct again.
    # Check that the second singleton's execution status is "executed", 
    # while the first singleton's execution status is "cached"

    map_reduce_step = MapReduceStep(
        {"step_toy_fn": SingletonStep(_step_toy_fn, {
            'version': '001'
        }), 
        "step_toy_fn_two": SingletonStep(_step_toy_fn, {
            'version': '001',
            'arg1': 'step_toy_fn'
        })},
        {"arg1": [2.9, 3.0, 3.1]}, 
        {"version": "001"}, 
        sum)
    step_dict = OrderedDict()
    step_dict['step_map_reduce'] = map_reduce_step
    conduct(cache_dir, step_dict, "test_orchestration")
    with open("outputs/test_orchestration/run_0000.json", 'r') as f:
        obj = json.load(f)
        first_step_index = 0
        second_step_index = 1
        assert obj[0][1][first_step_index][1]['execution_status'] == "executed"
        assert obj[0][1][second_step_index][1]['execution_status'] == "executed"
    
    map_reduce_step = MapReduceStep(
        {"step_toy_fn": SingletonStep(_step_toy_fn, {
            'version': '001'
        }),
        "step_toy_fn_two": SingletonStep(_step_toy_fn, {
            'version': '002',
            'arg1': 'step_toy_fn'
        })},
        {"arg1": [2.9, 3.0, 3.1]}, 
        {"version": "001"},
        sum)
    step_dict = OrderedDict()
    step_dict['step_map_reduce'] = map_reduce_step
    conduct(cache_dir, step_dict, "test_orchestration")

    with open("outputs/test_orchestration/run_0001.json", 'r') as f:
        obj = json.load(f)
        first_step_index = 0
        second_step_index = 1
        # TODO: need to fix this
        assert obj[0][1][first_step_index][1]['execution_status'] == "cached"
        assert obj[0][1][second_step_index][1]['execution_status'] == "executed"
