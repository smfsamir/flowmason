import shutil
import ipdb
import os
import pytest
from collections import OrderedDict
from flowmason.dag import conduct, MapReduceStep, SingletonStep

def _step_toy_fn(step_name, version, 
                 arg1: float):
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

def test_map_reduce_steps(cache_dir):
    step_dict = OrderedDict()
    print(f"Cache dir: {cache_dir}")
    step_dict['step_map_reduce'] = MapReduceStep(
        {
            "step_toy_fn": _step_toy_fn 
        },
        {
            "arg1": [2.9, 3.0, 3.1], 
        }, 
        {"version": "001"}, 
    sum)
    ipdb.set_trace()
    with open("tests/outputs/test_orchestration/run_0000.json", 'r') as f:
        print(f.read())
    conduct(cache_dir, step_dict, "test_orchestration")

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
    # TODO: use the singleton step output as an argument to 
    # the map reduce step.
    conduct()

def test_parameter_passing():
    def my_function(parameter, **kwargs):
        print(parameter)
        return parameter
    assert my_function(parameter="hello") == "hello"
    assert my_function(**{"parameter": "hello"}) == "hello"
    assert my_function("hello", x=3)