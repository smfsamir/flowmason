# flowmason
A lightweight orchestration management framework for wrangling messy experiments and encouraging exploration

# Installation
```
git clone https://github.com/smfsamir/flowmason
cd flowmason
pip install -e .
```

Upload to PyPi is forthcoming. 

# Example usage
```
from collections import OrderedDict

# implementations of steps here...

if __name__ == "__main__":
    step_dict = OrderedDict()
    step_dict["download_datasets"] = (step_download_datasets, {
        "version": "001"
    })
    step_dict["train_model"] = (step_train_model, {
        "version": "001",
        "held_out_languages": ("bn_in", "fr_fr", "ar_eg", "ja_jp", "yue_hant_hk"),
        "model_save_dir": "whisper_small_finetune",
        "whisper_size": "small",
        "clear_save_dir": True
    })
    step_dict["construct_eval_partition_frame"] = (step_construct_eval_partition_frame, {
        "version": "001",
        "held_out_languages": ("bn_in", "fr_fr", "ar_eg", "ja_jp", "yue_hant_hk"),
    })
    step_dict["evaluate_model_quantitative"] = (step_quantitative_evaluate_model, {
        "version": "001",
        "checkpoint_path": f"{WHISPER_SAVE_PATH}/whisper_small_finetune/checkpoint-4000",
        "partition_frame": "construct_eval_partition_frame",
        "whisper_size": "small"
    })
    step_dict["evaluate_model_qualitative"] = (step_qualitative_eval, {
        "version": "001",
        "checkpoint_path": f"{WHISPER_SAVE_PATH}/whisper_small_finetune/checkpoint-4000",
        "partition_frame": "construct_eval_partition_frame",
        "whisper_size": "small"
    })
    conduct(os.path.join(SCRATCH_DIR, "anyspeech_flowmason_cache"), step_dict, "anyspeech_experiment_logs") # will execute steps in order and cache results. 
