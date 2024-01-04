# Face Race - mobile backbone

- use backbones : mobilenetv3-small , mobilenetv3-large, mobileFacenet


## Dataset
- FairFace


## Basic Training
### 1. Training

- Config
  - Need to write config file within configs directory

- Train command
  ```bash
  # mobilenetv3 (small or large)
  python train_mbn.py --config {config_path}

  # mobileFacenet
  python train.py --config {config_path}
  ```

### 2. Test

- Test command
  ```bash
  # mobilenetv3 (small or large)
  1. get results
  python test_result_mbn.py --model_path {model_path} --save_name {save_result_name} --load_type {backbone load_type}
  2. eval
  python test_race_new.py --result_base {result_directory_path}


  # mobileFacenet
  1. get results
  python test_result.py --model_path {model_path} --save_name {save_result_name}
  2. eval
  python test_race_new.py --result_base {result_directory_path}

  ```