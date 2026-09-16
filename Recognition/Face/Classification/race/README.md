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
  python test_result_mbn.py --model_path {model_path} --save_name {save_result_name} --load_type {backbone load_type}

  # mobileFacenet
  python test_result.py --model_path {model_path} --save_name {save_result_name}
  ```
  - 결과는 `--result_base`(기본 `./test_result/`) 아래 `--save_name`으로 저장되며, 저장된 결과를 이용해 정확도를 평가합니다.