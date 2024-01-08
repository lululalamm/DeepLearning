# Face Liveness

- original git : https://github.com/kprokofi/light-weight-face-anti-spoofing


## Dataset
- Training or Validation

    1. crop (based on FaceInTheWild Dataset)
    ``` bash 
    python crop_pool.py --base {wild list txt path} --prefix {shared prefix path} --tv {Training   Validation ...} --save_base {save dir} --dt_name {detection model name} --dt_path {detection model path}
    ```
    2. Create list csv: Create csv by referring to list_txt.ipynb
    3. Edit datasets/database.py and create a class function for dataset


## Basic Training

## Config
- Create a config file in configs/
- Refer to config_large_01_wildcelebA.py

### 1. Training

- Train command
  ```bash
  python train.py --config {config path}
  ```

### 2. Converting model

- torch2onnx
  ```bash
  python convert/torch2onnx.py --config {config path} --model_path {torch model path} --output_path {output onnx path}
  ```
- onnx2openvino (need to install openvino-dev)
  ```bash
  mo --input_model {onnx model path} --output_dir {openvino model output dir} --input_shape '[1,3,128,128]'
  ```

### 3. Test

- Test command (db: wild, celebA, casia)
  ```bash
  python eval/eval_csv_cropped.py --model_path {openvino model path .vino} --db {db name}
  ```