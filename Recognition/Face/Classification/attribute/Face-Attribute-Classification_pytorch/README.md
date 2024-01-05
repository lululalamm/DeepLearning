# Face Expression

- original git : https://github.com/Malikanhar/Face-Attribute-Classification
- The original git is tensorflow code, so the code was modified with pytorch


## Dataset
- CelebA (open dataset)

    1. aligned using src/aligned224_add_pool.py
    2. Make .h5 file using src/make_h5.py


## Basic Training

## Config
- Create a config file within the configs directory

### 1. Training

- Train command
  ```bash
  python src/train.py --cfg_path {config path}
  ```

### 2. Test

- Test command (db: wild, celebA, casia)
  ```bash
  python src/test.py --cfg_path {config path} --model_path {model dir or model pth path} --test_format {h5 or csv} --test_base {test data path} --false_save
  ```
  - If you add false_save and run it, the result and data path where the prediction result is false will be saved as npy

### 3. Converting model

- torch2onnx
  : torch2onnx.ipynb 
- onnx2openvino (install openvino-dev)
  ```bash
  mo --input_model {onnx model path} --output_dir {openvino model output dir} --input_shape '[1,3,224,224]'
  ```
