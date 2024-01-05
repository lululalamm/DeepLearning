# Face Mask Classification

- backbones : mnasnet, mobilenetv3, mobileFacenet



## Dataset
- Dataset
    - Aihub FaceMask Dataset

    - rec build
      1. preprocessing/align/new_align_230411.py ( use csv )
      2. preprocessing/new_build_rec_230411.py ( aligned -> rec )

## Basic Training
### 1. Training



- Train command

  ```bash
  python train_rec.py --model_name {backbone name} --rec_path {dataset path} --data_aug --save_name {save result dir name} --input_size {input image size} --load_type {backbone load type}
  ```

### 2. Test

- Test command
  ```bash
  1. aihub FaceMask test
  python test.py --model_name {backbone name} --model_path {model_path} --test_path {test csv path} --image_base {test image dir} --input_size {input image size} --net_size {if mnasnet, net size set}

  2. mfr2 test
  python test_mfr2.py --model_name {backbone name} --model_path {model_path} --test_path {test csv path} --image_base {test image dir} --input_size {input image size} --net_size {if mnasnet, net size set}
  ```


### 3. converting
- torch2onnx
```bash
python torch2onnx.py --input {torch model path} --output {onnx output path} --model_name {backbone name}
```


### 4. speed check
```bash
1. torch model
python speed_check.py --model_name {backbone name} --model_path {torch model path}

2. converted model (tensorRT, openvino)
python speed_check_cvt.py --model_path {converted model path} --model_type {trt or vino}
```
