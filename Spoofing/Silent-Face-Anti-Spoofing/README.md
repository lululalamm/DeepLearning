
# Face Spoofing

- original git : https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/tree/master

## Dataset
- FaceInTheWild + CelebA
- Training or Validation
- scale cropped dataset
    ```bash
    python preprocessing/get_cropImage.py --scale {scale float} --out_w {out_w_size} --out_h {out_h_size}
    ```

## Basic Training
### 1. Training

- Train command
  ```bash
  python train_h5.py --patch_info {scale_outsize str} --data_name {db_name} --h5_format {h5 dataset format path} ...
  ```

### 2. Converting model

- torch2onnx
  ```bash
  python torch2onnx.py --model_path {torch model path} --output_path {output onnx path}
  ```
- onnx2openvino (install openvino-dev)
  ```bash
  mo --input_model {onnx model path} --output_dir {openvino model output dir} --input_shape '[1,3,80,80]'
  ```

### 3. Test

- Test command
  ```bash
  1. test on wild
  python eval/test_scrfd_txt_wild_vino.py --model_path {openvino model path .vino} --scale {scale float} --save_name {test result save name} ...

  2. test on celebA
  python eval/test_scrfd_txt_wcelebAvino.py --model_path {openvino model path .vino} --scale {scale float} --save_name {test result save name} ...

  ```