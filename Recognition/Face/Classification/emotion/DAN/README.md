# Face Emotion - DAN

- original git : https://github.com/yaoing/DAN


## Dataset
- alinged 112x112 , h5 format


## Basic Training
### 1. Training

- Train command
  ```bash
  python train_aihub.py --save_base {save_directory} --input_size 112 --batch_size 64 --num_classes 5 
  ```

### 2. torch2onnx

  - Run "torch2onnx.ipynb"


### 3. Test

- Test command
  ```bash
  python demo.py {image file or directory} --model {model_path}
  ```



