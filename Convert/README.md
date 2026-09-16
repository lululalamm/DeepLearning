# Convert

학습한 모델을 배포/추론 환경에 맞게 다른 포맷으로 변환하는 스크립트 모음입니다.

| 스크립트 | 변환 | 비고 |
| --- | --- | --- |
| `mxnet2onnx.py` | MXNet → ONNX | MXNet 학습 모델을 ONNX로 |
| `onnx_simplify.py` | ONNX → ONNX | `onnx-simplifier`로 그래프 단순화 |
| `onnx2trt_v7.py` | ONNX → TensorRT | TensorRT 7.x |
| `onnx2trt_v8.py` | ONNX → TensorRT | TensorRT 8.x |
| `onnx2trt_multiple.py` | ONNX → TensorRT | 여러 모델 일괄 변환 |
| `onnx2openvino.sh` | ONNX → OpenVINO | OpenVINO `mo` 래퍼 |
| `onnx2tflite.py` | ONNX → TFLite | 모바일/엣지 배포용 |
| `openvino_quantization.py` | OpenVINO 양자화 | INT8 등 양자화 |

## 사용 예시

```bash
# ONNX 단순화
python onnx_simplify.py --input {model.onnx} --output {model_sim.onnx}

# ONNX → TensorRT (버전에 맞는 스크립트 사용)
python onnx2trt_v8.py --input {model.onnx} --output {model.trt}

# ONNX → OpenVINO
bash onnx2openvino.sh
```

각 스크립트의 인자는 파일 상단 `argparse` 정의를 참고하세요. TensorRT 버전(7/8)에 따라 API가 다르므로 환경에 맞는 스크립트를 선택합니다.
