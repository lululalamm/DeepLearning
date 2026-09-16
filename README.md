# DeepLearning

얼굴(Face)과 사람(Human)을 중심으로 한 딥러닝 연구/실험 코드 모음입니다.
탐지(Detection)·인식(Recognition)·위변조 판별(Spoofing)·추적(Tracking)의 학습·평가·모델 변환 코드와,
이를 손쉽게 추론에 쓸 수 있도록 묶은 파이썬 패키지(`totalface`, `totalhuman`)를 포함합니다.

## 목차
- [저장소 구조](#저장소-구조)
- [영역별 개요](#영역별-개요)
  - [Detection](#detection)
  - [Recognition](#recognition)
  - [Spoofing](#spoofing)
  - [Tracking](#tracking)
  - [Convert](#convert)
  - [Package](#package)
- [설치 패키지](#설치-패키지)
- [참고](#참고)

## 저장소 구조

```
DeepLearning/
├── Detection/      얼굴 탐지 · 마스크 착용/구간 분류
├── Recognition/    얼굴 임베딩(ArcFace) · 속성/감정/인종 분류
├── Spoofing/       얼굴 위변조(Face Anti-Spoofing) 판별
├── Tracking/       사람 추적(HiFT, SiamMask)
├── Convert/        모델 포맷 변환 스크립트 (ONNX/TensorRT/OpenVINO/TFLite)
└── package/        추론용 파이썬 패키지 (totalface / totalhuman)
```

각 하위 폴더에는 학습·테스트·변환 방법을 담은 자체 `README.md`가 있습니다.

## 영역별 개요

### Detection
`Detection/Face/`
- **maskedFace** — 얼굴 마스크 착용 여부 분류. 백본: MnasNet, MobileNetV3, MobileFaceNet. AI Hub Face Mask 데이터셋 사용. 학습/테스트/`torch→onnx` 변환/속도 측정 스크립트 포함. → [README](Detection/Face/maskedFace/README.md)
- **maskWearing** — 얼굴 랜드마크 기반 마스크 착용 위치 추론(`mask_landmark.ipynb`). `totalface` 패키지 필요. → [README](Detection/Face/maskWearing/README.md)

### Recognition
`Recognition/Face/`
- **Embedding** — 얼굴 인증용 임베딩 학습(ArcFace 계열, 다양한 loss 포함).
- **Classification/**
  - **attribute** — 얼굴 속성 분류. CelebA 기반 Face-Attribute-Classification(PyTorch 이식판)과 Conditional Multitask Learning(`arcface-cmt`, 임베딩 백본을 pretrain으로 다중 속성 동시 학습). → [attribute README](Recognition/Face/Classification/attribute/Face-Attribute-Classification_pytorch/README.md), [arcface-cmt README](Recognition/Face/Classification/attribute/arcface-cmt-pytorch/README.md)
  - **emotion** — 표정 분류(DAN 기반, 112×112 정렬 입력). → [README](Recognition/Face/Classification/emotion/DAN/README.md)
  - **race** — 인종 분류(FairFace, MobileNetV3-small/large·MobileFaceNet 백본). → [README](Recognition/Face/Classification/race/README.md)

### Spoofing
`Spoofing/` — 얼굴 위변조(Face Anti-Spoofing) 판별
- **SAFAS** — CVPR 2023 "Rethinking Domain Generalization for Face Anti-spoofing" 구현. → [README](Spoofing/SAFAS/README.md)
- **light-fas** — 경량 FAS([light-weight-face-anti-spoofing](https://github.com/kprokofi/light-weight-face-anti-spoofing) 기반). 학습·`onnx/openvino` 변환·평가 포함. → [README](Spoofing/light-fas/README.md)
- **Silent-Face-Anti-Spoofing** — Silent Face Anti-Spoofing. → [README](Spoofing/Silent-Face-Anti-Spoofing/README.md)

### Tracking
`Tracking/Human/` — 사람 추적
- **HiFT** — Hierarchical Feature Transformer 기반 추적. → [README](Tracking/Human/HiFT/README.md)
- **SiamMask** — 추적 + 세그멘테이션. → [README](Tracking/Human/SiamMask/README.md)

### Convert
`Convert/` — 모델 포맷 변환 유틸리티
| 스크립트 | 용도 |
| --- | --- |
| `mxnet2onnx.py` | MXNet → ONNX |
| `onnx2trt_v7.py`, `onnx2trt_v8.py`, `onnx2trt_multiple.py` | ONNX → TensorRT (버전별) |
| `onnx2openvino.sh` | ONNX → OpenVINO |
| `onnx2tflite.py` | ONNX → TFLite |
| `onnx_simplify.py` | ONNX 그래프 단순화 |
| `openvino_quantization.py` | OpenVINO 양자화 |

### Package
`package/` — 실제 추론에 바로 쓰는 파이썬 패키지 소스.
- **totalface** — 얼굴 탐지/랜드마크/임베딩/속성/블러 등을 통합한 GPU 패키지. 탐지 백본으로 RetinaFace·SCRFD·BlazeFace 등, 인식으로 ArcFace 지원.
- **totalface-python-cpu** — 위의 CPU 전용 버전(`totalface_cpu`).
- **totalhuman** — 사람 탐지/포즈/추적 통합 패키지.

각 패키지 폴더의 `inference_example.ipynb`에서 사용 예시를 확인할 수 있습니다.

## 설치 패키지

PyPI에 배포된 패키지는 아래처럼 설치합니다.

```bash
pip install totalface        # GPU 버전 (얼굴)
pip install totalface_cpu    # CPU 버전 (얼굴)
pip install totalhuman       # 사람
```

## 참고
- 대부분의 하위 프로젝트는 원본 저장소를 기반으로 실험/이식한 것이며, 해당 출처와 인용은 각 폴더 README에 명시되어 있습니다.
- 학습에 필요한 데이터셋·가중치 파일은 저장소에 포함되어 있지 않으며, 각 README의 안내를 따라 준비해야 합니다.
