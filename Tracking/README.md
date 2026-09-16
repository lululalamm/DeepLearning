# Tracking

객체 추적(Tracking) 코드입니다. 현재는 사람(Human) 추적을 포함합니다.

## Human

`Tracking/Human/`

| 폴더 | 방식 | README |
| --- | --- | --- |
| `HiFT` | Hierarchical Feature Transformer 기반 추적. `inference_demo.ipynb`, 블러 추론(`infer_blur.py`) 포함. | [README](Human/HiFT/README.md) |
| `SiamMask` | Siamese 네트워크 기반 추적 + 세그멘테이션. `inference_model.ipynb`, 앱 추론(`infer_app.py`) 포함. | [README](Human/SiamMask/README.md) |

> `sample_pose.jpg`는 데모용 샘플 이미지입니다. 사람 탐지/포즈/추적을 통합한 추론 패키지는 [`totalhuman`](../package/totalhuman-python/README.md)을 참고하세요.
