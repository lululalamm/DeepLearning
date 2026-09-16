# Detection

탐지(Detection) 관련 코드입니다. 현재는 얼굴(Face) 하위 작업을 포함합니다.

## Face

`Detection/Face/`

| 폴더 | 내용 | README |
| --- | --- | --- |
| `maskedFace` | 얼굴 마스크 착용 여부 분류. 백본: MnasNet, MobileNetV3, MobileFaceNet. AI Hub Face Mask 데이터셋. 학습·테스트·`torch→onnx` 변환·속도 측정 스크립트 포함. | [README](Face/maskedFace/README.md) |
| `maskWearing` | 얼굴 랜드마크 기반 마스크 착용 위치 추론(`mask_landmark.ipynb`). `totalface` 패키지 필요. | [README](Face/maskWearing/README.md) |

> 얼굴 탐지 백본(RetinaFace·SCRFD·BlazeFace 등)은 추론용 패키지 [`totalface`](../package/totalface-python/README.md)에 포함되어 있습니다.
