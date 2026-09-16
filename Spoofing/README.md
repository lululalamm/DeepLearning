# Spoofing

얼굴 위변조 판별(Face Anti-Spoofing, FAS) 코드입니다. 사진·영상·마스크 등으로 위장한 얼굴을 진짜 얼굴과 구분합니다.

| 폴더 | 방식 | README |
| --- | --- | --- |
| `SAFAS` | CVPR 2023 "Rethinking Domain Generalization for Face Anti-spoofing: Separability and Alignment" 구현. 도메인 일반화 중심. | [README](SAFAS/README.md) |
| `light-fas` | 경량 FAS([light-weight-face-anti-spoofing](https://github.com/kprokofi/light-weight-face-anti-spoofing) 기반). 학습·`onnx/openvino` 변환·평가 포함. | [README](light-fas/README.md) |
| `Silent-Face-Anti-Spoofing` | Silent Face Anti-Spoofing([minivision-ai](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) 기반). scale-crop 패치 기반. | [README](Silent-Face-Anti-Spoofing/README.md) |

## 데이터셋
프로젝트별로 다르며, 주로 FaceInTheWild·CelebA-Spoof 및 공개 FAS 벤치마크(OULU-NPU, CASIA-FASD, Replay-Attack, MSU-MFSD)를 사용합니다. 자세한 준비 방법은 각 폴더 README를 참고하세요.
