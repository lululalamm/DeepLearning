# package

학습한 모델을 실제 추론에 바로 쓸 수 있도록 묶은 파이썬 패키지 소스입니다.

| 폴더 | 패키지 | 설치 | 설명 |
| --- | --- | --- | --- |
| `totalface-python` | `totalface` | `pip install totalface` | 얼굴 탐지/랜드마크/임베딩/속성/블러 통합 (GPU) |
| `totalface-python-cpu` | `totalface_cpu` | `pip install totalface_cpu` | 위의 CPU 전용 버전 |
| `totalhuman-python` | `totalhuman` | `pip install totalhuman` | 사람 탐지/포즈/추적 통합 |

## totalface
- 탐지 백본: RetinaFace(InsightFace/torch), SCRFD, BlazeFace, dlib
- 인식: ArcFace
- 랜드마크, 속성(attribute), 블러 처리 등을 하나의 API로 제공
- 사용 예시: `totalface-python/inference_example.ipynb` → [README](totalface-python/README.md)

## totalhuman
- 사람 탐지, 포즈(pose), 추적(tracker) 모듈 제공
- 사용 예시: `totalhuman-python/inference_example.ipynb` → [README](totalhuman-python/README.md)

> 각 패키지에는 동작에 필요한 소형 데모 가중치/설정 파일이 포함되어 있습니다. 그 외 모델 가중치는 별도로 준비해야 합니다.
