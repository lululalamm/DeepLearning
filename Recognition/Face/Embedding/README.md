# Face Embedding - Loss

얼굴 인증용 임베딩 학습에 사용하는 margin/distillation 계열 loss 함수 구현 모음입니다.
각 파일은 학습 스크립트에서 임포트해 head/criterion으로 사용합니다.

| 파일 | Loss | 설명 |
| --- | --- | --- |
| `ArcFace.py` | ArcFace | Additive Angular Margin loss |
| `LiArcFace.py` | Li-ArcFace | ArcFace 변형(각도 선형화) |
| `ElasticFace.py` | ElasticFace | margin에 랜덤성을 준 변형 |
| `BroadFace.py` | BroadFace | 대규모 후보를 활용한 학습 |
| `MarginDistillation.py` | Margin Distillation | 지식 증류 기반 margin loss |
| `DistillLoss.py` | Distillation | 임베딩 지식 증류 loss |
| `MLLoss.py` | Metric Learning | 거리 기반 metric learning loss |

> 이 폴더는 loss 정의만 담고 있으며, 데이터 로딩·백본·학습 루프는 별도 학습 스크립트에서 구성합니다.
