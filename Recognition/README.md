# Recognition

얼굴 인식(Recognition) 관련 코드입니다. 임베딩(Embedding)과 분류(Classification)로 나뉩니다.

## Face

`Recognition/Face/`

### Embedding
`Recognition/Face/Embedding/`
- 얼굴 인증용 임베딩 학습에 쓰이는 loss 함수 모음(ArcFace, ElasticFace, LiArcFace, BroadFace, MarginDistillation, DistillLoss, MLLoss).
- → [README](Face/Embedding/README.md)

### Classification
`Recognition/Face/Classification/`

| 폴더 | 내용 | README |
| --- | --- | --- |
| `attribute/Face-Attribute-Classification_pytorch` | CelebA 기반 얼굴 속성 분류(PyTorch 이식판) | [README](Face/Classification/attribute/Face-Attribute-Classification_pytorch/README.md) |
| `attribute/arcface-cmt-pytorch` | Conditional Multitask Learning — 임베딩 백본을 pretrain으로 다중 속성 동시 학습 | [README](Face/Classification/attribute/arcface-cmt-pytorch/README.md) |
| `emotion/DAN` | 표정 분류(DAN 기반, 112×112 정렬 입력) | [README](Face/Classification/emotion/DAN/README.md) |
| `race` | 인종 분류(FairFace, MobileNetV3-small/large·MobileFaceNet 백본) | [README](Face/Classification/race/README.md) |
