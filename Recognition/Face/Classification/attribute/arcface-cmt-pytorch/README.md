# Conditional Multitask Learning (arcface-cmt)

얼굴 인증용 임베딩 백본을 pretrain으로 삼아 여러 속성을 동시에 학습하는 Conditional Multitask Learning 코드입니다.

- 예시로 사용한 pretrained embedding backbone: `/data/notebook/NAS/Gender-Age/models/arcface-cmt/pretrained/backbone.pth`
- 인자 표기: `{설명, 타입, 기본값}`

## 파일 구성
| 파일 | 용도 |
| --- | --- |
| `train_arcfaceCmt.py` | 학습 (임베딩 pretrain 백본 기반) |
| `test_arcfaceCmt_h5.py` | h5 테스트셋 평가 |
| `test_agengender_vsSSR_hdf5_cmt_c14.py` | age/gender 비교 평가 |
| `dataset_agengender_h5.py` | h5 데이터셋 로더 |
| `losses.py` | loss 정의 |
| `early_stopping.py` | early stopping 유틸 |
| `convert_onnx.py` | torch → ONNX 변환 |

## Train

### `train_arcfaceCmt.py`
임베딩(얼굴인증용) 모델을 pretrain으로 학습. 입력 데이터는 h5 1개.

```bash
python train_arcfaceCmt.py \
  --date {date,str} --prepath {pretrained model path} --input_data {input data path,h5} \
  --momentum {optimizer momentum,float,0.9} --train_batch {train batch size,int,50} --val_batch {valid batch size,int,50} \
  --image_size {image size,int,112} --lr {learning rate,float,0.001} --embedding_size {512} --num_epoch {total epoch,int,100} \
  --early_stop {early stopping,bool,True}
```

> 실제 인자와 기본값은 `train_arcfaceCmt.py` 상단의 `argparse` 정의를 기준으로 확인하세요.

## Test

### `test_arcfaceCmt_h5.py`

```bash
python test_arcfaceCmt_h5.py \
  --weight {trained model path} --h5_path {test data path,h5 format}
```

## Convert

### `convert_onnx.py`
학습한 torch 모델을 ONNX로 변환합니다. 인자는 스크립트 상단을 참고하세요.
