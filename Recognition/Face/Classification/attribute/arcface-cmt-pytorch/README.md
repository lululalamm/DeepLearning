# Conditional Multitask Learning



# > train command --argument {detail,type,default}
- 기존에 사용한 pretrained embedding backbone : /data/notebook/NAS/Gender-Age/models/arcface-cmt/pretrained/backbone.pth
# train_arcfaceCmt.py : embedding 까지의 모델(얼굴인증용) 을 pretrain 으로 하여 학습, input data (h5) 가 1개.
- train_arcfaceCmt.py
    python train_arcfaceCmt.py \n
    --date {date,str} --prepath {pretrained model path} --input_data {input data path,h5} \n
    --momentum {optimizer momentum,float,0.9} --train_batch {train batch size,int,50} --val_batch {valid batch size,int,50}\n
    --image_size {image size,int,112} --lr {learning rate,float,0.001} --embedding_size {512} --num_epoch {total epoch,int,100}\n
    --early_stop {early stopping,bool,True}

# train_arcfaceCmt2.py : embedding 까지의 모델(얼굴인증용) 을 pretrain 으로 하여 학습, input data (h5) 가 train/valid 2개.
- train_arcfaceCmt2.py
    python train_arcfaceCmt2.py \n
    --date {date,str} --prepath {pretrained model path} \n
    --train_data {input training data path,h5} --val_data {input valid data path,h5}\n
    --momentum {optimizer momentum,float,0.9} --train_batch {train batch size,int,50} --val_batch {valid batch size,int,50}\n
    --image_size {image size,int,112} --lr {learning rate,float,0.001} --embedding_size {512}--num_epoch {total epoch,int,100}\n
    --early_stop {early stopping,bool,True}

# train_arcfaceCmt3.py : 위 1,2 번으로 학습한 cmt 모델을 pretrain 으로 학습, input data (h5) 가 1개.
- train_arcfaceCmt3.py
    python train_arcfaceCmt3.py \n
    --date {date,str} --prepath {pretrained model path} --prename {pretrained model name}\n
    --input_data {input data path,h5} \n
    --momentum {optimizer momentum,float,0.9} --train_batch {train batch size,int,50} --val_batch {valid batch size,int,50}\n
    --image_size {image size,int,112} --lr {learning rate,float,0.001} --embedding_size {512} --num_epoch {total epoch,int,100}\n
    --early_stop {early stopping,bool,True}

- prename 은 script 사용시 이전 학습의 최종 best model 을 pretrain 으로 하여 학습하기 위함, 없으면 prepath 사용


# > test command
- test_arcfaceCmt_h5.py
    python test_arcfaceCmt_h5.py
    --weight {trained model path} --h5_path {test data path,h5 format}