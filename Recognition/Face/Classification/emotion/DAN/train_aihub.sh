NCCL_SHM_DISABLE=1 \
python train_aihub.py \
--save_base checkpoints_aihub_230504_newlb_112 \
--input_size 112 \
--batch_size 1024 \
--num_classes 5