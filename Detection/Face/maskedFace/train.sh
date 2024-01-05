# filtering id aug focal
echo "Training - id"
echo "Train aug+focal Start - mobilenetv3-small"
python -u train_rec.py \
--model_name mobilenetv3-small \
--rec_path /data/shared/Face/FaceDetection/datasets/mask_classification/FaceMask/filtering \
--data_aug \
--save_name mobilenetv3-small_facemask_filtering-id_230314_lr00005_aug_focal \
--loss_metric focal_loss \
--rec_version _id
echo "Finish"

echo "Train aug+focal Start - mobilefacenet"
python -u train_rec.py \
--model_name mobilefacenet \
--rec_path /data/shared/Face/FaceDetection/datasets/mask_classification/FaceMask/filtering \
--data_aug \
--train_batch_size 256 \
--val_batch_size 256 \
--lr 0.0001 \
--save_name mobilefacenet_facemask_filtering-id_230314_lr00001_aug_focal \
--loss_metric focal_loss \
--rec_version _id
echo "Finish"

# filtering id aug focal
echo "Training - id"
echo "Train aug+focal Start - mobilenetv3-large"
python -u train_rec.py \
--model_name mobilenetv3-large \
--rec_path /data/shared/Face/FaceDetection/datasets/mask_classification/FaceMask/filtering \
--data_aug \
--train_batch_size 256 \
--val_batch_size 256 \
--lr 0.001 \
--save_name mobilenetv3-large2_facemask_filtering-id_230314_lr0001_aug_focal \
--loss_metric focal_loss \
--rec_version _id
echo "Finish"

# mnasnet
# filtering aug+focal
echo "Train aug+focal Start - mnasnet075"
python -u train_rec_mnas.py \
--model_name mnasnet075 \
--rec_path /data/notebook/shared/Face/FaceDetection/datasets/mask_classification/FaceMask/filtering \
--data_aug \
--save_name mnasnet075_facemask_filtering_230313_lr00005_aug_focal \
--loss_metric focal_loss
echo "Finish"

