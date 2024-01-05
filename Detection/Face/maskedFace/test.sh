echo "mbnv3-small id"
python -u test.py \
--model_name mobilenetv3-small \
--model_path ./save_models/mobilenetv3-small_facemask_filtering-id_230314_lr00005_aug_focal/last.pth \
--test_path FaceMask/filtering/saved_test_id.csv \
--load_type 1

echo "mbf id"
python -u test.py \
--model_name mobilefacenet \
--model_path ./save_models/mobilefacenet_facemask_filtering-id_230314_lr00001_aug_focal/last.pth \
--test_path FaceMask/filtering/saved_test_id.csv


echo "mbnv3-large2 id"
python -u test.py \
--model_name mobilenetv3-large \
--model_path ./save_models/mobilenetv3-large2_facemask_filtering-id_230314_lr0001_aug_focal/last.pth \
--test_path FaceMask/filtering/saved_test_id.csv \
--load_type 2
