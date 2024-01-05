echo "Test 224 add best, test h5 celeba"
python -u src/test.py \
--cfg_path configs/efficientNet_B0_celebA_crop.py \
--model_path save_model_crop224/best_epoch17.pth \
--test_format h5 \
--test_base "celebA/h5/230720/celeba_crop_224_test.h5" \
--false_save