# echo "Aligned add 224"
# python src/aligned224_add_pool.py
# echo "Finish"

# echo "make h5 add 224"
# python src/make_h5.py
# echo "Finish"

# echo "Train celebA aligned224 Add"
# python -u src/train.py \
# --cfg_path configs/efficientNet_B0_celebA_add.py

# ps -ef | grep "train.py" | grep -v grep | awk '{print "kill -9 "$2}' | sh
# echo "Finish"


# echo "Train celebA aligned224"
# python -u src/train.py \
# --cfg_path configs/efficientNet_B0_celebA.py

# ps -ef | grep "train.py" | grep -v grep | awk '{print "kill -9 "$2}' | sh
# echo "Finish"



# echo "Aligned add 224"
# python src/crop224_pool.py
# echo "Finish"

# echo "make h5 add 224"
# python src/make_h5.py
# echo "Finish"

echo "Train celebA crop224"
python -u src/train.py \
--cfg_path configs/efficientNet_B0_celebA_crop.py

ps -ef | grep "train.py" | grep -v grep | awk '{print "kill -9 "$2}' | sh
echo "Finish"

echo "Test crop224 - celeb h5"
python -u src/test.py \
--cfg_path configs/efficientNet_B0_celebA_crop.py \
--model_path save_model_crop224/ \
--test_format h5 \
--test_base "celebA/h5/230720/celeba_add_224_test.h5" \
--false_save