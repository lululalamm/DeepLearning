echo "Test mbnv3-large load1"
python -u test_result_mbn.py \
--model_path race_FairFace-mobilenetv3-large-celoss-sgd_230504_nf_l1/final_best.pth \
--save_name mbnv3-large_l1


echo "Test mbnv3-large load3"
python -u test_result_mbn.py \
--model_path race_FairFace-mobilenetv3-large-celoss-sgd_230504_nf_l3/final_best.pth \
--save_name mbnv3-large_l3 \
--load_type 3


