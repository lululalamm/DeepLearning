python -u src/speed_check.py \
--cfg_path configs/efficientNet_B0_celebA.py \
--model_path save_model_224/best_epoch28.pth

python -u src/speed_check.py \
--cfg_path configs/efficientNet_B0_celebA.py \
--model_path efficientNetB0_celebA_224.onnx

python -u src/speed_check.py \
--cfg_path configs/efficientNet_B0_celebA.py \
--model_path efficientNetB0_celebA_224.vino