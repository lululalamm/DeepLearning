path_list=( 
            retinaface_r50_insightface.vino \
            retinaface_r50_insightface.onnx \
            retinaface_r50_insightface.v8.trt \
            retinaface_mnet.25.vino \
            retinaface_mnet.25.onnx \
            retinaface_mnet.25.v8.trt
)

name_list=( retinaface_insightface retinaface_insightface retinaface_insightface \
            retinaface_insightface retinaface_insightface retinaface_insightface)

# singlescale, only cuda onnx
for (( i = 0 ; i < 6 ; i++ )) ; do
    echo "get widerface ${path_list[$i]}"
    python -u get_widerface_txt.py \
    --dt_name ${name_list[$i]} \
    --dt_path ${path_list[$i]} \
    --save_format result_txt_220517_retest/${save_name[$i]}
    
    echo "eval widerface ${path_list[$i]}"
    python -u eval_widerface.py \
    --dt_name ${name_list[$i]} \
    --dt_path ${path_list[$i]} \
    --pred_format result_txt_220517_retest/${save_name[$i]} \
    --save_format result_pkl_220517_retest/${save_name[$i]}.pkl
    echo "finish"

done




echo "Start Our model test - multiscale"
for (( i = 300 ; i <= 1000 ; i+=10 )) ; do
    echo "start $i"

    python -u get_widerface_txt.py \
    --dt_name retinaface_torch \
    --dt_path retinaface_r50_final_multiple.onnx \
    --save_format retinaface_torch_r50_own_dynamic.onnx_multiscale_target$i \
    --target_size $i \
    --max_size 1200 \
    --multiscale \
    --write_time

    python -u eval_widerface.py \
    --dt_name retinaface_torch \
    --dt_path retinaface_torch/retinaface_r50_final_multiple.onnx \
    --pred_format retinaface_torch_r50_own_dynamic.onnx_multiscale_target$i \
    --save_format retinaface_torch_r50_own_dynamic.onnx_multiscale_target$i.pkl

done