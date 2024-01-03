path_list=(
    glint360k-r100-arcface_multiple.onnx \
    glint360k-r100m-arcface_multiple.onnx \
    glint360k-r100m-pfc-arcface_multiple.onnx \
)

echo "convert onnx to openvino"
# 'mo' is openvino command, need to install openvino 
for (( i = 0 ; i < ${#path_list[@]} ; i++ )) ; do
    echo "${path_list[$i]}"
    mo \
    --input_model ${path_list[$i]} \
    --output_dir ./convert_openvino/ \
    --input_shape '[-1,3,640,640]'
done
