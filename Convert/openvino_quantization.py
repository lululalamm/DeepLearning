## tested openivno-dev == 2022.2.dev~ 

import os
import argparse
import numpy as np
import cv2
from openvino.tools.pot import DataLoader
from openvino.tools.pot import IEEngine, load_model, save_model,compress_model_weights, create_pipeline


class ImageLoader(DataLoader):
    def __init__(self,dataset_path):
        self._files=[]
        
        # need to modify 
        for imgname in os.listdir(dataset_path):
            if not ".jpg" in imgname: continue
            img_path = os.path.join(dataset_path,imgname)
            self._files.append(img_path)
            
        self._shape = (112,112) # need to modify 
        
        
    def __len__(self):
        return len(self._files)
    
    def __getitem__(self,index):
        if index >= len(self):
            raise IndexError("Index out of dataset size")

        # need to modify  
        image_path = self._files[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,self._shape)
        image = image.astype(np.float32)
        image = (image/255. - 0.5)/0.5
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image,0)
        
        return image,None      
    





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='openvino quantization')
    parser.add_argument('--data_path', type=str,default="./FairFace/aligned_fair/val/", help='data samples path')
    parser.add_argument('--model', type=str,default="./race/race-mbf-arcface-sgd_221209_nf.xml", help='openvino xml path')
    parser.add_argument('--weights', type=str, default="./race/race-mbf-arcface-sgd_221209_nf.bin", help='openvino bin path')
    parser.add_argument('--out_dir', type=str, default='./race', help='quantization model save directory path')
    parser.add_argument('--out_name', type=str, default='race-mbf-arcface-sgd_221209_nf_quantINT8', help='quantization model save file base name')
    
    args = parser.parse_args()

    # Load sample data
    data_loader = ImageLoader(args.data_path)

    # Set quantization parameters
    q_params=[{
    "name":"DefaultQuantization",
    "params":{
        "target_device":"CPU",
        "preset":"performance",
        "stat_subset_size":300,
    "stat_batch_size":1},}]

    # set config
    model_config={
    "model":args.model,
    "weights":args.weights
    }
    engine_config={'device':'CPU'}


    # 1. model load
    model = load_model(model_config=model_config)

    # 2. engine initialize
    engine = IEEngine(config=engine_config, data_loader=data_loader)

    # 3. pipeline
    pipeline = create_pipeline(q_params,engine)
    compressed_model = pipeline.run(model=model)

    # 4. (optional) Compress model weights to reduce bin file size
    compress_model_weights(compressed_model)

    # 5. Save quantization model
    compressed_model_paths = save_model(
        model=compressed_model,
        save_path = args.out_dir,
        model_name = args.out_name,
    )


