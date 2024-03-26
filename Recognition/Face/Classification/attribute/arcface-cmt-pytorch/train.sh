python -u train_arcfaceCmt2.py \
--date 202106 \
--train_data Gender-Age/arcface_aligned/210530_fan/arcface_aligned_0610_train_predAge_addMask.h5 \
--val_data Gender-Age/arcface_aligned/210530_fan/arcface_aligned_0610_val_predAge_addMask.h5 \
--train_batch 128 \
--val_batch 20 \
--lr 0.005 \
--save_name arcface-0610_210530fan_addMask_predAge_res50 \
--prepath Gender-Age/models/pretrained/2022_02_14_ms1m_r50_arcface/model.pt


echo "220319 Test cmt unmask cvt res50"
echo "wiki korean"
python -u test_agengender_vsSSR_hdf5_cmt.py \
--backbone_name res50 \
--weight /data/notebook/arcface_GenderAge/best_cmt_models/202106_arcface-0610_210530fan_addMask_predAge_res50.pth \
--h5_path Gender-Age/test_data/mtcnn_aligned_112x112/arcface_wiki_korean_finish_new.h5
echo "japan celeb"
python -u test_agengender_vsSSR_hdf5_cmt.py \
--backbone_name res50 \
--weight /data/notebook/arcface_GenderAge/best_cmt_models/202106_arcface-0610_210530fan_addMask_predAge_res50.pth \
--h5_path Gender-Age/test_data/mtcnn_aligned_112x112/arcface_japan_celeb_new.h5
echo "celeb list"
python -u test_agengender_vsSSR_hdf5_cmt.py \
--backbone_name res50 \
--weight /data/notebook/arcface_GenderAge/best_cmt_models/202106_arcface-0610_210530fan_addMask_predAge_res50.pth \
--h5_path Gender-Age/test_data/mtcnn_aligned_112x112/arcface_celeb_list_new.h5