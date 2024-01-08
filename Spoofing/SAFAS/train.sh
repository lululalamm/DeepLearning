python -u train_custom.py \
--data_dir /data/shared/Face/FaceLiveness/datasets/temp/safas/csv/ \
--protocol W_CA_to_C \
--ca align \
--result_path results_align \
--batch_size 382 \
--num_epochs 25 \
--step_size 10 \
--align_epoch 5

ps -ef | grep "train_custom.py"  | grep -v grep | awk '{print "kill -9 "$2}' | sh
