echo "Start train sgd optim mbn-small load1"
NCCL_P2P_DISABLE=1 \
python -u train_mbn.py \
--config configs/race_mbn-small_230504.py

ps -ef | grep "train_mbn.py" | grep -v grep | awk '{print "kill -9 "$2}' | sh
