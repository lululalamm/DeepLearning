python -u train.py \
--config configs/config_large_01_wildcelebA.py

ps -ef | grep "train.py"  | grep -v grep | awk '{print "kill -9 "$2}' | sh
