python -u src/train.py \
--cfg_path configs/efficientNet_B0_celebA.py

ps -ef | grep "train.py" | grep -v grep | awk '{print "kill -9 "$2}' | sh