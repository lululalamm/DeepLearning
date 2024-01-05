
while true
do
    if ps -ef | grep "eval_csv_cropped.py" | grep -v grep
    then 
        echo "Test light-fas running..."
    else
        echo "Test light-fas end, Start Run Train expression"
        nohup ./train.sh  &
        break
    fi
    sleep 36000 # 10h
done