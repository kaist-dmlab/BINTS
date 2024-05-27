
for dataset in 'nyc'
do
    for khop in 3 5 7
    do
        for seq_day in 4
        do
            for pred_day in 7 14 30
            do
                python3 main.py\
                --multi_gpu 1\
                --gpu_id 0\
                --batch_size 8\
                --dataset $dataset\
                --seq_day $seq_day\
                --pred_day $pred_day\
                --khop $khop
            done
        done
    done
done
