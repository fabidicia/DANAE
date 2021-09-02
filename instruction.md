# First of all, you will need the filtered orientation data obtained through the Linear or the Extended Kalman Filter. You will need to run:
>> python main_LKF.py 

##to launch danae++ with 3 angles estimation:
>> CUDA_VISIBLE_DEVICES=3 python main_DANAE.py --input_type ekf_est_complete --path ./data/preds/slow_walking_ekf_train

#another example
>> CUDA_VISIBLE_DEVICES=3 python main_DANAE.py --input_type ekf_est_complete --path ./data/preds/1_7_slow_walking_ekf_train/ --epochs 20
