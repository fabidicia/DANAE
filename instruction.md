# DANAE - Launching procedures

## Dataset acquisition
To download the Oxford Inertial Odometry Dataset (OxIOD) you will need to send a request through the following [link](https://forms.gle/wjE7u5AonoyyrgXJ7). They will provide you with a folder (which we renamed as Oxio_Dataset) containing a ReadMe.txt file with the data list order, and a set of subfolders (e.g. handheld, pocket, slow walking, etc). Each of them contains instructions on which set of IMU measures has been used as train or test sets among the data*n* folders, where *n* stands for the number of acquisition. We used those contained in the "./data/Oxio_Dataset/slow walking/data1/syn/" folder (i.e. imu1.csv, imu2.csv, imu3.csv and so on).

The Underwater Caves Sonar Dataset (UCSD) can be easily downloaded from the following [site](https://cirs.udg.edu/caves-dataset/). We used the *full_dataset.zip* containing the main bag file, in order to directly obtain the data without using ROS: we downloaded them in a directory named "caves". However, they report precise instructions to explore the ROS bags too.
Data are given in a single file for both the standard low-cost Xsens MTi AHRS and the Analog Devices ADIS16480 used during the acquisition. We then decided to split the data and use the first 80% to train DANAE++ and the remaining 20% to test the performances. The path to our file is for instance "./data/caves/full_dataset/imu_adis.txt".

Once set the data, you will need to combine them for obtaining the filtered data through the Kalman Filtering algorithms.

## Filtering algorithms
You can test both the Linear and the Extended Kalman Filters by running the main_LKF.py or the main_EKF.py files respectively. You will need to specify the chosen dataset and the related csv data file. For example, as previously said, we used the slow_walking set of Oxio Dataset, and inside the data1/syn subfolder we set the imu1.csv as test file and imu2.csv from imu7.csv as training. The resulting call to the algorithm is:

```python
python main_LKF.py --dataset oxford --path ./data/Oxio_Dataset/slow_walking/data1/syn/imu1.csv
```

This will produce its corresponding output in the "preds" folder: "./data/preds/dict_data1_imu1.pkl".
You need to run the main_LKF.py (and/or the main_EKF.py) for all the imu files contained in the chosen folder.
At the end of each run, you we suggesto to move the obtained predicted data (i.e. the .pkl file) in a newly created folder, as we will explain later, since it will be used in the training-testing phase.

## Training and testing phase
The .pkl files obtained through the KF algorithms will be moved in two separate folders: foldername_train and foldername_test. We suggest to use the same folder name of the origin data. For example, dict_data1_imu1.pkl will be stored in two newly created slow_walking_ekf_train and slow_walking_ekf_test folders as follows:

```Python
mv preds/dict_data1_imu1.pkl ./preds/slow_walking_ekf_test/
mv preds/dict_data1_imu2.pkl ./preds/slow_walking_ekf_train/
mv preds/dict_data1_imu3.pkl ./preds/slow_walking_ekf_train/
..
mv preds/dict_data1_imu8.pkl ./preds/slow_walking_train/
```

Having set all the elements, we can now train and then test DANAE++ specifying the previous used LKF or EKF with *input_type*, and the corresponding path we created. In the first case, we will run:

```Python
python main_DANAE.py --input_type lkf_est_complete --path ./preds/slow_walking_lkf_train
```

while in the latter:

```Python
python main_DANAE.py --input_type ekf_est_complete --path ./preds/slow_walking_ekf_train
```

*N.B.:* please notice that you only need to specify the training folder. That is because the code has been written to automatically test on the corresponding test folder. For this reason, remember to give to the train/test directory the same name! As an example, in our case we specify  --path ./preds/slow_walking_lkf_train, and the test will automatically be performed on the corresponding _test folder, i.e. "slow_walking_ekf_test"
