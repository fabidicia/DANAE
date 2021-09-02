### DANAE

DANAE is a deep Denoising AutoeNcoder for Attitude Estimation which works on Kalman filter IMU/AHRS data integration with the aim of reducing any kind of noise, independently of its nature.
![plot](./Results_Figure/oxford_LKF_phi.jpg)
![plot](./Results_Figure/oxford_danae1_phi.jpg)

![plot](./Results_Figure/ucs_lkf_theta.jpg)
![plot](./Results_Figure/ucs_danae1_theta.jpg)

DANAE++ is the enhanced version of the first architecture: it is able to denoise IMU/AHRS data obtained through Linear (LKF) and Extended (EKF) Kalman filter-derived values, obtaining better results also when compared to common low-pass filters (in our study, the Butter LP filter and the Uniform filter both provided by the Scipy library in pytorch).
![plot](./Results_Figure/oxford_EKF_phi.jpg)
![plot](./Results_Figure/oxford_danae++_phi.jpg)
![plot](./Results_Figure/comparative_filters_butter_phi.jpg)
![plot](./Results_Figure/comparative_filters_uniform_phi.jpg)

![plot](./Results_Figure/ucs_ekf_theta.jpg)
![plot](./Results_Figure/ucs_danae++_theta.jpg)

### Reference
[Conference Paper]: https://arxiv.org/abs/2011.06853 presented @Metrology for the Sea 2020

[Extended Article]: https://www.mdpi.com/1424-8220/21/4/1526 published by Sensors, MDPI - 2021


