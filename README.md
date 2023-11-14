# DANAE++ - A Deep Learning based approach to denoise attitude estimations using an autoencoder

All the instructions to install and correctly run DANAE++ can be found in the [instruction.md](./instruction.md) file.
Please remember that the results presented in the paper can be replicated using the datasets chosen by the authors. In fact, the autoencoder captures the characteristics of the sensors used to build the dataset.
If you wish to try DANAE++ on a custom dataset, keep in mind that probably a basic finetuning (or even a full train from scratch) of the autoencoder weights will be necessary to obtain good results! For this reason, we suggest to make a first training on the proposed datasets (also to further verify if it correctly runs) *having prior modified the script to save and load the weights*. At this point, you can proceed with their tuning. 

If you liked our work and you wish to consider (or use) it in your research, please consider to cite us!

> Russo, P., Di Ciaccio, F., & Troisi, S. (2021). DANAE++: A Smart Approach for Denoising Underwater Attitude Estimation. Sensors, 21(4), 1526.

> @article{russo2021danae++,
  title={DANAE++: A Smart Approach for Denoising Underwater Attitude Estimation},
  author={Russo, Paolo and Di Ciaccio, Fabiana and Troisi, Salvatore},
  journal={Sensors},
  volume={21},
  number={4},
  pages={1526},
  year={2021},
  publisher={Multidisciplinary Digital Publishing Institute}
}

# DANAE++ - Overview
DANAE is a deep Denoising AutoeNcoder for Attitude Estimation which works on Kalman Filter IMU/AHRS data integration with the aim of reducing any kind of noise, independently of its nature. In the first implementation, the Linear KF has been implemented on two set of data: the [Oxford Inertial Odometry Dataset](http://deepio.cs.ox.ac.uk/) (OxIOD) and the [Underwater Cave Sonar Dataset](https://cirs.udg.edu/caves-dataset/) (UCSD). The following images show the results of both the LKF and DANAE applied on the roll (phi) and the pitch angles estimation for the OXIOD and the UCSD respectively.

### DANAE Roll estimation - OXIO Dataset

![plot](./Results_Figure/oxford_LKF_phi.jpg)
![plot](./Results_Figure/oxford_danae1_phi.jpg)

### DANAE Pitch estimation - UCS Dataset
![plot](./Results_Figure/ucs_lkf_theta.jpg)
![plot](./Results_Figure/ucs_danae1_theta.jpg)

DANAE++ is the enhanced version of the first architecture: it is able to denoise IMU/AHRS data obtained through both the Linear (LKF) and Extended (EKF) Kalman filter-derived values. Better results are achieved by DANAE++ also when compared to common low-pass filters (in our study, the [Butter LP filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
) and the [Uniform1d filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.uniform_filter.html) both provided by the Scipy library).

The following images shows the results obtained by DANAE++ w.r.t. the roll angle estimation provided by the EKF and the LP filters for the OXIO Dataset, together with DANAE++ performance on the pitch angle estimation for the UCS Dataset.

### DANAE++ Roll estimation - OXIO Dataset
![plot](./Results_Figure/oxford_EKF_phi.jpg)
![plot](./Results_Figure/oxford_danae++_phi.jpg)
![plot](./Results_Figure/comparative_filters_butter_phi.jpg)
![plot](./Results_Figure/comparative_filters_uniform_phi.jpg)

### DANAE++ Pitch estimation - UCS Dataset
![plot](./Results_Figure/ucs_ekf_theta.jpg)
![plot](./Results_Figure/ucs_danae++_theta.jpg)

# References
[Conference Paper]: https://www.scopus.com/record/display.uri?eid=2-s2.0-85123283881&origin=resultslist published by IMEKO-International Measurement Federation Secretariat, MetroSea 2020 - TC19 International Workshop on Metrology for the Sea

[Extended Article]: https://www.mdpi.com/1424-8220/21/4/1526 published by Sensors, MDPI - 2021



