## AQUALOC

This repository contains the AQUALOC dataset sequences.

The sequences are split in two parts: one for the sequences acquired in an harbor and one for the sequences acquired on deep archaeological sites.

The sequences are both available as ROS bags and in a raw format.

### Citation

If you use this dataset for your research, please cite it as:

@inproceedings{ferrera2019aqualocijrr, title={AQUALOC: An Underwater Dataset for Visual-Inertial-Pressure Localization}, author={Ferrera, Maxime and Creuze, Vincent and Moras, Julien and Trouv{\'e}-Peloux, Pauline}, booktitle={The International Journal of Robotics Research}, year={2019} }

### Calibration 

Each repository contains the calibration files relative to the acquisitions.  The calibration of the Camera-IMU setup has been performed with the [Kalibr library](http://https://github.com/ethz-asl/kalibr "Kalibr library") [1,2].

The provided calibration files follow the output format of Kalibr.  Each repository contains the camera intrinsic parameters, the camera-IMU extrinsic parameters and the IMU noise model.

### Groundtruth

For each sequence, a groundtruth trajectory is provided. The GT trajectories have been computed offline with the SfM library [Colmap](http://https://github.com/colmap/colmap "Colmap") [3].  The trajectory files format is the following:

------------
\# img_number tx ty tz qx qy qz qw

------------

The tx, ty, tz represent the three elements of the position of the camera in the world frame and qx, qy, qz, qw represent the four elements of the quaternion related to the camera's orientation.  As not every image in one sequence have been used to compute the trajectories, img_number gives the position of the corresponding image in the full sequence.

As Colmap computes a trajectory by first performing an exhaustive matching attempt between all the provided images, it is able to detect the loop closure in the processed sequences.  The resulting images matching is provided in text files with the following format:

------------
1 1 0 0 ... 1 1
1 1 1 0 ... 1 1
0 1 1 1 ... 1 0
0 0 1 1 ... 1 0
....

------------
Each row and column correspond to one image and a 1 indicates an overlapping between both images.  Hence, a 1 between the third row and the fourth column indicates that the third and fourth images are overlapping (which corresponds to the images of the third and fourth lines in the related trajectory file).


------------
[1] Paul Furgale, Joern Rehder, Roland Siegwart (2013). Unified Temporal and Spatial Calibration for Multi-Sensor Systems. In Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Tokyo, Japan.
[2] Paul Furgale, T D Barfoot, G Sibley (2012). Continuous-Time Batch Estimation Using Temporal Basis Functions. In Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), pp. 2088–2095, St. Paul, MN.
[3] J. L. Schönberger and J.-M. Frahm, “Structure-from-Motion Revisited,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.