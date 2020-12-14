import numpy as np
from squaternion import Quaternion
from scipy.spatial.transform import Rotation as R 
from math import sin, cos, tan, pi, atan2, sqrt
from math import *

#######################################################################################
#The followings are functions definitions of general and specific use
#######################################################################################

def rad2deg(rad):
    return rad / np.pi * 180

def deg2rad(deg):
    return deg / 180 * np.pi

def getRotMat(q):
    c00 = q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2
    c01 = 2 * (q[1] * q[2] - q[0] * q[3])
    c02 = 2 * (q[1] * q[3] + q[0] * q[2])
    c10 = 2 * (q[1] * q[2] + q[0] * q[3])
    c11 = q[0] ** 2 - q[1] ** 2 + q[2] ** 2 - q[3] ** 2
    c12 = 2 * (q[2] * q[3] - q[0] * q[1])
    c20 = 2 * (q[1] * q[3] - q[0] * q[2])
    c21 = 2 * (q[2] * q[3] + q[0] * q[1])
    c22 = q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2

    rotMat = np.array([[c00, c01, c02], [c10, c11, c12], [c20, c21, c22]])
    return rotMat

def getEulerAngles(q):
    m = getRotMat(q)
    test = -m[2, 0]
    if test > 0.99999:
        yaw = 0
        pitch = np.pi / 2
        roll = np.arctan2(m[0, 1], m[0, 2])
    elif test < -0.99999:
        yaw = 0
        pitch = -np.pi / 2
        roll = np.arctan2(-m[0, 1], -m[0, 2])
    else:
        yaw = np.arctan2(m[1, 0], m[0, 0])
        pitch = np.arcsin(-m[2, 0])
        roll = np.arctan2(m[2, 1], m[2, 2])

    yaw = rad2deg(yaw)
    pitch = rad2deg(pitch)
    roll = rad2deg(roll)

    return yaw, pitch, roll

def quaternion_to_euler(x, y, z, v):
    x, y, z, v = float(x), float(y), float(z), float(v)
    t0 = +2.0 * (v * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    t2 = +2.0 * (v * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = np.arcsin(t2)
    t3 = +2.0 * (v * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return [roll, pitch, yaw]

#######################################################################################
#The following defines the EXTENDED KALMAN FILTER CLASS AND ITS METHODS
#######################################################################################

class System:
    def __init__(self):
        quaternion = np.array([1, 0, 0, 0])     # Initial estimate of the quaternion
        bias = np.array([0, 0, 0])              # Initial estimate of the gyro bias

        self.xHat = np.concatenate((quaternion, bias)).transpose() #state estimate, cioè Ax
        self.yHatBar = np.zeros(3).transpose()
        self.p = np.identity(7) #* 0.01 #mxm con m = dim(xHat)
        self.Q = np.identity(7) #* 0.001 #mxm con m = dim(xHat)
        self.R = np.identity(6) #* 0.1 #nxn con n = dim(yHatBar)
        self.K = None
        self.A= None
        self.B = None
        self.C = None
        self.xHatBar = None
        self.xHatPrev = None
        self.pBar = None
        self.accelReference = np.array([0, 0, -1]).transpose()
        self.magReference = np.array([0, 1, 0]).transpose()
        #values got from specific calibrations of Oxford Mag Data
        self.mag_Ainv = np.array([[ 0.05486116, -0.00011346, 0.00146397],
                                  [-0.00011346,  0.05951535,  0.00731025],
                                  [0.00146397,  0.00731025,  0.07483137]])
        self.mag_b = np.array([-2.50627418, -11.4804655, -29.53963506]).transpose() 

    def normalizeQuat(self, q):
        mag = (q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)**0.5
        return q / mag

    def getAccelVector(self, a):
        accel = np.array(a).transpose()
        accelMag = (accel[0] ** 2 + accel[1] ** 2 + accel[2] ** 2) ** 0.5
        return accel / accelMag

    def getMagVector(self, m):
        magGaussRaw = np.matmul(self.mag_Ainv, np.array(m).transpose() - self.mag_b)
        magGauss_N = np.matmul(getRotMat(self.xHat), magGaussRaw)
        magGauss_N[2] = 0
        magGauss_N = magGauss_N / (magGauss_N[0] ** 2 + magGauss_N[1] ** 2) ** 0.5 #normalizzazione
        magGuass_B = np.matmul(getRotMat(self.xHat).transpose(), magGauss_N) #prodotto vettoriale matrice rotazione per magnet
        return magGuass_B

    def getJacobianMatrix(self, reference):
        qHatPrev = self.xHatPrev[0:4]
        e00 = qHatPrev[0] * reference[0] + qHatPrev[3] * reference[1] - qHatPrev[2] * reference[2]
        e01 = qHatPrev[1] * reference[0] + qHatPrev[2] * reference[1] + qHatPrev[3] * reference[2]
        e02 = -qHatPrev[2] * reference[0] + qHatPrev[1] * reference[1] - qHatPrev[0] * reference[2]
        e03 = -qHatPrev[3] * reference[0] + qHatPrev[0] * reference[1] + qHatPrev[1] * reference[2]
        e10 = -qHatPrev[3] * reference[0] + qHatPrev[0] * reference[1] + qHatPrev[1] * reference[2]
        e11 = qHatPrev[2] * reference[0] - qHatPrev[1] * reference[1] + qHatPrev[0] * reference[2]
        e12 = qHatPrev[1] * reference[0] + qHatPrev[2] * reference[1] + qHatPrev[3] * reference[2]
        e13 = -qHatPrev[0] * reference[0] - qHatPrev[3] * reference[1] + qHatPrev[2] * reference[2]
        e20 = qHatPrev[2] * reference[0] - qHatPrev[1] * reference[1] + qHatPrev[0] * reference[2]
        e21 = qHatPrev[3] * reference[0] - qHatPrev[0] * reference[1] - qHatPrev[1] * reference[2]
        e22 = qHatPrev[0] * reference[0] + qHatPrev[3] * reference[1] - qHatPrev[2] * reference[2]
        e23 = qHatPrev[1] * reference[0] + qHatPrev[2] * reference[1] + qHatPrev[3] * reference[2]
        jacobianMatrix = 2 * np.array([[e00, e01, e02, e03],
                                       [e10, e11, e12, e13],
                                       [e20, e21, e22, e23]])
        return jacobianMatrix

    def predictAccelMag(self):
        # Accel
        hPrime_a = self.getJacobianMatrix(self.accelReference)
        accelBar = np.matmul(getRotMat(self.xHatBar).transpose(), self.accelReference)
        #print(accelBar)

        # Mag
        hPrime_m = self.getJacobianMatrix(self.magReference)
        magBar = np.matmul(getRotMat(self.xHatBar).transpose(), self.magReference)
        #print(magBar)

        tmp1 = np.concatenate((hPrime_a, np.zeros((3, 3))), axis=1)
        tmp2 = np.concatenate((hPrime_m, np.zeros((3, 3))), axis=1)
        self.C = np.concatenate((tmp1, tmp2), axis=0)

        return np.concatenate((accelBar, magBar), axis=0)

    
    ####################### EXTENDED KALMAN FILTER LOOP, internally defined ######################
    
    def predict(self, w, dt): #w è il vettore gyro!
        q = self.xHat[0:4] #xHat ha dim = 7, ma gli ultimi 3 valori sono i bias del gyr
        Sq = np.array([[-q[1], -q[2], -q[3]],
                       [ q[0], -q[3],  q[2]],
                       [ q[3],  q[0], -q[1]],
                       [-q[2],  q[1],  q[0]]])
        tmp1 = np.concatenate((np.identity(4), -dt / 2 * Sq), axis=1)
        tmp2 = np.concatenate((np.zeros((3, 4)), np.identity(3)), axis=1)
        self.A = np.concatenate((tmp1, tmp2), axis=0)

        self.B = np.concatenate((dt / 2 * Sq, np.zeros((3, 3))), axis=0)

        self.xHatBar = np.matmul(self.A, self.xHat) + np.matmul(self.B, np.array(w).transpose())
        #state_estimate = A(state_estimate) + b(gyro_input)
        self.xHatBar[0:4] = self.normalizeQuat(self.xHatBar[0:4]) #prende gli elementi del quat
        self.xHatPrev = self.xHat # prende la PRIMA stima e la rende PREVISIONE PASSATA + aggiunge di nuovo i bias
        self.yHatBar = self.predictAccelMag() # calcolo di y
        self.pBar = np.matmul(np.matmul(self.A, self.p), self.A.transpose()) + self.Q #calcolo di P


    def update(self, a, m):
        tmp1 = np.linalg.inv(np.matmul(np.matmul(self.C, self.pBar), self.C.transpose()) + self.R) #CPC^T + R
        self.K = np.matmul(np.matmul(self.pBar, self.C.transpose()), tmp1) #kalman gain

        magGuass_B = self.getMagVector(m) #prende misure raw di mag e acc
        accel_B = self.getAccelVector(a)

        measurement = np.concatenate((accel_B, magGuass_B), axis=0) #ne fa il vettore misura z
        self.xHat = self.xHatBar + np.matmul(self.K, measurement - self.yHatBar) #state estimate + (K(z-y))
        self.xHat[0:4] = self.normalizeQuat(self.xHat[0:4])
        self.p = np.matmul(np.identity(7) - np.matmul(self.K, self.C), self.pBar)
        self.x = self.xHat[0] 
        self.y = self.xHat[1]
        self.z = self.xHat[2] 
        self.v = self.xHat[3]

        quat = Quaternion(self.x, self.y, self.z, self.v)
        e = quat.to_euler(degrees=False)
        d = quat.to_dict()
        psi_hat = (e[0])
        theta_hat = (e[1])
        phi_hat = (e[2])
        
        return phi_hat, theta_hat, psi_hat 
        #import pdb; pdb.set_trace()


#######################################################################################
#The following defines the LINEAR KALMAN FILTER CLASS AND ITS METHODS
#######################################################################################

class LSystem:
    def __init__(self, inits=None):
        dt = 0.1
        self.xHat = np.array([[0], [0], [0], [0], [0], [0]]) #state estimate, cioè Ax
        if inits is not None:
            self.xHat[0][0] = inits[0]
            self.xHat[2][0] = inits[1]
            self.xHat[4][0] = inits[2]
        self.P = np.eye(6)  #* 0.01 #mxm con m = dim(xHat)
        self.Q = np.eye(6)  #* 0.001 #mxm con m = dim(xHat)
        self.R = np.eye(3) #* 0.1 #nxn con n = dim(yHatBar)
        self.A = np.array([[1, -dt, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0],
                        [0, 0, 1, -dt, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, -dt],
                        [0, 0, 0, 0, 1, 0]])
        self.B = np.array([[dt, 0, 0],
                            [0, 0, 0],
                            [0, dt, 0],
                            [0, 0, 0],
                            [0, 0, dt],
                            [0, 0, 0]])
        self.C = np.array([[1, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0]])
        #self.xHatPrev = None
        #self.PBar = None
        self.gyro_input = None

    def getAccAng(self, ax, ay, az, mx, my, mz):
        phi_hat = self.xHat[0][0]
        theta_hat = self.xHat[2][0]

        phi_acc = np.arctan2(ay, np.sqrt(ax ** 2.0 + az ** 2.0))
        theta_acc = np.arctan2(-ax, np.sqrt(ay ** 2.0 + az ** 2.0))
        #psi = np.arctan2(np.sqrt(ax ** 2.0 + ay ** 2.0), az)
        m_norm = sqrt((mx*mx)+(my*my)+(mz*mz))
        mx = (mx/m_norm)
        my = (my/m_norm)
        mz = (mz/m_norm)
        psi_acc = np.arctan2((-my*cos(phi_hat) + mz*sin(phi_hat)), (mx*cos(theta_hat) + my*sin(theta_hat)*sin(phi_hat) + mz*sin(theta_hat)*cos(phi_hat)))
        return [phi_acc, theta_acc, psi_acc]

    def gyroAng(self, p, q, r):
        phi_hat = self.xHat[0][0]
        theta_hat = self.xHat[2][0]
        psi_hat = self.xHat[4][0]

        phi_dot = (p + sin(phi_hat) * tan(theta_hat) * q + cos(phi_hat) * tan(theta_hat) * r)
        theta_dot = cos(phi_hat) * q - sin(phi_hat) * r
        psi_dot = (sin(phi_hat) / cos(theta_hat)*q + cos(phi_hat) / cos(theta_hat) * r)

        gyro_input = np.array([[phi_dot], [theta_dot], [psi_dot]])
        return gyro_input

    ####################### LINEAR KALMAN FILTER LOOP, internally defined ######################
    
    def predict(self, w, dt): #dt is a dummy param
        p, q, r = w
        self.gyro_input = self.gyroAng(p, q, r)
        xHat = self.A.dot(self.xHat) + self.B.dot(self.gyro_input) 
        #self.xHatPrev = self.xHat # prende la PRIMA stima e la rende PREVISIONE PASSATA 
        self.P = self.A.dot(self.P.dot(np.transpose(self.A))) + self.Q #calcolo di P


    def update(self, a, m):
        ax, ay, az = a
        mx, my, mz = m        
        measurement = self.getAccAng( ax, ay, az, mx, my, mz) #ne fa il vettore misura z
        y_tilde = measurement - self.C.dot(self.xHat)
        S = self.R + self.C.dot(self.P.dot(np.transpose(self.C)))
        K = self.P.dot(np.transpose(self.C).dot(np.linalg.inv(S)))
        self.xHat = self.xHat+ K.dot(y_tilde)
        self.P = (np.eye(6) - K.dot(self.C)).dot(self.P)
        phi_hat = self.xHat [0][0]
        theta_hat = self.xHat [2][0]
        psi_hat = self.xHat[4][0]

        return psi_hat, theta_hat, phi_hat 
        #import pdb; pdb.set_trace()

