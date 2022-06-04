import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
from map import *

class LoadDataFile:

  def __init__(self):
      self.pathLidar='data/sensor_data/lidar.csv'
      self.pathEncoder='data/sensor_data/encoder.csv'
      self.pathFog='data/sensor_data/fog.csv'
      self.pathLeftCamera='data/stereo_images/stereo_left/'
      self.pathRightCamera='data/stereo_images/stereo_right/'

      self.lidarTime,self.lidarData=self.read_data_from_csv(self.pathLidar)

      self.encoderTime,self.encoderData=self.read_data_from_csv(self.pathEncoder)

      self.FogTime,self.fogData=self.read_data_from_csv(self.pathFog)

      self.timeAll = self.FogTime.shape[0];
      self.timeStart=np.ceil(self.FogTime[0]/1000000)
      self.LidarTimeNew=np.ceil(self.lidarTime/10000000)
      self.EncoderTimeNew = np.ceil(self.encoderTime / 10000000)
      self.FogTimeNew = np.ceil(self.FogTime / 1000000)


      ## for encoder parameter: car model

  def read_data_from_csv(self,filename):
      '''
      INPUT
      filename        file address

      OUTPUT
      timestamp       timestamp of each observation
      data            a numpy array containing a sensor measurement in each row
      '''
      data_csv = pd.read_csv(filename, header=None)
      data = data_csv.values[:, 1:]
      timestamp = data_csv.values[:, 0]
      return timestamp, data

  def testEncoder(self):
      pass

  def testEncoderAndFog(self):
      pass

  def testLidarAndEcoderFog(self):
      pass

class Param :
    def __init__(self):
        self.EncoderResolution = 4096
        self.EncoderLeftRadius = 0.623479;
        self.EncoderRightRadius = 0.622806;
        self.wheel_base = 1.52439
        self.EncoderFPS = 100

        ## for Lidar parameter:
        self.LidarToBodyTrans = np.array([[0.00130201, 0.796097, 0.605167, 0.8349],
                                          [0.999999, - 0.000419027, - 0.00160026, -0.0126869],
                                          [- 0.00102038, 0.605169, - 0.796097, 1.76416], [0, 0, 0, 1]])
        self.LidarStartAngel = -5
        self.LidarEndAngel = 185
        self.LidarNum = 286
        self.LidarMaxRange = 80
        self.LidarMinRange = 0.1
        self.LidarFPS = 100

        # for Fog data:
        self.FogToBodyTrans = [[1, 0, 0, -0.335], [0, 1, 0, -0.035], [0, 0, 1, 0.78], [0, 0, 0, 1]]
        self.fogFPS = 1000

        # for stero camera:
        self.cameraToBodyTrans = [[-0.00680499, - 0.0153215, 0.99985, 1.64239],
                                  [- 0.999977, 0.000334627, - 0.00680066, 0.247401],
                                  [- 0.000230383, - 0.999883, - 0.0153234, 1.58411],
                                  [0, 0, 0, 1]]
        self.cameraFPS = 1
        self.cameraRows = 560
        self.cameraCols = 1280
        self.leftCameraMatrix = np.array([[8.1690378992770002e+02, 5.0510166700000003e-01, 6.0850726281690004e+02],
                                          [0., 8.1156803828490001e+02, 2.6347599764440002e+02],
                                          [0., 0., 1.]])
        self.leftCameraDistortion = np.array([-5.6143027800000002e-02, 1.3952563200000001e-01,
                                              -1.2155906999999999e-03, -9.7281389999999998e-04,
                                              -8.0878168799999997e-02])
        self.leftCameraRectification = np.array(
            [[9.9996942080938533e-01, 3.6208456669806118e-04, -7.8119357978017733e-03],
             [-3.4412461339106772e-04, 9.9999729518344416e-01, 2.3002617343453663e-03],
             [7.8127475572218850e-03, -2.2975031148170580e-03, 9996684067775188e-01]])

        self.leftCameraProjection = np.array([[7.7537235550066748e+02, 0., 6.1947309112548828e+02, 0.],
                                              [0., 7.7537235550066748e+02, 2.5718049049377441e+02, 0.],
                                              [0., 0., 1., 0.]])
        self.rightCameraMatrix = np.array([[8.1378205539589999e+02, 3.4880336220000002e-01, 6.1386419539320002e+02],
                                           [0., 8.0852165574269998e+02, 2.4941049348650000e+02],
                                           [0., 0., 1.]])
        self.rightCameraDistortion = np.array([-5.4921981799999998e-02, 1.4243657430000001e-01,
                                               7.5412299999999996e-05, -6.7560530000000001e-04,
                                               -8.5665408299999996e-02])
        self.rightCameraRectification = np.array(
            [[9.9998812489422739e-01, 2.4089155522231892e-03, -4.2364131513853301e-03],
             [-2.4186483057924992e-03, 9.9999444433315865e-01, -2.2937835970734117e-03],
             [4.2308640843048539e-03, 2.3040027516418276e-03, 9.9998839561287933e-01]])
        self.rightCameraProjection = np.array(
            [[7.7537235550066748e+02, 0., 6.1947309112548828e+02, -3.6841758740842312e+02],
             [0., 7.7537235550066748e+02, 2.5718049049377441e+02, 0.],
             [0., 0., 1., 0.]])
        self.fsu=8.1690378992770002e+02
        self.b=0.475143600050775
        self.invCam=np.array([[0.0012 ,  -0.0000],[0, 0.0012]])
        self.cu=6.0850726281690004e+02
        self.cv=2.6347599764440002e+02