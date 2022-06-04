import cv2
import numpy as np
from map import *
from LodaData import *
import itertools


class PSLAM:
    def __init__ (self,
                  predict_sigma,
                  map_minx,
                  map_miny,
                  map_maxx,
                  map_maxy,
                  map_resolution,
                  map_logodds_max,
                  map_logodds_min,
                  map_logodds_free,
                  map_logodds_occu,
                  map_logodds_binary_theshold,
                  particleNumber=300,
                  resample_theshold=1,
                  map_texture_update_alpha=0.2):
         self.particlesNum=particleNumber
         self.resampleThreshold=resample_theshold
         self.mapMinx=map_minx
         self.mapMiny=map_miny
         self.mapMaxx=map_maxx
         self.mapMaxy=map_maxy
         self.mapLogOddsMax=map_logodds_max
         self.mapLogOddsMin=map_logodds_min
         self.map_free=map_logodds_free
         self.map_occu=map_logodds_occu
         self.predictNoise=predict_sigma
         # 3 is x, y, theta
         self.particle=np.zeros((particleNumber,3))
         self.weights=np.ones((particleNumber))/particleNumber
         self.map=MAP(map_minx,map_maxx,map_miny,map_maxy,map_resolution)
         self.texture=np.zeros((self.map.ysize,self.map.xsize,3),dtype=np.float64)
         self.CurrentData={}
         self.mapBinaryThres=map_logodds_binary_theshold
         self.textureAlpha=map_texture_update_alpha
         self.param=Param();
         self.z=np.zeros((2,self.param.cameraCols*self.param.cameraRows),dtype=np.int32)

         for i in range(self.param.cameraRows):
             for j in range(self.param.cameraCols):
                 self.z[0,i*self.param.cameraCols+j]=j
                 self.z[1,i*self.param.cameraCols+j]=i





    def getCarState(self):
        state=np.sum(self.weights.reshape(-1, 1) * self.particle/self.weights.sum(), axis=0)
        return state


    def setCarData(self,data):
        self.CurrentData=data

    def getBinaryMap(self):
        tmpMap=(self.map.map>self.mapBinaryThres).astype(np.int32)
        return tmpMap


    def mapProb(self):
        tmpProb=1.0-1.0/(1.0+np.exp(self.map.map))
        return tmpProb


    def getTexture(self):
        return self.texture.astype(np.int32)


    def inMap(self, coor):
        return self.map.inMap(coor)


    def CoorToIndex(self, coor):
        return  self.map.coorToIndex(coor)

    def resample(self):
        weights=self.weights
        num=self.particlesNum
        indices=[]
        C=[0.0]+ [np.sum(weights[:i+1]) for i in range(num)]
        u0, j = np.random.random(), 0
        for u in [(u0 + i) / num for i in range(num)]:
            while u > C[j]:
                j += 1
            indices.append(j - 1)
        return indices

    def predictParticles(self,ux,uy,angel):
        u=np.array([ux,uy,angel])
        u0=np.random.multivariate_normal(u,self.predictNoise,self.particlesNum)
        self.particle+=u0
        return


    def updateParticles(self,lidarData):
        cooralate=np.zeros_like(self.weights)
        mapBina=self.getBinaryMap()
        #Transform LidarData to Body frame;
        angles = np.linspace(-5, 185, 286) / 180 * np.pi
        indValid = np.logical_and((lidarData < self.param.LidarMaxRange), (lidarData > self.param.LidarMinRange))
        rangeFilter = lidarData[indValid]
        anglesFilter = angles[indValid]
        xs0 = rangeFilter * np.cos(anglesFilter)
        ys0 = rangeFilter * np.sin(anglesFilter)
        zs0 = np.zeros_like(rangeFilter);
        ones = np.ones_like(rangeFilter);
        DataMatrix = np.vstack([xs0, ys0, zs0, ones])
        BodyCoor=self.param.LidarToBodyTrans.dot(DataMatrix)

        #update every particle;
        for i, p in enumerate(self.particle):
            #
            worldToBodyTrans=np.array([[np.cos(p[2]), -np.sin(p[2]), 0,p[0]],
                     [np.sin(p[2]), np.cos(p[2]), 0,p[1]],
                     [0, 0, 1,0],[0,0,0,1]])
            lidarWorld=worldToBodyTrans.dot(BodyCoor)

            #get the position:
            lidarWorldxyz=lidarWorld[:3,:]
            #filer the point above the ground
            Filter0=lidarWorldxyz[2,:]>0.1;
            lidarFilter=lidarWorldxyz[:,Filter0]

            #filter the point in the map
            lidarInMap=lidarFilter[:,self.map.inMap(lidarFilter)]
            if lidarInMap.shape[1]==0 or lidarInMap.shape[0]==0:
                print('errorINdexxxxx')
                return

            #get the coor in the map:
            lidarIndex=self.map.coorToIndex(lidarInMap)
            if lidarIndex.shape[1]==0 or lidarIndex.shape[0]==0:
                print('error 00000')
                return

            bias={}
            c=np.zeros(25)
            for j, (y,x) in enumerate(itertools.product(range(-2,3),range(-2,3))):
                bias[j]=(y,x)
                #if np.max(lidarIndex[1,:])+y<mapBina.shape[0] and np.max(lidarIndex[0,:])+x<mapBina.shape[1] and np.min(lidarIndex[1,:])+y>mapBina.shape[0]and np.min(lidarIndex[0,:])+x>mapBina.shape[1]:
                c[j]=np.sum(mapBina[lidarIndex[1,:]+y,lidarIndex[0,:]+x])

            index=np.argmax(c)
            if np.argmax(c):
              self.particle[i,0]+=bias[index][1]*self.map.resolution
              self.particle[i,1]+=bias[index][0]*self.map.resolution
            cooralate[i]=c.max()

        #update prob:

       # logWights=np.exp(self.weights)+cooralate
        #self.weights=np.log(logWights)
        #tmp=self.weights/self.weights.sum()
        #n_coee=1/np.sum(tmp**2)
        self.weights=cooralate+0.000001

        tmp = self.weights / self.weights.sum()
        n_coee = 1 / np.sum(tmp ** 2)


        # calculate if resample:
        if n_coee<=self.resampleThreshold:
            idx=self.resample()
            self.particle=self.particle[idx,:]
            self.weights=self.weights[idx]
            self.weights/=self.weights.sum()
        return cooralate, n_coee


    def updateMap(self,lidarData):
        # Transform LidarData to Body frame;
        angles = np.linspace(-5, 185, 286) / 180 * np.pi
        indValid = np.logical_and((lidarData < self.param.LidarMaxRange), (lidarData > self.param.LidarMinRange))
        rangeFilter = lidarData[indValid]
        anglesFilter = angles[indValid]
        xs0 = rangeFilter * np.cos(anglesFilter)
        ys0 = rangeFilter * np.sin(anglesFilter)
        zs0 = np.zeros_like(rangeFilter);
        ones = np.ones_like(rangeFilter);
        DataMatrix = np.vstack([xs0, ys0, zs0, ones])
        BodyCoor = self.param.LidarToBodyTrans.dot(DataMatrix)

        #plt.figure()
        #ax = plt.subplot(111, projection='polar')
        #ax.plot(anglesFilter, rangeFilter)
        #ax.set_rmax(40)
        #ax.set_rticks([0.5, 1, 1.5, 2])  # fewer radial ticks
        #ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
        #ax.grid(True)
        #ax.set_title("Lidar scan data", va='bottom')
        #plt.show()

        #get car state:
        state=self.getCarState()

        #transform to world
        worldToBodyTrans = np.array([[np.cos(state[2]), -np.sin(state[2]), 0, state[0]],
                                     [np.sin(state[2]), np.cos(state[2]), 0, state[1]],
                                     [0, 0, 1, 0], [0, 0, 0, 1]])
        lidarWorld = worldToBodyTrans.dot(BodyCoor)

        # get the position:
        lidarWorldxyz = lidarWorld[:3, :]
        # filer the point above the ground
        lidarFilter = lidarWorldxyz[:,lidarWorldxyz[2, :] > 0.1]


        # filter the point in the map
        #print(lidarFilter.shape);
        lidarInMap = lidarFilter[:,self.map.inMap(lidarFilter)]

        # get the coor in the map:

        lidarIndex = self.map.coorToIndex(lidarInMap)
       # plt.figure()
       # plt.plot(lidarIndex[0,:],lidarIndex[1,:])
       # plt.show()



        #may be an error
        if(lidarIndex.shape[1]==0):
            return
        xx=[lidarIndex.reshape(-1,1).astype(np.int32)]
        self.map.map+= cv2.drawContours(np.zeros_like(self.map.map),contours=[lidarIndex.T],
                                       contourIdx=-1, color=self.map_free,thickness=-1)
       # tmp=cv2.drawContours(np.zeros_like(self.map.map),contours=[lidarIndex.T],
        #                               contourIdx=-1, color=255,thickness=-1).astype(np.uint8)

       # cv2.imshow('map1', tmp)
       # cv2.waitKey(1)


        self.map.map[lidarIndex[1,:],lidarIndex[0,:]]+=self.map_occu-self.map_free

        self.map.map=np.clip(self.map.map,self.mapLogOddsMin,self.mapLogOddsMax)
        return

    def updateMapTexture(self,imageLeft,imageRight):

        image_l = cv2.cvtColor(imageLeft, cv2.COLOR_BAYER_BG2BGR)
        image_r = cv2.cvtColor(imageRight, cv2.COLOR_BAYER_BG2BGR)

        image_l_gray = cv2.cvtColor(image_l, cv2.COLOR_BGR2GRAY)
        image_r_gray = cv2.cvtColor(image_r, cv2.COLOR_BGR2GRAY)

        # You may need to fine-tune the variables `numDisparities` and `blockSize` based on the desired accuracy
        stereo = cv2.StereoBM_create(numDisparities=32, blockSize=9)
        disparity = stereo.compute(image_l_gray, image_r_gray)
        tmp=self.param.fsu*self.param.b
        z=tmp/((disparity.astype(np.float32)/16)+0.0001)

        state = self.getCarState()

        # transform to world
        worldToBodyTrans = np.array([[np.cos(state[2]), -np.sin(state[2]), 0, state[0]],
                                     [np.sin(state[2]), np.cos(state[2]), 0, state[1]],
                                     [0, 0, 1, 0], [0, 0, 0, 1]])
        CamToWorld=worldToBodyTrans@self.param.cameraToBodyTrans

        # get xyz rgb in an array;
        depth=z.flatten()
        tmpz=self.z-[[self.param.cu],[self.param.cv]]
        tmpp=tmpz*np.vstack([depth,depth])
        bodyxy=self.param.invCam@tmpp
        Bodyy=np.vstack([bodyxy,depth,np.ones(self.param.cameraRows*self.param.cameraCols)])
        world=CamToWorld@Bodyy
        rgb=image_l.reshape(self.param.cameraRows*self.param.cameraCols,-1).T


        ## in the map
        inMapFilter=self.inMap(world)
        world=world[:,inMapFilter]
        rgb=rgb[:,inMapFilter]

        ##around the ground
        groundMask=np.logical_and(world[2,:]<0.1,world[2,:]>-0.1)

        ##mask the dark color
        darkMask=np.logical_and(np.logical_and(rgb[0,:]>20,rgb[1,:]>20),rgb[2,:]>20)

        maskFinal=np.logical_and(groundMask,darkMask)
        finalPoints=world[:,maskFinal]
        rgb=rgb[:,maskFinal]

        pointIndex=self.CoorToIndex(finalPoints)
        self.texture[pointIndex[1,:],pointIndex[0,:],:] *=(1.0-self.textureAlpha)
        self.texture[pointIndex[1,:],pointIndex[0,:],:]+=(self.textureAlpha*rgb[:,:].T)

        cv2.imshow('window', self.texture.astype(np.uint8))
        cv2.waitKey(1)

        return