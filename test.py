import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D


import os
import pylab
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import gc

from map import  *
from LodaData import  *
from ParticleSLAM import  *


def plot_map(mp,
             pos=None,
             npos=None,
             figsize=20,
             save_fig_name=None,
             navigation_heading=None,
             show_navigation=False):
    PATH_COLOR = '#ff4733'
    NAVIGATION_COLOR = '#3888ff'

    fig, ax = plt.subplots(figsize=(figsize, figsize), dpi=80)
    mp = np.flip(mp, 1)
    if mp.ndim == 2:
        ax.imshow(mp, cmap='bone')
    else:
        ax.imshow(mp)

    if pos is not None:
        posx, posy = pos
        assert len(posx) == len(posy)

        for i, (px, py) in enumerate(zip(posx, posy)):
            ax.plot(mp.shape[1] - px, py, marker='o', color=PATH_COLOR, ms=1)

    if show_navigation:
        npos = len(posx)
        px, py = posx[-1], posy[-1]
        dx = -np.cos(navigation_heading)
        dy = np.sin(navigation_heading)

        ax.arrow(mp.shape[1] - px,
                 py,
                 dx,
                 dy,
                 length_includes_head=True,
                 head_width=15,
                 head_length=20,
                 head_starts_at_zero=True,
                 overhang=0.2,
                 zorder=999,
                 facecolor=NAVIGATION_COLOR,
                 edgecolor='black')

    if save_fig_name:
        Path(os.path.dirname(save_fig_name)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_fig_name, bbox_inches='tight')
        # Clear the current figure.
        plt.clf()
        # Closes all the figure windows.
        plt.close('all')
        plt.close(fig)
        del fig
        del ax
        gc.collect()
        return None, None
    else:
        plt.show(block=True)
        return fig, ax


def main():
    data=LoadDataFile()
    param=Param()
    predict_sigma=np.diag([1,1,1e-2])
    map_minx=-200
    map_miny=-600
    map_maxx=600
    map_maxy=200
    map_resolution=0.5
    map_logodds_max=100
    map_loggodds_min=-100
    map_loggodss_free=-np.log(9)
    map_loggodds_occu=np.log(9)
    map_logodds_binary_thred=0
    particle_num=300
    resample_thred=1
    map_texture_alpha=1

    slam=PSLAM(predict_sigma,
               map_minx,
               map_miny,
               map_maxx,
               map_maxy,
               map_resolution,
               map_logodds_max,
               map_loggodds_min,
               map_loggodss_free,
               map_loggodds_occu,
               map_logodds_binary_thred,
               particle_num,
               resample_thred,
               map_texture_alpha
    )

    states=[np.array([0,0,0])]
    dtime=10;

    for i in range(1,100000):


        lidarData =data.lidarData[i*dtime,:]
        encData=data.encoderData[i*dtime,0]
        encDataNext=data.encoderData[(i+1)*dtime,0]

        theta=data.fogData[i*dtime*10:(i+1)*dtime*10-1,2]
        angel=theta.sum()+states[i-1][2]
        speed=(encDataNext-encData)/param.EncoderResolution*3.14*param.EncoderLeftRadius;
        speedx=speed*np.cos(angel)
        speedy=speed*np.sin(angel)


        print(i)
        slam.updateMap(lidarData)

        slam.predictParticles(speedx,speedy,theta.sum())

        cor, err = slam.updateParticles(lidarData)

        states.append(slam.getCarState())
        sss=np.array(states)



        if (i+1)%10==0 :
          gamma = ((1.0 - 1.0 / (1.0 + np.exp(slam.map.map)))*255).astype(np.uint8)
          gamma=cv2.cvtColor(gamma, cv2.COLOR_GRAY2BGR)
          #cv2.imshow('map', gamma)
          a=str(i)
          b=a+'.png'
          coor=slam.CoorToIndex(sss.T)

          for j in range(sss.shape[0]):
              cv2.circle(gamma,[coor[0][j],coor[1][j]],1,[0,0,255],thickness=-1)

          cv2.imwrite(b,gamma)




        if (i + 1) % 10 == 0:
         for j in range(i):
            plt.plot([states[j][0], states[j + 1][0]], [states[j][1], states[j + 1][1]], color='red', linewidth=3.0,
                     linestyle='--')
            aa=str(i+1)
            bb=aa+'.png'
            plt.savefig(bb)
       # if (i+1) %10==0:
        if 1:

            path_l = 'data/image_left.png'
            path_r = 'data/image_right.png'
            image_l = cv2.imread(path_l, 0)
            image_r = cv2.imread(path_r, 0)
            slam.updateMapTexture(image_l,image_r)
            print(i)

    gamma = 1.0 - 1.0 / (1.0 + np.exp(slam.map.map))
    print(gamma.max(), gamma.min())
    print(slam.map.map.shape)



    for i in range(998):
        plt.plot([states[i][0], states[i + 1][0]], [states[i][1], states[i + 1][1]], color='red', linewidth=3.0,
                 linestyle='--')
    plt.show()
    print(1)
    index = slam.CoorToIndex(np.array(states))
   # plot_map(gamma, [index[0, :], index[1, :]])


def plot_trac():
    data = LoadDataFile()
    param=Param()
    state=[np.array([0,0,0])]
    dtime=100;
    for i in range(1,1000):

        encData = data.encoderData[i * dtime, 0]
        encDataNext = data.encoderData[(i + 1) * dtime, 0]
        theta = data.fogData[i * dtime * 10:(i + 1) * dtime * 10 - 1, 1]
        angel = theta.sum()+state[i-1][2]
        speed = (encDataNext - encData) / param.EncoderResolution * 3.14 * param.EncoderLeftRadius;
        speedx = speed * np.cos(angel)
        speedy = speed * np.sin(angel)
        x=state[i-1][0]+speedx;
        y=state[i-1][1]+speedy;
        state.append(np.array([x,y,angel]))

    for i in range(1000):
      plt.plot([state[i][0],state[i+1][0]],[state[i][1],state[i+1][1]],color = 'red',linewidth = 3.0,linestyle = '--')
    plt.show()
    print(1)


# nosie is 0,partile is 1, update the map the show lidar data.
def LidarScanVisual():
    data = LoadDataFile()
    param = Param()
    predict_sigma = np.diag([0.00, 0.00, 0.00])
    map_minx = -50
    map_miny = -200
    map_maxx = 200
    map_maxy = 30
    map_resolution = 0.5
    map_logodds_max = 100
    map_loggodds_min = -100
    map_loggodss_free = -np.log(9)
    map_loggodds_occu = np.log(9)
    map_logodds_binary_thred = 0
    particle_num = 1
    resample_thred = 5
    map_texture_alpha = 1

    slam = PSLAM(predict_sigma,
                 map_minx,
                 map_miny,
                 map_maxx,
                 map_maxy,
                 map_resolution,
                 map_logodds_max,
                 map_loggodds_min,
                 map_loggodss_free,
                 map_loggodds_occu,
                 map_logodds_binary_thred,
                 particle_num,
                 resample_thred,
                 map_texture_alpha
                 )

    states = [np.array([0, 0, 0])]
    dtime = 25;

    for i in range(1, 2):

        lidarData = data.lidarData[i * dtime, :]
        encData = (data.encoderData[i * dtime, 0] + data.encoderData[i * dtime, 1]) / 2.0
        encDataNext = (data.encoderData[(i + 1) * dtime, 0] + data.encoderData[(i + 1) * dtime, 1]) / 2.0

        theta = data.fogData[i * dtime * 10:(i + 1) * dtime * 10 - 1, 2]
        angel = theta.sum() + states[i - 1][2]
        speed = (encDataNext - encData) / param.EncoderResolution * 3.14 * param.EncoderLeftRadius;
        speedx = speed * np.cos(angel)
        speedy = speed * np.sin(angel)

        slam.updateMap(lidarData)

        slam.predictParticles(speedx, speedy, theta.sum())

        cor, err = slam.updateParticles(lidarData)

        states.append(slam.getCarState())
        sss = np.array(states)
        # cv2.waitKey(1)

        # if (i+1) %10==0:
        if 1:
            gamma = ((1.0 - 1.0 / (1.0 + np.exp(slam.map.map))) * 255).astype(np.uint8)
            # cv2.imshow('map', gamma)
            gamma = cv2.cvtColor(gamma, cv2.COLOR_GRAY2BGR)
            a = str(i + 2)
            strf = a + '.png'
            stt =  strf
            coor = slam.CoorToIndex(sss.T)

            for j in range(sss.shape[0]):
                cv2.circle(gamma, [coor[0][j], coor[1][j]], 1, [0, 0, 255], thickness=-1)
            cv2.imwrite(stt, gamma)

            texx = 'map801/'
            a = str(i + 20000) + '.png'
            cc =  a
            cv2.imwrite(cc, slam.texture.astype(np.uint8))


# nosie is 0 and particle is 1, the map will only has lidar data

def deadReckoning():
    data = LoadDataFile()
    param = Param()
    predict_sigma = np.diag([0.0, 0.0, 0.0])
    map_minx = -500
    map_miny = -2000
    map_maxx = 2000
    map_maxy = 300
    map_resolution = 0.5
    map_logodds_max = 100
    map_loggodds_min = -100
    map_loggodss_free = -np.log(9)
    map_loggodds_occu = np.log(9)
    map_logodds_binary_thred = 0
    particle_num = 1
    resample_thred = 5
    map_texture_alpha = 1

    slam = PSLAM(predict_sigma,
                 map_minx,
                 map_miny,
                 map_maxx,
                 map_maxy,
                 map_resolution,
                 map_logodds_max,
                 map_loggodds_min,
                 map_loggodss_free,
                 map_loggodds_occu,
                 map_logodds_binary_thred,
                 particle_num,
                 resample_thred,
                 map_texture_alpha
                 )

    states = [np.array([0, 0, 0])]
    dtime = 25;

    for i in range(1, 4600):

        lidarData = data.lidarData[i * dtime, :]
        encData = (data.encoderData[i * dtime, 0] + data.encoderData[i * dtime, 1]) / 2.0
        encDataNext = (data.encoderData[(i + 1) * dtime, 0] + data.encoderData[(i + 1) * dtime, 1]) / 2.0

        theta = data.fogData[i * dtime * 10:(i + 1) * dtime * 10 - 1, 2]
        angel = theta.sum() + states[i - 1][2]
        speed = (encDataNext - encData) / param.EncoderResolution * 3.14 * param.EncoderLeftRadius;
        speedx = speed * np.cos(angel)
        speedy = speed * np.sin(angel)

        print(i)
        slam.updateMap(lidarData)

        slam.predictParticles(speedx, speedy, theta.sum())
        states.append(slam.getCarState())
        sss = np.array(states)

        if (i + 1) % 10 == 0:
            gamma = ((1.0 - 1.0 / (1.0 + np.exp(slam.map.map))) * 255).astype(np.uint8)
            # cv2.imshow('map', gamma)
            gamma = cv2.cvtColor(gamma, cv2.COLOR_GRAY2BGR)
            a = str(i + 2)
            strf = a + '.png'
            stt = 'map801/' + strf
            coor = slam.CoorToIndex(sss.T)

            for j in range(sss.shape[0]):
                cv2.circle(gamma, [coor[0][j], coor[1][j]], 1, [0, 0, 255], thickness=-1)
            cv2.imwrite(stt, gamma)

            texx = 'map801/'
            a = str(i + 20000) + '.png'
            cc = texx + a
            cv2.imwrite(cc, slam.texture.astype(np.uint8))

    gamma = 1.0 - 1.0 / (1.0 + np.exp(slam.map.map))
    print(gamma.max(), gamma.min())
    print(slam.map.map.shape)





if __name__=='__main__':
   #plot_trac()
   LidarScanVisual()
   #main()