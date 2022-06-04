import os
import numpy as np
import cv2
import matplotlib.pyplot as plt; plt.ion()


def timeCon():
    folder='data/stereo_images/stereo_left/'
    folder1='data/stereo_images/stereo_right/'
    #n=len(next(os.walk(folder))[2])
    dir=os.listdir(folder)
    x=np.zeros(len(dir))
    for i in range(len(dir)):
        x[i]=int(dir[i].strip('gnp.'))

    z=np.argsort(x)
    time=str(z)
    leftdir=[]
    rightdir=[]
    np.set_printoptions(suppress=True)
    for i in range(len(dir)):
        l=folder+dir[z[i]]
        leftdir.append(l)
        r=folder1+dir[z[i]]
        rightdir.append(r)

    for i in range(len(dir)):
        limage=cv2.imread(leftdir[i])
        rimage=cv2.imread(rightdir[i])



        try:
            rimage.shape
        except:
            print('r image not exist')
            continue

        cv2.imshow("left", limage)
        #cv2.imshow("right", rimage)
        cv2.waitKey(5)





def main():
   timeCon()
   return







if __name__=='__main__':
    main()