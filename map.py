import numpy as np

class MAP:
    def __init__(self,xmin=-50,xmax=50,ymin=-50,ymax=50, res=0.05,dtype=np.float64):
        self.resolution=res
        self.xmin=xmin
        self.xmax=xmax
        self.ymin=ymin
        self.ymax=ymax
        self.dtype=dtype
        self.xsize=int(np.ceil((self.xmax-self.xmin)/self.resolution+1))
        self.ysize=int(np.ceil((self.ymax-self.ymin)/self.resolution+1))
        self.map=np.zeros((self.ysize,self.xsize),dtype=self.dtype)
    def getMap(self):
        return self.map

    def setMap(self,data):
        assert(data.shape==self.map.shape)
        self.map=data

    def inMap(self,coor):
        tmp=np.logical_and(np.logical_and(self.xmin<=coor[0,:],coor[0,:]<=self.xmax),
                              np.logical_and(self.ymin<=coor[1,:],coor[1,:]<=self.ymax))

        if tmp.shape[0]==0:
            print('ooooo')
        return tmp

    #coor need to be an np arrary and coor needs to be colum vector;

    #return a row vector;/// nnot a  colum!!!!!
    def coorToIndex(self, coor):
        #assert coor.shape[1]>1
        tmp=np.vstack([
            np.ceil((coor[0,: ] - self.xmin) / self.resolution),
            np.ceil((coor[1,: ] - self.ymin) / self.resolution)]).astype(np.int32)
        return tmp
