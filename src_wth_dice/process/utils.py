import numpy as np
from scipy import ndimage

from torchvision import transforms
from monai.transforms import AddChannel
import torch
import matplotlib.pyplot as plt
########################################### Vectorization functions #############################################

#compute a vector field for a given contour f and kernel K
def VFC(f,K):
    def cropTo(img, toShape):
        fromShape = np.array(img.shape)
        toShape = np.array(toShape)

        cropVal = ((fromShape-toShape)/2).astype(np.int8)
        xCrop,yCrop,zCrop = cropVal[0], cropVal[1], cropVal[2]
        img_crop = img[xCrop:-xCrop,yCrop:-yCrop,zCrop:-zCrop]

        return img_crop
    # pad
    kx,ky,kz = K.shape[0],K.shape[1], K.shape[2]
    f_pad = np.pad(f, ((kx,kx),(ky,ky),(kz,kz)), 'reflect')
    FFTsize = np.array(f_pad.shape) + kx-1

    fftf = np.fft.fftn(f_pad, FFTsize)
    Fz = np.real(np.fft.ifftn(fftf * np.fft.fftn(K[:,:,:,2], FFTsize)))

    temp = np.fft.ifftn(fftf*np.fft.fftn(K[:,:,:,0]+1j*K[:,:,:,1], FFTsize))
    Fy = np.imag(temp)
    Fx = np.real(temp)
    
    Fext = np.zeros(np.array([f.shape[0], f.shape[1],f.shape[2],3]))
    Fext[:,:,:,0]=cropTo(Fx, f.shape)
    Fext[:,:,:,1]=cropTo(Fy, f.shape)
    Fext[:,:,:,2]=cropTo(Fz, f.shape)
    
    return Fext

#build a kernel for VFC
def VFK(r=10,a=1.5,R=np.array([1,1,1])):
    # VFK
    eps = 1e-8
    r0 = np.floor(r/R)
    xx,yy,zz = R[0]*np.arange(r0[0],-r0[0]-1,-1), R[1]*np.arange(r0[1],-r0[1]-1,-1), R[2]*np.arange(r0[2],-r0[2]-1,-1)
    XX,YY,ZZ = np.meshgrid(xx,yy,zz)
    dist = np.sqrt(XX**2+YY**2+ZZ**2)
    
    MASK=dist<r
    m=dist**(a+1)
    Kx,Ky,Kz = XX/(m+eps), YY/(m+eps), ZZ/(m+eps)
    Kx, Ky, Kz = Kx*MASK, Ky*MASK, Kz*MASK

    Kx = np.expand_dims(Kx, axis=3)
    Ky = np.expand_dims(Ky, axis=3)
    Kz = np.expand_dims(Kz, axis=3)

    K=np.concatenate((Kx,Ky,Kz), axis=3)
    
    return K


def image_contour(image):
    """ Function to compute image contour """
  
    # Get x-gradient in "sx"
    sx = ndimage.sobel(image,axis=0,mode='constant')
    # Get y-gradient in "sy"
    sy = ndimage.sobel(image,axis=1,mode='constant')
    # Get z-gradient in "sz"
    sz = ndimage.sobel(image,axis=2,mode='constant')
    # Get square root of sum of squares
    sobel=np.hypot(sx,sy,sz)
    
    return sobel


def to_vector(scan):
    #only works for 3D scan, need to be call on dataloader to modify the data
    vector_field=VFC(image_contour(scan),VFK(r=10,a=1.5,R=np.array([1,1,1])))
    norm_vector_field=np.linalg.norm(vector_field, axis=3)
    return (vector_field/np.max(norm_vector_field))

########################################### transformations functions #############################################

def to_tensor(scan, t):
    """
    input : 4D dimension array shape (*values,channel)
    output : 4D dimension tensor (*values,channel)
    """
    scan_0=t(scan[:,:,:,0])
    scan_1=t(scan[:,:,:,1])
    scan_2=t(scan[:,:,:,2])
    return torch.cat((scan_0,scan_1,scan_2),dim=0)


########################################### visualizations #############################################

def show_slices(slices):

    """ Function to display row of image slices """
    if type(slices)!=list:
        plt.imshow(slices.T, origin="lower")
    else :
        fig, axes = plt.subplots(len(slices)//5, 5, figsize=(20,20))
        for i, slice in enumerate(slices):
            axes[(i//5),i-((i//5)*5)].imshow(slice.T, origin="lower")

def show_scanner(scanners, showned_slice=None):
    """
    inputs : scanners : list or input
             seg : linked seg to superpose to the scanners (if you want to only print the seg, put it in as a scanners)
             showned_slice : int for last showned coordinate
             seg superposition is not implement yet
    """
    if showned_slice is None:
        showned_slice=64
    
    if type(scanners)!=list:
        show_slices(scanners[:,:,showned_slice])
    else :
        slices=[scanner[:,:,showned_slice] for scanner in scanners]
        show_slices(slices)
                             
            
            
def loss_recovery(path):
    train_loss=[]
    test_loss=[]
    with open(path, 'r') as f:
        data=f.readlines()
        for line in data:
            if 'Train Loss' in line:
                train_loss.append(float(line[-6:]))
            elif 'Test Loss' in line:
                test_loss.append(float(line[-6:]))
    plt.figure(figsize=(20,7))
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.legend(['train loss','test loss'])
    plt.show()
    
def Dice_recovery(path):
    train_dice=[]
    test_dice=[]
    with open(path, 'r') as f:
        data=f.readlines()
        for line in data:
            if 'Train Mean Dice' in line:
                train_dice.append(float(line[29:]))
            elif 'Test Mean Dice' in line:
                if 'Best Test Mean Dice' not in line:
                    test_dice.append(float(line[29:]))
    plt.figure(figsize=(20,7))
    
    plt.plot(train_dice)
    plt.plot(test_dice)

   # plt.scatter([i for i in range(len(train_dice))],train_dice)
   # plt.scatter([i for i in range(len(test_dice))],test_dice)
    plt.legend(['train dice','test dice'])
    plt.show()
    
def plot_vector_field(vector_field):
    x,y = np.meshgrid(np.linspace(0,1,96),np.linspace(0,1,96))
    u,v = vector_field[:,:,64,0],vector_field[:,:,64,1]
    plt.quiver(x,y,u,v)
    plt.show()