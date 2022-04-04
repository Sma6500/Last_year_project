import torch
from torch.nn.functional import one_hot

import os

os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm

#https://docs.monai.io/en/0.8.0/metrics.html#metric
from monai.metrics import compute_meandice


########################################### Dice #############################################
def one_hot_encoding(tensor):
    """ 
    input : tensor of shape (batch_size, 1 (channel for ohe), h,w,l) requires int64 Tensor !!!
    output: tensor ohe (batch_size, nb_of_class, h, w,l)
    """
    nb_class=len(torch.unique(tensor))
    one_hot = torch.FloatTensor(tensor.size(0), nb_class, *tensor.shape[2:]).zero_()
    one_hot.scatter_(1, tensor, 1) #c'est si malin
    return one_hot
    

def compute_dice(y_pred, y):
    """
    input : two tensor (Batch_size, 1, H, W, L)
    output : tensor of lenght nb_class -1 (we dont count the domain) with every dice score
    """
    y_pred_ohe=one_hot_encoding(torch.round(y_pred).to(torch.int64)) #round to label 
    y_ohe=one_hot_encoding(y.to(torch.int64))
    
    #à implémenter plus tard au cas où
    #if len(torch.unique(y_pred))!=len(torch.unique(y)):
    #   try :
    #       y_pred
    
    return compute_meandice(y_pred_ohe, y_ohe, include_background=False) #we don't compare the dice of the 0 zone
    
########################################### MSE #############################################

def MSE(y_pred, y_true):
    return vxm.losses.MSE().loss(y_pred, y_true)

########################################### Gradient #############################################

class Grad:
    """
    N-D gradient loss.
    We overwrite this loss for a 2D input or 3D input and create the flow_fiel null directly here
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult
        
    def loss(self, flow_field): 
        
        flow_field_zero=torch.zeros(flow_field.size())
        
        if len(flow_field.shape)==5 : #3D scanners
            return vxm.losses.Grad(self.penalty).loss(flow_field_zero, flow_field)
        
        else : #necessarly 4
            dy = torch.abs(flow_field[:, :, 1:, :] - flow_field[:, :, :-1, :])
            dx = torch.abs(flow_field[:, :, :, 1:] - flow_field[:, :, :, :-1])
        
            if self.penalty == 'l2':
                dy = dy * dy
                dx = dx * dx

            d = torch.mean(dx) + torch.mean(dy)
            grad = d / 3.0

            if self.loss_mult is not None:
                grad *= self.loss_mult
            return grad

