#from class Model, keys are : net, optimizer, device, criterion, scheduler (not mandatory)

import torch
import os

os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm
from process.losses import compute_dice, MSE, Grad

class Model: #very basic, need inheritage later and a more complexe build, but it's not necessary right now
    
    def __init__(self, model_config, criterion_config, scheduler_config):#, optimizer_config, scheduler_config=None):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = vxm.networks.VxmDense(inshape=model_config['inshape'],
                                         nb_unet_features=model_config['nb_unet_features'],
                                         src_feats=model_config['src_feats'],
                                        trg_feats=model_config['trg_feats']).to(self.device)
        #make training easier
        self.net=self.net.float()
        self.optimizer=torch.optim.Adam(self.net.parameters(), 1e-3)
        
        self.scheduler=self.init_scheduler(scheduler_config)
        #pour ce premier essai on config directement ici, mais pour plus tard on écrira une généralisation
        #self.optimizer_config = optimizer_config
        self.criterion_config = criterion_config
        #self.scheduler_config = scheduler_config
        
    def criterion(self, pred_image, fixed_image, flow_field):
        
        #MSE
        if self.criterion_config['MSE'] is not(None) :
            image_loss=MSE(pred_image, fixed_image)*self.criterion_config['MSE']
            loss=image_loss
            
        #Gradient of flow field
        if self.criterion_config['Grad'] is not(None) :
            regularization_loss=Grad(self.criterion_config['Grad']['Norm']).loss(flow_field)*self.criterion_config['Grad']['weight']
            loss+=regularization_loss
        
        return loss
    
    
    def init_scheduler(self, scheduler_config):
        
        if scheduler_config['scheduler']=='ROP':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                                   mode=scheduler_config['mode'],
                                                                  factor=scheduler_config['factor'],
                                                                  patience=scheduler_config['patience'],
                                                                  threshold=scheduler_config['threshold'],
                                                                  verbose=scheduler_config['verbose'])
            return scheduler
        else : 
            print("scheduler badly configured")
            return None
            
        