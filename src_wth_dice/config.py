######################################### DATALOADER ##########################################
from torchvision import transforms
from monai.transforms import AddChannel

#careful : if vectorize is True the transformations will be apply on each axes of the vectorized scanners independantly
t=transforms.Compose([transforms.ToTensor()])#, AddChannel()])
#'/home/luther/Documents/Projet_3A/data/L2R_2021_Task3_test/reduced_data'
dataloader_config = {
    'rootdir': '/home/luther/Documents/Projet_3A/data/2D_data/MedNIST/Hand/',
    'batch_size': 8,
    'valid_ratio':0.2,
    'num_workers': 1, # for loading data in parallel. Should be <= nb cores in CPU.
    'transformation':t,
    'vectorize':False #Vectorize or not
    'a'=1.5
}


########################################### TRAIN #############################################
    
train_config = {
    'nb_epochs' : 1000, # arbitrary high
    'checkpoints_path': 'Documents/Projet_3A/VectorMorph/models/', 
    'verbose': True,
    'checkpoint':50 #save the weights every 50 epochs
}

########################################### Model #############################################

model_config = {
    #'inshape' : (224,160,192), # arbitrary high
    #'inshape' : (160,192),
    'inshape':(64,64),
    'nb_unet_features':[[16, 32, 32, 32],[32, 32, 32, 32, 32, 16, 16]], #[[encoder], [decoder]]
    #'nb_unet_features':[[16, 32, 32],[32, 16, 16]], #[[encoder], [decoder]]
    'src_feats': 1, #1 for non-vectorized src scanners, else 3
    'trg_feats':1, # ""
}
########################################### criterion config #############################################

criterion_config = {
    'MSE' : 1, # eventual features and weights of losses (put None instead of 0 if you don't want the loss to be compute)
    'Grad' : {'Norm':'l2','weight':0.05}
}


######################################### SCHEDULER ###########################################

scheduler_config = {
    'scheduler': 'ROP', # ReduceOnPlateau: when the loss isn't decreasing for too long, reduce lr only ROP is configure for now
    'mode': 'min', # we want to detect a decrease and not an increase. 
    'factor': 0.2, # when loss has stagnated for too long, new_lr = factor*lr
    'patience': 10,# how long to wait before updating lr
    'threshold': 0.00001, # min lr
    'verbose': True
}
