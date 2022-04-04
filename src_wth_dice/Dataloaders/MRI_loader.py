import os

import nibabel as nib

from PIL import Image

from numpy import asarray

from process.utils import to_vector

"""
A dictionarie is created:
- partition:
    * in partition['train'] a list of training IDs (== path to the folder containing the mri files)
    * in partition['validation'] a list of validation IDs
"""

def get_labels_list(dataset_path):
    """
    get the root list of all MRI files we use for training and validation
    """
    labels = os.listdir(dataset_path)
    try:
        labels.remove('LICENSE')
        labels.remove('README.md')
    except:
        pass
    return labels



def get_data_dicts(dataset_path, valid_ratio):
    """
    This function will be called in the get_dataloaders function from registration_loader.py
    """
    labels_list = get_labels_list(dataset_path)
    train_size=int(len(labels_list)*(1-valid_ratio))
    train_ids, valid_ids = [], []
    for i,label in enumerate(labels_list):
        if i < train_size:
            train_ids.append(os.path.join(dataset_path, label))
        else :
            valid_ids.append(os.path.join(dataset_path, label))
    partition = {'train': train_ids, 'validation': valid_ids}
    return partition


def load(path_fixed,path_moving, vectorize=False):
    
    #for loading 3D scans
    #fixed_image=nib.load(os.path.join(path_fixed,"aligned_norm.nii.gz"))
    #moving_image=nib.load(os.path.join(path_moving,"aligned_norm.nii.gz"))
    #fixed_mask=nib.load(os.path.join(path_fixed,"aligned_seg4.nii.gz"))
    #moving_mask=nib.load(os.path.join(path_moving,"aligned_seg4.nii.gz"))

    #for loading 2D Jpeg and no mask
    fixed_image=asarray(Image.open(path_fixed))
    moving_image=asarray(Image.open(path_moving))
                           
    #for loading 2D scans
    #fixed_image=nib.load(os.path.join(path_fixed,"slice_norm.nii.gz"))
    #moving_image=nib.load(os.path.join(path_moving,"slice_norm.nii.gz"))
    #fixed_mask=nib.load(os.path.join(path_fixed,"slice_seg4.nii.gz"))
    #moving_mask=nib.load(os.path.join(path_moving,"slice_seg4.nii.gz"))

    if vectorize :
        return to_vector(fixed_image), to_vector(moving_image)
        
    return fixed_image, moving_image




