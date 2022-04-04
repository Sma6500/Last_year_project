# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         Resize                                          | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

import numpy as np

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             Due to computation capacity issues                        | #
# |                   I had to resize the scanners, that's the purpose of this file       | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #



def pooling3d(image, kernel, strides, technique = 'avg'):
    x,y,z = image.shape
    kx, ky, kz = kernel
    sx, sy, sz = strides
    dimx, dimy, dimz = (x-kx)//sx+1, (y-ky)//sy+1, (z-kz)//sz+1
    new_image = np.zeros((dimx,dimy,dimz))
    for i in range(dimx):
        for j in range(dimy):
            for k in range(dimz):
                kernel_window = image[sx*i:sx*i+kx,sy*j:sy*j+ky,sz*k:sz*k+kz]
                if technique=='most':
                    values, count = np.unique(kernel_window, return_counts=True)
                    new_image[i,j,k] = values[np.argmax(count)]
                else:
                    new_image[i,j,k] = np.mean(kernel_window)
    return new_image

def addPadding(image, padding):
    shape = np.shape(image)
    new_image = np.zeros(tuple(map(sum, zip(shape, padding, padding))))
    new_image[padding[0]:padding[0]+shape[0],padding[1]:padding[1]+shape[1],padding[2]:padding[2]+shape[2]] = image
    return new_image


if __name__ == '__main__':
#script to resize the image
    import nibabel as nib
    import os
    from tqdm import tqdm
    path_data_dir="/home/luther/Documents/Projet_3A/data/L2R_2021_Task3_test/mask"
    path_new_data_dir='/home/luther/Documents/Projet_3A/data/L2R_2021_Task3_test/reduced_data'
    for dir_name in tqdm(os.listdir(path_data_dir)):
        image=nib.load(os.path.join(path_data_dir,dir_name,"aligned_norm.nii.gz"))
        mask=nib.load(os.path.join(path_data_dir,dir_name,"aligned_seg4.nii.gz"))

        processed_image = addPadding(pooling3d(image.get_fdata(), (2,2,2), (2,2,2), technique='avg'),(8,0,8))
        processed_mask = addPadding(pooling3d(mask.get_fdata(), (2,2,2), (2,2,2), technique='most'),(8,0,8))

        ni_img = nib.Nifti1Image(processed_image, image.affine)
        ni_mask= nib.Nifti1Image(processed_mask, mask.affine)
        
        os.makedirs(os.path.join(path_new_data_dir, dir_name))
        nib.save(ni_img, os.path.join(path_new_data_dir, dir_name, 'aligned_norm.nii.gz'))
        nib.save(ni_mask, os.path.join(path_new_data_dir, dir_name, 'aligned_seg4.nii.gz'))

        
    
    
