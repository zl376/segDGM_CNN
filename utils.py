import nibabel as nib
import numpy as np
import itertools
from keras.utils import np_utils
from sklearn.feature_extraction.image import extract_patches as sk_extract_patches
from matplotlib import pyplot as plt
import bcolz
import random
from skimage import transform

precision_global = 'float32'

# General utils for loading and saving data
def read_data(case_idx, type_name, loc='datasets'):
    file_name = '{0}/{1}/{2}.nii.gz'.format(loc, type_name, case_idx)
    return nib.load(file_name).get_data()

def save_data(data, case_idx, type_name, loc='results'):
    file_name_ex = '{0}/{1}/{2}.nii.gz'.format('datasets', 'QSM', case_idx)
    file_name = '{0}/{1}/{2}.nii.gz'.format(loc, type_name, case_idx)
    nib.save(nib.Nifti1Image(data.astype('uint8'), None, nib.load(file_name_ex).header), file_name)

def save_array(file_name, arr):
    c = bcolz.carray(arr, rootdir=file_name, mode='w')
    c.flush()
    
def load_array(file_name):
    return bcolz.open(file_name)[:]


# Data preparation utils
def extract_patches(volume, patch_shape, extraction_step) :
    patches = sk_extract_patches(
        volume,
        patch_shape=patch_shape,
        extraction_step=extraction_step)
    
    ndim = len(volume.shape)
    npatches = np.prod(patches.shape[:ndim])
    return patches.reshape((npatches, ) + patch_shape)

def build_set(input_vols, label_vols, extraction_step=(9, 9, 9), patch_shape=(27, 27, 27), predictor_shape=(9, 9, 9), mask=None, num_dim_aux=0):
    #patch_shape = (27, 27, 27)
    #label_selector = [slice(None)] + [slice(9, 18) for i in range(3)]
    label_selector = [slice(None)] + [slice(int((patch_shape[i]-predictor_shape[i])/2), int((patch_shape[i]-predictor_shape[i])/2+predictor_shape[i])) for i in range(3)]
    
    num_classes = len(np.unique(label_vols))
    num_channel = input_vols.shape[1]

    # Extract patches from input volumes and ground truth
    x = np.zeros((0, num_channel) + patch_shape, dtype=precision_global)
    y = np.zeros((0, predictor_shape[0]*predictor_shape[1]*predictor_shape[2], num_classes), dtype=precision_global)
    if type(mask) is not type(None):
        x_add = np.zeros((0, num_channel) + patch_shape, dtype=precision_global)
        y_add = np.zeros((0, predictor_shape[0]*predictor_shape[1]*predictor_shape[2], num_classes), dtype=precision_global)
        
    
    for idx in range(len(input_vols)) :
        #print(idx)
        y_length = len(y)

        label_patches = extract_patches(label_vols[idx], patch_shape, extraction_step)
        label_patches = label_patches[label_selector]

        # Select only those who are important for processing
        valid_idxs = np.where(np.sum(label_patches, axis=(1, 2, 3)) != 0)
        
        # Enforce including patches covering mask
        if type(mask) is not type(None):
            mask_patches = extract_patches(mask[idx], patch_shape, extraction_step)
            mask_patches = mask_patches[label_selector]
            valid_idxs_mask = np.where((np.sum(mask_patches, axis=(1, 2, 3)) != 0))
            valid_idxs_add = set2idx(idx2set(valid_idxs_mask) - idx2set(valid_idxs))
            
            y_add_length = len(y_add)
            label_patches_add = label_patches[valid_idxs_add]
            x_add = np.vstack((x_add, np.zeros((len(label_patches_add), num_channel) + patch_shape, dtype=precision_global)))
            y_add = np.vstack((y_add, np.zeros((len(label_patches_add), predictor_shape[0]*predictor_shape[1]*predictor_shape[2], num_classes), dtype=precision_global)))     
            for i in range(len(label_patches_add)) :
                y_add[i+y_add_length, :, :] = np_utils.to_categorical(label_patches_add[i, : ,: ,:], num_classes).reshape((-1, num_classes))
            del label_patches_add

            # Sampling strategy: reject samples which labels are only zero
            for i_channel in range(num_channel):
                input_patches_add = extract_patches(input_vols[idx, i_channel], patch_shape, extraction_step)
                x_add[y_add_length:, i_channel, :, :, :] = input_patches_add[valid_idxs_add]
            del input_patches_add
        
        # Filtering extracted patches
        label_patches = label_patches[valid_idxs]

        # Extend volume
        x = np.vstack((x, np.zeros((len(label_patches), num_channel) + patch_shape, dtype=precision_global)))
        y = np.vstack((y, np.zeros((len(label_patches), predictor_shape[0]*predictor_shape[1]*predictor_shape[2], num_classes), dtype=precision_global)))
        
        for i in range(len(label_patches)) :
            y[i+y_length, :, :] = np_utils.to_categorical(label_patches[i, : ,: ,:], num_classes).reshape((-1, num_classes))

        del label_patches

        # Sampling strategy: reject samples which labels are only zero
        for i_channel in range(num_channel):
            input_patches = extract_patches(input_vols[idx, i_channel], patch_shape, extraction_step)
            x[y_length:, i_channel, :, :, :] = input_patches[valid_idxs]
        del input_patches

    if type(mask) is not type(None):
        x = np.concatenate((x, x_add))
        y = np.concatenate((y, y_add))
        
    # Crop auxillary dimension
    if num_dim_aux != 0:
        x, aux = extract_aux(x, predictor_shape, num_dim_aux)
        
        return x, y, aux
    else:
        return x, y

    
def extract_aux(data, predictor_shape, num_dim_aux):
    num_channel, *patch_shape = data.shape[1:]
    aux_selector = [slice(None)] + [slice(num_channel-num_dim_aux, num_channel)] + [slice(int((patch_shape[i]-predictor_shape[i])/2), int((patch_shape[i]-predictor_shape[i])/2+predictor_shape[i])) for i in range(3)]
    aux = data[aux_selector]
    data = data[:,:num_channel-num_dim_aux,:,:,:]
    
    return data, aux
    
    
    

# Random shuffle (1st dim)
def shuffle(data, idxs = None):
    N = len(data)
    
    if idxs == None:
        idxs = list(range(N))
        for i in range(N-1, -1, -1):
            j = random.randint(0, i)
            idxs[i], idxs[j] = idxs[j], idxs[i]
    
    idxs_target = [0] * N
    for i, idx in enumerate(idxs):
        idxs_target[idx] = i
    
    for i in range(N):
        while i != idxs_target[i]:
            j = idxs_target[i]
            data[i], data[j] = data[j], data[i]
            idxs_target[i], idxs_target[j] = idxs_target[j], idxs_target[i]
        
    return idxs
        
# Normalize each patch (1st dim) in each channel (2nd dim)
def norm_patch(data):
    #mean_data = np.tile(np.mean(data, (2,3,4)).reshape(data.shape[:2]+(1,1,1)), (1,1,)+data.shape[2:])
    #std_data = np.tile(np.std(data, (2,3,4)).reshape(data.shape[:2]+(1,1,1)), (1,1,)+data.shape[2:])
    mean_data = np.mean(data, (2,3,4))
    std_data = np.std(data, (2,3,4))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if std_data[i,j] != 0:
                data[i,j,:,:,:] = (data[i,j,:,:,:] - mean_data[i,j])/std_data[i,j]
    
        
    
def idx2set(data_idx):
    return set([ tuple(map(int, y)) for y in zip(*[ x for x in data_idx ]) ])
    
def set2idx(data_set):
    return tuple( np.asarray(y) for y in zip(*[ x for x in data_set ]))
    

# Reconstruction utils
def generate_indexes(patch_shape, expected_shape, pad_shape) :
    ndims = len(patch_shape)

    #poss_shape = [patch_shape[i+1] * (expected_shape[i] // patch_shape[i+1]) for i in range(ndims-1)]
    poss_shape = [patch_shape[i+1] * ((expected_shape[i]-pad_shape[i]*2) // patch_shape[i+1]) + pad_shape[i]*2 for i in range(ndims-1)]

    #idxs = [range(patch_shape[i+1], poss_shape[i] - patch_shape[i+1], patch_shape[i+1]) for i in range(ndims-1)]
    idxs = [range(pad_shape[i], poss_shape[i] - pad_shape[i], patch_shape[i+1]) for i in range(ndims-1)]
    
    return itertools.product(*idxs)

def reconstruct_volume(patches, expected_shape, pad_shape=(9, 9, 3)) :
    patch_shape = patches.shape

    assert len(patch_shape) - 1 == len(expected_shape)

    reconstructed_img = np.zeros(expected_shape, dtype=precision_global)

    for count, coord in enumerate(generate_indexes(patch_shape, expected_shape, pad_shape)) :
        selection = [slice(coord[i], coord[i] + patch_shape[i+1]) for i in range(len(coord))]
        reconstructed_img[selection] = patches[count]

    return reconstructed_img

# Image manipulation utils
def resize_and_crop(img, matrix_size, voxel_size, matrix_size_new, voxel_size_new):
    method = 'nearest'
    
    matrix_size_tmp = tuple( int(round(matrix_size[i] * voxel_size[i] / voxel_size_new[i])) for i in range(3) )
    img_tmp = transform.resize(img, matrix_size_tmp, order=0, preserve_range=True, mode='symmetric')
    
    pad_L = [ max(int((matrix_size_tmp[i] - matrix_size_new[i])/2),0) for i in range(3) ]
    pad_R = [ pad_L[i] + min(matrix_size_tmp[i], matrix_size_new[i]) for i in range(3) ]
    pad_L_new = [ max(int((matrix_size_new[i] - matrix_size_tmp[i])/2),0) for i in range(3) ]
    pad_R_new = [ pad_L_new[i] + min(matrix_size_tmp[i], matrix_size_new[i]) for i in range(3) ]
    
    res = np.zeros(matrix_size_new, dtype=img.dtype)
    res[pad_L_new[0]:pad_R_new[0], pad_L_new[1]:pad_R_new[1], pad_L_new[2]:pad_R_new[2]] = \
        img_tmp[pad_L[0]:pad_R[0], pad_L[1]:pad_R[1], pad_L[2]:pad_R[2]]
    
    return res

def scale(img, window):
    val_min, val_max = window
    
    res = np.copy(img)
    res[res < val_min] = val_min
    res[res > val_max] = val_max
    
    res = (res - val_min) / (val_max - val_min)
    
    res = (res * 256).astype(int)
    res[res == 256] = 255
    
    return res


    
    
# Utils for plotting
def plots(ims, figsize=(12,6), rows=1, scale=None, interp=False, titles=None):
    
    if scale != None:
        lo, hi = scale
        ims = (ims - lo)/(hi - lo) * 255
        
    if(ims.ndim == 2):
        ims = np.tile(ims, (1,1,1));
    
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if(ims.shape[-1] != 3):
            ims = np.tile(ims[:,:,:,np.newaxis], (1,1,1,3));
            
    #print(ims.shape)
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')