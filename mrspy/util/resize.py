def center_crop_kspace(kspace_data, crop_size):
    # Assuming kspace_data is a square 2D tensor 
    # and crop_size is the size of one dimension of the square crop
    center = kspace_data.shape[0] // 2
    half_crop = crop_size // 2
    
    # Define the indices for cropping
    start = center - half_crop
    end = center + half_crop
    
    # Crop the k-space data
    return kspace_data[start:end, start:end]
