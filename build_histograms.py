import numpy as np

import numpy as np

def build_histograms(data):
    skin_histogram = np.zeros((32, 32, 32), dtype=int)
    non_skin_histogram = np.zeros((32, 32, 32), dtype=int)
    
    for pixel in data:
        b, g, r, label = pixel
        rgb_pixel = (r, g, b)
        
        # skin er not skin bub????
        if label == 1:
            histogram = skin_histogram
        elif label == 2:
            histogram = non_skin_histogram
        else:
            continue  
        
        r_bin = int(rgb_pixel[0] * 31 / 255)
        g_bin = int(rgb_pixel[1] * 31 / 255)
        b_bin = int(rgb_pixel[2] * 31 / 255)
        
        
        histogram[r_bin, g_bin, b_bin] += 1
    
    skin_histogram = skin_histogram / np.sum(skin_histogram)
    non_skin_histogram = non_skin_histogram / np.sum(non_skin_histogram)
    
    return skin_histogram, non_skin_histogram
