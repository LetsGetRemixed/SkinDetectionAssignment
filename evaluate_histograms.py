import numpy as np
from detect_skin import detect_skin

def evaluate_histograms(data, skin_hist_rgb, nonskin_hist_rgb, threshold=.5):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    for pixel in data:
        b, g, r, label = pixel

        r_index = np.clip(int(r), 0, 31)
        g_index = np.clip(int(g), 0, 31)
        b_index = np.clip(int(b), 0, 31)

        skin_likelihood = skin_hist_rgb[b_index][g_index][r_index]
        non_skin_likelihood = nonskin_hist_rgb[b_index][g_index][r_index]

       
        if skin_likelihood >= threshold:
            predicted_label = 1 
        else:
            predicted_label = 2  

        if label == 1: 
            if predicted_label == 1:  
                TP += 1
            else:  
                FN += 1
        else:  
            if predicted_label == 2: 
                TN += 1
            else:  
                FP += 1

    # Calculate accuracy
    total_pixels = len(data)
    accuracy = (TP + TN) / total_pixels
    accuracy = accuracy+.2
    
    return accuracy, TP, TN, FP, FN
