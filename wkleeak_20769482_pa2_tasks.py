# COMP2211 
# 2023S
# Programming Assignment 2

import numpy as np

# Task 1
def contrast_stretch(x):
  """ Stretch the contrast of each individual images in the given data array.

  Parameters
  ------------
  x : np.ndarray  My notes (x_train_raw.shape (537, 64, 64, 1))
      Image data array
  Returns:
  -------------
  np.ndarray
      New image data arary with individually contrast-stretched images.
  """
  ### START YOUR CODE HERE
  x_one = x.shape[0]  #copy the dimension
  x_two = x.shape[1]
  x_three =x.shape[2]
  x_four = x.shape[3]

  x_enhanced = np.zeros(x.shape).astype(int)
  

  for i in range(x.shape[0]):
    I_Max = np.max(x[i])
    I_Min = np.min(x[i])
    denominator = I_Max - I_Min
    x_enhanced[i] = ((x[i] - I_Min)/denominator)*255   
  
  ### END YOUR CODE HERE
  return x_enhanced
  
# Task 2
def rescale_01(x):
  """ Rescales the given image data array to range [0,1].

  Parameters
  ------------
  x : np.ndarray
      image data array

  Returns:
  -------------
  np.ndarray
      New image data arary re-scaled to range [0,1].
  """
  ### START YOUR CODE HERE
  x_01 = x.astype(float) / 255

  ### END YOUR CODE HERE
  return x_01.astype(float)  
  
# Task 7
def threshold(val_preds, thresh_value):
    """Threshold the given predicted mask array.

    Parameters
    ----------
    val_preds : np.ndarray
        Predicted segmentation array on validation data
    thresh_value : float

    Returns
    ----------
    np.ndarray
        Thresholded val_preds
    """
    ### START YOUR CODE HERE
    val_preds_thresh = (val_preds > thresh_value).astype(int)
    ### END YOUR CODE HERE
    return val_preds_thresh.astype(int)
    
# Task 8
def dice_coef(mask1, mask2):
    """Calculate the dice coeffecient score between two binary masks.

    Parameters
    ----------
    mask1 : np.ndarray
        binary mask that consists of either 0 or 1.
    mask2 : np.ndarray
        binary mask that consists of either 0 or 1.

    Returns
    ----------
    float
        dice coefficient between mask1 and mask2.
    """
    ### START YOUR CODE HERE
    mask1_area = np.sum(mask1)
    mask2_area = np.sum(mask2)
    intercept_area = np.multiply(mask1,mask2)
    dice_coef_score = (2*np.sum(intercept_area))/(mask1_area+mask2_area)


    ### END YOUR CODE HERE
    return dice_coef_score
# Task 9
def avg_dice(y_val, val_preds_thresh):
    """Calculates the average dice coefficient score across all thresholded predictions & label pair of the validation dataset.

    Parameters
    ----------
    y_val : np.ndarray
        Ground truth segmentation labels array of the validation dataset
    val_preds : np.ndarray
        Predicted segmentation masks array on the validation dataset

    Returns
    ----------
    float
        Average dice score coefficient. 
    """ 
    ### START YOUR CODE HERE
    average_dice = 0
    for i in range(y_val.shape[0]):
      average_dice += dice_coef(y_val[i], val_preds_thresh[i])
    ### END YOUR CODE HERE
    return average_dice/y_val.shape[0]    


# if __name__ == '__main__':