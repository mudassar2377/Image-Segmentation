import numpy as np
import os
import cv2
from scipy.ndimage import sobel

def gamma_correction(img,gamma):
    img = img/255
    img = img**gamma
    img = img*255
    return img

def mask_to_class(path):
    mask_image = cv2.imread(os.path.join(path))  

    mask_array = np.array(mask_image)
    color_to_label = {
        (108, 0, 115): 0,    
        (145, 1, 122): 1,
        (216, 47, 148): 2,    
        (254, 246, 242): 3,
        (181, 9, 130): 4,    
        (236, 85, 157): 5,
        (73, 0, 106): 6,    
        (248, 123, 168): 7,
        (0, 0, 0): 8,    
        (127, 255, 255): 9,
        (127, 255, 142): 10,    
        (255, 127, 127): 11,
    }

    converted_mask = np.zeros(mask_array.shape[:2], dtype=np.float32)
    mask_array = cv2.cvtColor(mask_array, cv2.COLOR_BGR2RGB)
    for i in range(mask_array.shape[0]):
        for j in range(mask_array.shape[1]):
            rgb = tuple(mask_array[i, j])
            converted_mask[i, j] = color_to_label.get(rgb, 0)      
    # converted_mask = converted_mask/11
    converted_mask = cv2.resize(converted_mask,(128,128))
    return converted_mask

def gradient_mag(image_array):
    image_array = image_array / 255.0

    # Split the image into separate channels (R, G, B)
    red_channel = image_array[:, :, 0]
    green_channel = image_array[:, :, 1]
    blue_channel = image_array[:, :, 2]

    # Apply the Sobel filter to each channel separately
    gradient_x_red = sobel(red_channel, axis=0)
    gradient_y_red = sobel(red_channel, axis=1)

    gradient_x_green = sobel(green_channel, axis=0)
    gradient_y_green = sobel(green_channel, axis=1)

    gradient_x_blue = sobel(blue_channel, axis=0)
    gradient_y_blue = sobel(blue_channel, axis=1)

    # Calculate the magnitude of the gradient for each channel
    gradient_magnitude_red = np.sqrt(gradient_x_red ** 2 + gradient_y_red ** 2)
    gradient_magnitude_green = np.sqrt(gradient_x_green ** 2 + gradient_y_green ** 2)
    gradient_magnitude_blue = np.sqrt(gradient_x_blue ** 2 + gradient_y_blue ** 2)

    # Stack the gradient magnitudes to form a 3D result
    gradient_magnitude_3d = np.stack(
        (gradient_magnitude_red, gradient_magnitude_green, gradient_magnitude_blue), axis=2
    )
    return gradient_magnitude_3d

def img_read(path):
    img = cv2.imread(os.path.join(path))
    img = gamma_correction(img,3)
    new_img = np.zeros(img.shape,dtype=np.float64)
    new_img = img/255.0 
    img = cv2.resize(new_img,(128,128))
    return img

os.system('clear')

train_imgs = np.zeros((1050,128,128,3),dtype=np.float32)
train_labels = np.zeros((1050,128,128),dtype=np.float32)
test_imgs = np.zeros((450,128,128,3),dtype=np.float32)
test_labels = np.zeros((450,128,128),dtype=np.float32)
train_classification_labels = []
test_classification_labels = []

p1 = '/home/sami/Downloads/DIP/DIP Project/Queensland Dataset CE42/Training'
p2 = '/home/sami/Downloads/DIP/DIP Project/Queensland Dataset CE42/Testing'

tr_types = os.listdir(p1)
ts_types = os.listdir(p2)

ts_types.sort()
tr_types.sort()

tr_count = 0
ts_count = 0

for a in range(len(tr_types)):
    # Training Data
    train_img_path = p1 + '/' + tr_types[a] + '/Images'
    train_labels_path = p1 + '/' + tr_types[a] + '/Masks'
    train_imgs_name = os.listdir(train_img_path)
    train_labels_name = os.listdir(train_labels_path)   

    # Testing Data
    test_imgs_path = p2 + '/' + ts_types[a] + '/Images'
    test_labels_path = p2 + '/' + ts_types[a] + '/Masks'
    test_imgs_name = os.listdir(test_imgs_path)
    test_labels_name = os.listdir(test_labels_path)  

    
    for b in range(len(train_imgs_name)):
        tr_img_pat = train_img_path + '/' + train_imgs_name[b]
        tr_lbl_pat = train_labels_path + '/' + train_labels_name[b]
        print(b + tr_count)
        train_imgs[b + tr_count, :, :, :] = img_read(tr_img_pat)
        train_labels[b + tr_count, :, :] = mask_to_class(tr_lbl_pat)
        train_classification_labels.append(a)       

    os.system('clear')

    for c in range(len(test_imgs_name)):
        ts_img_pat = test_imgs_path + '/' + test_imgs_name[c]
        ts_lbl_pat = test_labels_path + '/' + test_labels_name[c]
        
        print(c + ts_count)

        test_imgs[c + ts_count, :, :, :] = img_read(ts_img_pat)
        test_labels[c + ts_count, :, :] = mask_to_class(ts_lbl_pat)
        test_classification_labels.append(a)
    
    os.system('clear')

    tr_count += len(train_imgs_name)
    print(tr_count)
    ts_count += len(test_imgs_name)

print(train_imgs.shape)
print(train_labels.shape)
print(test_imgs.shape)
print(test_labels.shape)

np.save('train_imgs.npy',train_imgs)
np.save('train_labels.npy',train_labels)
np.save('test_imgs.npy',test_imgs)
np.save('test_labels.npy',test_labels)
np.save('train_classification_labels.npy',train_classification_labels)
np.save('test_classification_labels.npy',test_classification_labels)