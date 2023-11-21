import numpy as np
import os
import cv2
from sklearn.cluster import KMeans
import time
from skimage.transform import resize
from skimage.feature import hog
from skimage import img_as_ubyte

#Đọc ảnh từ thư mục
def ReadImage(Path, size):
    image_list = []
    label= []
    file_list = os.listdir(Path)
    
    for y, file_name in enumerate(file_list):
        dir = os.path.join(Path, file_name)
        image_dir =  os.listdir(dir)
        for name in image_dir:
            if name.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(dir, name)
                try:
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    image = resize(image, (size,size))
                    image = img_as_ubyte(image)
                    image_list.append(image)
                    label.append(y)
                except Exception as e:
                    print("Thư mục chứa tệp không phải ảnh: {}".format(image_path))
        
    return image_list, label

def Histogram(image):
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    return histogram
#Preprocess
def Preprocess(image_list):
    resized_img_list = []
    for image in image_list:
        resized_img = resize(image, (round(image.shape[0]/2), round(image.shape[1]/2)))
        resized_img_list.append(resized_img)
    resized_img_list = np.array(resized_img_list)
    shape = resized_img_list.shape
    resized_img_list = resized_img_list.astype("float") / 255
    resized_img_list = np.reshape(resized_img_list,(shape[0], shape[1]*shape[2]))
    return resized_img_list
 
""" 
PCA(feature, n_component) 
    feature: Vector đặc trưng
    n_component: Số chiều muốn lấy 
"""
    
def PCA(feature, keep=0.9):
    if feature.ndim > 2:
        feature = Preprocess(feature)
    feature = feature.T

    #feature = Preprocess(feature)
    mean = np.mean(feature, axis=0)
    std_dev = np.std(feature, axis=0)
    standardized_data = (feature - mean) / std_dev
    # Tính ma trận hiệp phương sai
    cov_matrix = np.cov(standardized_data)

    # Tính vector riêng và giá trị riêng
    eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)

    # Sắp xếp vector riêng theo giá trị riêng giảm dần
    sorted_index = np.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[sorted_index]
    eigen_vectors = eigen_vectors[:, sorted_index]

    # Chọn số lượng thành phần chính cần giữ lại

    # Tính toán phần trăm thông tin giữ lại
    total = np.sum(eigen_values)
    explained_var = np.cumsum(eigen_values) / np.sum(eigen_values)
    n_components = np.argmax(explained_var >= keep) + 1
    ratio = explained_var[n_components - 1] * 100
    
    # Giảm chiều dữ liệu
    components = eigen_vectors[:, :n_components]

    # Biến đổi dữ liệu gốc sang không gian mới
    transformed = np.dot(standardized_data.T, components)

    return transformed, ratio
