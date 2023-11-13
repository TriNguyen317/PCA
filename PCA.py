import numpy as np
import os
import cv2
from sklearn.cluster import KMeans
import time


#Đọc ảnh từ thư mục
def ReadImage(Path):
    image_list = []
    file_list = os.listdir(Path)
    
    for name in file_list:
        if name.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(Path, name)
            try:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image_list.append(image)
                
            except Exception as e:
                print("Thư mục chứa tệp không phải ảnh: {}".format(image_path))
        
    return image_list

def Histogram(image):
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    return histogram
#Preprocess
def Preprocess(image_list):
    shape = image_list.shape
    image_list = image_list.astype("float16") / 255
    image_list = np.reshape(image_list,(shape[0], shape[1]*shape[2]))
    return image_list
""" 
PCA(feature, n_component) 
    feature: Vector đặc trưng
    n_component: Số chiều muốn lấy. Default: 1/3 số chiều của feature.
"""
    
def PCA(feature, n_components=0):
    X = feature.T
    if n_components == 0:
        n_components = int(feature.shape[1] / 3)
    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0)
    standardized_data = (X - mean) / std_dev
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
    total = sum(eigen_values)
    explained_variance_ratio = [(eigen/total)*100 for eigen in eigen_values[:n_components]] 
    ratio = np.sum(explained_variance_ratio)
    
    # Giảm chiều dữ liệu
    components = eigen_vectors[:, :n_components]

    # Biến đổi dữ liệu gốc sang không gian mới
    transformed = np.dot(feature, components)

    return transformed, ratio

def main():
    #feature_extraction = ReadImage(PATH) # Path là đường dẫn đến folder chứa ảnh
    # Đang mặc định để thử hàm
    feature_extraction = np.random.randint(0,255,(50000,100,30))
    feature = Preprocess(feature_extraction)
    print("---Lay thanh phan chinh bang pca---")
    #data_PCA = PCA(feature)
    start = time.time()
    data_PCA = PCA(feature,n_components=2545)


    end = time.time()

    
    
    
    start = time.time()
    kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(feature)
    end = time.time()
    print ("So chieu dau vao khong su dung PCA: {}".format(feature.shape[1]))
    print ("Thoi gian chay khi khong su dung PCA: {}".format(end-start))
    print()
    
    start = time.time()
    print("Lượng data được giữ lại: {}%".format(data_PCA[1]))
    print ("So chieu dau vao su dung PCA: {}".format(data_PCA[0].shape[1]))
    kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(data_PCA[0])
    
    end = time.time()
    print ("Thoi gian chay khi su dung PCA: {}".format(end-start))

if __name__=="__main__":
    main()