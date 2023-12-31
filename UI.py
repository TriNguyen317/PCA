import tkinter as tk
from tkinter import filedialog
from PCA import *
from tkinter import font

directory_path = ""
feature = None
label = None
num_label = None

# Chọn thư mục ảnh
'''Cấu trúc thư mục:
    - Thư mục chính
        - Class 1
            - img1.jpg
            - img2.jpg
        - Class 2
            - img1.jpg
            - img2.jpg
        - Class 3
            - img1.jpg
            - img2.jpg
'''
def select_directory():
    global directory_path
    try:
        global num_label
        directory_path = filedialog.askdirectory(title="Giảm chiều dữ liệu trong trích xuất đặc trưng ảnh ")
        if directory_path:
            status_label.config(text=f"Hoàn thành! Đường dẫn đã chọn: {directory_path}")
            global data
            global label
            size = int(get_size_image.get())
            # Lấy ảnh và nhãn của mỗi ảnh trong thư mục theo cấu trúc
            data, label = ReadImage(directory_path, size)
            data = np.array(data)
            label = np.array(label)
            shape_input.config(text=f"Số chiều đầu vào: ({data.shape[0]},{data.shape[1]},{data.shape[2]})")
            num_label = len(np.unique(label))
    except Exception as e:
        status_label.config(text=f"Lỗi: {str(e)}")

# Xử lý vector để tăng hiệu suất tính toán và lấy đặc trưng bằng cách flatten
'''
    - Đưa các giá trị xám về trong khoảng 0,1
    - Giảm kích thước ảnh xuống 1/2 --- vd:(250*250) --> (125,125)
'''
def preprocess_data():
    try:

        global flatten_feature
        # Lấy đặc trưng ảnh bằng cách flatten
        processed_data = Preprocess(data)
        shape_feature.config(text=f"Số chiều đặc trung lấy được: ({processed_data.shape[0]},{processed_data.shape[1]})")
        global feature
        # Gán cho dữ liệu dùng trong ứng dụng là đặc trung flatten
        flatten_feature = processed_data
        feature = flatten_feature
        print(feature.shape)

    except Exception as e:
        if str(e)=="'NoneType' object has no attribute 'shape'":
            error.config(text=f"Lỗi khi xử lý dữ liệu: Chưa chọn đường dẫn")
        else:
            error.config(text=f"Lỗi khi xử lý dữ liệu: {str(e)}")

# Lấy đặc trưng histogram 
def Histogram_data():
    try:
        global feature
        global histogram_feature
        histogram_data = []
        
        for image in data:
            histogram_data.append(Histogram(image))
            
        histogram_data = np.array(histogram_data,dtype=np.uint8)
        
        shape_feature.config(text=f"Số chiều đặc trung lấy được: ({histogram_data.shape[0]},{histogram_data.shape[1]})")
        
        feature = histogram_data.reshape(histogram_data.shape[0], histogram_data.shape[1])
        print(feature.shape)

    except Exception as e:
        if str(e)=="'NoneType' object has no attribute 'shape'":
            error.config(text=f"Lỗi khi xử lý dữ liệu: Chưa chọn đường dẫn")
        else:
            error.config(text=f"Lỗi khi xử lý dữ liệu: {str(e)}")


# Chạy thuật toán Kmean để phân lớp và tính thời gian để so sánh thời gian chạy
def time2ExecuteKMean():
    try: 
        global num_label
        print(feature)
        total = feature.shape[0]
        #Chạy PCA
        start = time.time()
        data_PCA = PCA(feature,float(ratio_PCA_input.get())/100)
        end = time.time()
        timeRun_PCA.config(text="Thoi gian chay PCA: {}".format(round(end-start,3)))
        print ("Thoi gian chay PCA: {}".format(round(end-start,3)))
        print()
    
        # Kmean với dữ liệu không sử dụng PCA
        start = time.time()
        kmeans = KMeans(n_clusters=num_label, random_state=0, n_init="auto").fit_predict(feature)
        end = time.time()
        timeRun_KMean_feature.config(text="So chieu dau vao khong su dung PCA: {}. Thoi gian chay khi khong su dung PCA: {} giay".format(feature.shape,round(end-start,3)))
        accuracy = np.sum(label == kmeans) / total
        accuracy_Kmean_feature.config(text="Độ chính xác: {}% ".format(accuracy*100))
        print ("So chieu dau vao khong su dung PCA: {}".format(feature.shape[1]))
        print ("Thoi gian chay khi khong su dung PCA: {}".format(round(end-start,3)))
        print()
        
        # Kmean với dữ liệu sử dụng PCA
        start = time.time()
        kmeans = KMeans(n_clusters=num_label, random_state=0, n_init="auto").fit_predict(data_PCA[0])
        end = time.time()
        timeRun_KMean_PCAfeature.config(text="So chieu dau vao su dung PCA: {}. Thoi gian chay khi khong su dung PCA: {} giay".format(data_PCA[0].shape,round(end-start,3)))
        accuracy = np.sum(label == kmeans) / total
        accuracy_Kmean_PCAfeature.config(text="Độ chính xác: {}% ".format(accuracy*100))
        print("Lượng data được giữ lại: {}%".format(data_PCA[1]))
        print ("So chieu dau vao su dung PCA: {}".format(data_PCA[0].shape[1]))
        print ("Thoi gian chay khi su dung PCA: {}".format(end-start))
        
    except Exception as e:
        print(e)


root = tk.Tk()

# Hiện hướng dẫn sử dụng
title_label = tk.Label(root, text="Ứng dụng sử dụng PCA để giảm chiều dữ liệu trong việc lấy đặc trưng ảnh", fg="red", font=font.Font(size=14, weight="bold"))
note_label = tk.Label(root, text="Ứng dụng lấy đặc trưng của ảnh bằng cách flatten các pixel có trong ảnh và sử dụng các đặc trưng đã lấy để phân lớp ảnh bằng thuật toán K-means")
user_manual = tk.Label(root, text="Hướng dẫn sử dụng",font=font.Font(size=12, weight="bold"))
user_manual_1 = tk.Label(root, text="1. Chọn đường dẫn chứa tập ảnh có cùng kích thước đầu vào.")
user_manual_2 = tk.Label(root, text="2. Chọn phương pháp lấy đặc trưng ảnh.")
user_manual_3 = tk.Label(root, text="2. Chọn bắt đầu dữ liệu sẽ được phân lớp bằng K-means hoặc mô hình Neural Network. Kết quả sẽ được so sánh trên cả 2 tập dữ liệu trước và sau PCA")


blank = tk.Label(root, text="")
title_label.pack()
blank.pack()
note_label.pack()

user_manual.pack()
user_manual_1.pack()
user_manual_2.pack()
user_manual_3.pack()
root.title("Chọn Đường Dẫn")

# Chức năng 1 : Chọn đường dẫn tệp ảnh
# Tạo button chọn đường dẫn
size_label = tk.Label(root, text="Nhập kích thước ảnh: ")
size_label.pack()
get_size_image = tk.Entry(root, width=30)
get_size_image.insert(0, 100)
get_size_image.pack(pady=10)

select_button = tk.Button(root, text="Chọn Thư Mục", command=select_directory)
select_button.pack(pady=20)

status_label = tk.Label(root, text="")
status_label.pack()

shape_input = tk.Label(root, text="")
shape_input.pack()

# Chức năng 2: Lấy đặc trưng ảnh
# Tạo button để chọn cách lấy vector đặc trưng
label_1 = tk.Label(root, text="Lấy đặc trưng ảnh", font=font.Font(size=12, weight="bold"))
label_1.pack()
Flatten_button = tk.Button(root, text="Flatten pixel", command=preprocess_data)
Flatten_button.pack(pady=5)

Histogram_button = tk.Button(root, text="Histogram pixel", command=Histogram_data)
Histogram_button.pack(pady=5)

shape_feature = tk.Label(root, text="")
shape_feature.pack()

error = tk.Label(root,text="", fg="red")
error.pack()

# Chức năng 3: Hiển thị thời gian khi sử dụng K-mean với 2 dữ liệu
label_2 = tk.Label(root, text="Phân lớp", font=font.Font(size=12, weight="bold"))
label_2.pack()

KMean_classifer_button = tk.Button(root, text="Phân lớp bằng KMeans", command=time2ExecuteKMean)
KMean_classifer_button.pack(pady=5)

ratio_PCA_label = tk.Label(root, text="Nhập phần trăm thông tin muốn giữ lại sau khi dùng PCA")
ratio_PCA_label.pack()

ratio_PCA_input = tk.Entry(root, width=30)
ratio_PCA_input.insert(0, 90)
ratio_PCA_input.pack()

info_PCAfeature = tk.Label(root, text="")
info_PCAfeature.pack()

timeRun_PCA = tk.Label(root, text="")
timeRun_PCA.pack()

timeRun_KMean_feature = tk.Label(root, text="")
timeRun_KMean_feature.pack()
accuracy_Kmean_feature = tk.Label(root, text="")
accuracy_Kmean_feature.pack()

timeRun_KMean_PCAfeature = tk.Label(root, text="")
timeRun_KMean_PCAfeature.pack()
accuracy_Kmean_PCAfeature = tk.Label(root, text="")
accuracy_Kmean_PCAfeature.pack()
# Chức năng 4: Trực quan hóa dữ liệu bằng cách dùng PCA đưa về 2 chiều và vẽ đồ thị

root.mainloop()
