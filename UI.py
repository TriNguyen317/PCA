import tkinter as tk
from tkinter import filedialog
from PCA import *

directory_path = ""
def select_directory():
    global directory_path
    try:
        directory_path = filedialog.askdirectory(title="Giảm chiều dữ liệu trong trích xuất đặc trưng ảnh ")
        if directory_path:
            status_label.config(text=f"Hoàn thành! Đường dẫn đã chọn: {directory_path}")
            global data
            data = np.array(ReadImage(directory_path))
            shape_input.config(text=f"Số chiều đầu vào: ({data.shape[0]},{data.shape[1]},{data.shape[2]})")
    except Exception as e:
        status_label.config(text=f"Lỗi: {str(e)}")

def preprocess_data():
    try:
        processed_data = Preprocess(data)
        shape_feature.config(text=f"Số chiều đặc trung lấy được: ({processed_data.shape[0]},{processed_data.shape[1]})")

        print(processed_data.shape)
    except Exception as e:
        if str(e)=="'NoneType' object has no attribute 'shape'":
            status_label.config(text=f"Lỗi khi xử lý dữ liệu: Chưa chọn đường dẫn")
        else:
            status_label.config(text=f"Lỗi khi xử lý dữ liệu: {str(e)}")
        
def Histogram_data():
    try:
        global histogram_data
        histogram_data = []
        for image in data:
            histogram_data.append(Histogram(image))
            
        histogram_data = np.array(histogram_data,dtype=np.uint8)
        shape_feature.config(text=f"Số chiều đặc trung lấy được: ({histogram_data.shape[0]},{histogram_data.shape[1]})")

        print(histogram_data.shape)
    except Exception as e:
        if str(e)=="'NoneType' object has no attribute 'shape'":
            status_label.config(text=f"Lỗi khi xử lý dữ liệu: Chưa chọn đường dẫn")
        else:
            status_label.config(text=f"Lỗi khi xử lý dữ liệu: {str(e)}")



root = tk.Tk()

# Hiện hướng dẫn sử dụng
title_label = tk.Label(root, text="Ứng dụng sử dụng PCA để giảm chiều dữ liệu trong việc lấy đặc trưng ảnh")
note_label = tk.Label(root, text="Ứng dụng lấy đặc trưng của ảnh bằng cách flatten các pixel có trong ảnh và sử dụng các đặc trưng đã lấy để phân lớp ảnh bằng thuật toán K-means")
user_manual = tk.Label(root, text="Hướng dẫn sử dụng")
user_manual_1 = tk.Label(root, text="1. Chọn đường dẫn chứa tập ảnh có cùng kích thước đầu vào.")
user_manual_2 = tk.Label(root, text="2. Chọn phương pháp lấy đặc trưng ảnh.")
user_manual_3 = tk.Label(root, text="2. Chọn bắt đầu dữ liệu sẽ được phân lớp bằng K-means. Kết quả sẽ được so sánh trên cả 2 tập dữ liệu trước và sau PCA")


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
select_button = tk.Button(root, text="Chọn Thư Mục", command=select_directory)
select_button.pack(pady=20)
status_label = tk.Label(root, text="")
status_label.pack()
shape_input = tk.Label(root, text="")
shape_input.pack()

# Chức năng 2: Lấy đặc trưng ảnh
# Tạo button để chọn cách lấy vector đặc trưng
label_1 = tk.Label(root, text="Lấy đặc trưng ảnh")
label_1.pack()
Flatten_button = tk.Button(root, text="Flatten pixel", command=preprocess_data)
Flatten_button.pack(pady=5)

Histogram_button = tk.Button(root, text="Histogram pixel", command=Histogram_data)
Histogram_button.pack(pady=5)

shape_feature = tk.Label(root, text="")
shape_feature.pack()

# Chức năng 3: Hiển thị thời gian khi sử dụng K-mean với 2 dữ liệu

# Chức năng 4: Trực quan hóa dữ liệu bằng cách dùng PCA đưa về 2 chiều và vẽ đồ thị

root.mainloop()
