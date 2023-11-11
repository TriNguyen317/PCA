import tkinter as tk
from tkinter import filedialog
from PCA import *

directory_path = ""
data = None
def select_directory():
    global directory_path
    try:
        directory_path = filedialog.askdirectory(title="Giảm chiều dữ liệu trong trích xuất đặc trưng ảnh ")
        if directory_path:
            status_label.config(text=f"Hoàn thành! Đường dẫn đã chọn: {directory_path}")
            global data
            data = np.array(ReadImage(directory_path))
            print(data.shape)
    except Exception as e:
        status_label.config(text=f"Lỗi: {str(e)}")

def preprocess_data():
    try:
        processed_data = Preprocess(data)
        # Thực hiện các công việc cần thiết với dữ liệu đã được xử lý
        # Ví dụ: In kích thước của dữ liệu sau khi xử lý
        print(processed_data.shape)
    except Exception as e:
        status_label.config(text=f"Lỗi khi xử lý dữ liệu: {str(e)}")




root = tk.Tk()
title_label = tk.Label(root, text="Ứng dụng sử dụng PCA để giảm chiều dữ liệu trong việc lấy đặc trưng ảnh")
note_label = tk.Label(root, text="Ứng dụng lấy đặc trưng của ảnh bằng cách flatten các pixel có trong ảnh và sử dụng các đặc trưng đã lấy để phân lớp ảnh bằng thuật toán K-means")
user_manual = tk.Label(root, text="Hướng dẫn sử dụng")
user_manual_1 = tk.Label(root, text="1. Chọn đường dẫn chứa tập ảnh có cùng kích thước đầu vào.")
user_manual_2 = tk.Label(root, text="1. Chọn phương pháp lấy đặc trưng ảnh.")

blank = tk.Label(root, text="")
status_label = tk.Label(root, text="")
title_label.pack()
blank.pack()
note_label.pack()

user_manual.pack()

root.title("Chọn Đường Dẫn")

# Tạo button chọn đường dẫn
select_button = tk.Button(root, text="Chọn Thư Mục", command=select_directory)
select_button.pack(pady=20)

# Tạo button để chọn cách lấy vector đặc trưng
label_1 = tk.Label(root, text="Lấy đặc trưng ảnh")
Histogram_button = tk.Button(root, text="Duỗi thẳng pixel", command=preprocess_data)
Histogram_button.pack(pady=20)

# Thông báo

root.mainloop()
