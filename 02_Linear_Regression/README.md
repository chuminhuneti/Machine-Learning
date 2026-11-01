# 02. Linear Regression – Hồi quy tuyến tính

## 1. Giới thiệu

Hồi quy tuyến tính (Linear Regression) là một mô hình học máy có giám sát, được sử dụng để dự đoán giá trị của biến đầu ra liên tục dựa trên các biến đầu vào.

Trong chương này, chúng ta tập trung vào quá trình **huấn luyện mô hình hồi quy tuyến tính**, với ba thành phần cốt lõi:

- **Mô hình (Model):** Xây dựng hàm dự đoán và xác định các tham số cần học.
- **Hàm mất mát (Loss Function):** Đo lường mức độ sai lệch giữa giá trị dự đoán và giá trị thực tế.
- **Thuật toán tối ưu (Optimization Algorithm):** Điều chỉnh các tham số để tìm mô hình có hàm mất mát nhỏ.

Chương học kết hợp lý thuyết, ví dụ tính toán, mã nguồn Python và thực hành trên Jupyter Notebook.

## 2. Mục tiêu học tập

Sau khi hoàn thành chương này, sinh viên có thể:

- Hiểu cách xây dựng mô hình hồi quy tuyến tính từ dữ liệu.
- Xác định và tính toán hàm mất mát của mô hình.
- Hiểu cách tính gradient và ý nghĩa của gradient trong huấn luyện.
- Trình bày và thực hiện thuật toán Gradient Descent.
- Sử dụng Python để huấn luyện và đánh giá mô hình hồi quy tuyến tính.
- Hiểu vai trò của các phương pháp điều chuẩn trong việc kiểm soát độ phức tạp của mô hình.

## 3. Nội dung chương học


## 4. Tài liệu học tập


### 4.1. Notebook thực hành

- [Xem Notebook trên GitHub](notebooks/Linear_Regression.ipynb)
- [Mở Notebook bằng Google Colab](https://colab.research.google.com/github/chuminhuneti/Machine-Learning/blob/main/02_Linear_Regression/notebooks/Linear_Regression.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chuminhuneti/Machine-Learning/blob/main/02_Linear_Regression/notebooks/Linear_Regression.ipynb)

### .42. Bài giảng HTML

- [Xem bài giảng Linear Regression – Từng bước](https://chuminhuneti.github.io/Machine-Learning/02_Linear_Regression/html/Linear_Regression_tung_buoc.html)
- [Danh sách tài liệu HTML](html/)

### 4.3. Mã nguồn Python

- [Xem và tải mã nguồn Python](code/)

### 4.4. Tài liệu học tập (Word và PDF)

- [Danh sách tài liệu](documents/)

### 4.5. Bài giảng trình chiếu

- [Danh sách bài giảng PowerPoint](slides/)



## 5. Công cụ thực hành

Sinh viên có thể sử dụng các công cụ sau:

- Python.
- Jupyter Notebook hoặc Google Colab.
- NumPy, Pandas.
- Matplotlib.
- Scikit-learn.

## 6. Hướng dẫn học tập

1. Đọc tài liệu lý thuyết để nắm được mô hình, hàm mất mát và thuật toán tối ưu.
2. Xem các ví dụ minh họa và bài giảng trình chiếu.
3. Mở Notebook bằng Google Colab để chạy và thực hành.
4. Thay đổi dữ liệu, tham số và thuật toán để quan sát ảnh hưởng đến quá trình huấn luyện.
5. Sử dụng mã nguồn Python để tự xây dựng và thử nghiệm mô hình.

**Lưu ý:** Khi thực hành trên Google Colab, sinh viên nên tạo một bản sao Notebook vào Google Drive trước khi chỉnh sửa để tránh làm thay đổi bản học liệu gốc trên GitHub.
