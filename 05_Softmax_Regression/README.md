# 05. Softmax Regression – Hồi quy Softmax

## 1. Giới thiệu

Softmax Regression là mô hình học máy có giám sát được sử dụng cho các bài toán phân loại đa lớp (Multiclass Classification).

Mô hình mở rộng Logistic Regression từ phân loại nhị phân sang trường hợp có nhiều lớp. Thay vì dự đoán xác suất thuộc một trong hai lớp, Softmax Regression dự đoán phân phối xác suất trên toàn bộ các lớp.

Chương này tập trung vào cách xây dựng mô hình, hàm mất mát và thuật toán tối ưu trong quá trình huấn luyện mô hình phân loại đa lớp.

## 2. Mục tiêu học tập

Sau khi hoàn thành chương này, sinh viên có thể:

- Hiểu cách xây dựng mô hình Softmax Regression từ dữ liệu.
- Xác định và tính toán hàm mất mát của mô hình.
- Hiểu cách tính gradient và ý nghĩa của gradient trong huấn luyện.
- Trình bày và thực hiện thuật toán Gradient Descent.
- Sử dụng Python để huấn luyện và đánh giá mô hình Softmax Regression.
...

## 3. Nội dung chương học


## 4. Tài liệu học tập


### 4.1. Notebook thực hành

- [Xem Notebook trên GitHub](notebooks/Softmax_Minh_Hoa.ipynb)
- [Mở Notebook bằng Google Colab](https://colab.research.google.com/github/chuminhuneti/Machine-Learning/blob/main/05_Softmax_Regression/notebooks/Softmax_Minh_Hoa.ipynb)

[![Open In Colab](https://colab.research.google.com/github/chuminhuneti/Machine-Learning/blob/main/05_Softmax_Regression/notebooks/Softmax_Minh_Hoa.ipynb)

### 4.2. Minh họa HTML

- [Xem minh họa Softmax Regression – Từng bước](https://chuminhuneti.github.io/Machine-Learning/05_Softmax_Regression/html/Softmax_Regression_ minh_họa.html)

### 4.3. Mã nguồn Python

- [Xem và tải mã nguồn Python](code/Softmax_Minh_Hoa.py)

### 4.4. Tài liệu học tập (Word và PDF)

- [Đọc tài liệu PDF](documents/Softmax_Regression.pdf)

### 4.5. Bài giảng trình chiếu

- [Xem slide PDF](slides/Softmax_Regression_Slides.pdf)


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

**Lưu ý:** Khi thực hành trên Google Colab, sinh viên nên chọn File → Save a copy in Drive để tạo bản Notebook riêng, thuận tiện cho việc chỉnh sửa và lưu trữ kết quả học tập. Notebook gốc trên GitHub được giữ nguyên.
