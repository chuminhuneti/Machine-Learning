# 04. Logistic Regression – Hồi quy Logistic

## 1. Giới thiệu

Logistic Regression là mô hình học máy có giám sát được sử dụng chủ yếu cho các bài toán phân loại. Mặc dù có tên gọi là hồi quy, mô hình này thực hiện dự đoán xác suất một mẫu dữ liệu thuộc về một lớp.

Khác với Perceptron sử dụng hàm kích hoạt Step Function, Logistic Regression sử dụng hàm Sigmoid để ánh xạ đầu ra của mô hình về khoảng (0, 1), từ đó biểu diễn xác suất dự đoán.

Chương này tập trung vào mô hình, hàm mất mát và các thuật toán tối ưu được sử dụng trong quá trình huấn luyện Logistic Regression.

## 2. Mục tiêu học tập

Sau khi hoàn thành chương này, sinh viên có thể:

- Hiểu cách xây dựng mô hình Logistic Regression từ dữ liệu.
- Xác định và tính toán hàm mất mát của mô hình.
- Hiểu cách tính gradient và ý nghĩa của gradient trong huấn luyện.
- Trình bày và thực hiện thuật toán Gradient Descent.
- Sử dụng Python để huấn luyện và đánh giá mô hình Logistic Regression.
...

## 3. Nội dung


## 4. Học liệu

| Thư mục | Nội dung |
|---|---|
| `notebooks/` | Notebook thực hành: [Xem trên GitHub](notebooks/perceptron_minh_hoa.ipynb) · [Mở bằng Google Colab](https://colab.research.google.com/github/chuminhuneti/Machine-Learning/blob/main/04_Logistic_Regression/Logistic_Regression_từng_bước.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chuminhuneti/Machine-Learning/blob/main/04_Logistic_Regression/Logistic_Regression_từng_bước.ipynb) |
| `code/` | Mã nguồn Python: [Xem và tải mã nguồn](code/Logistic_Minh_Hoa.py) |
| `documents/` | Tài liệu học tập: [Đọc tài liệu PDF](documents/Logistic_Regression.pdf) |
| `html/` | Bài giảng trực tuyến: [Xem minh họa Logistic Regression – Từng bước](https://chuminhuneti.github.io/Machine-Learning/04_Logistic_Regression/html/Logistic_Regression_từng_bước.html) |
| `slides/` | Bài giảng trình chiếu: [Xem slide PDF](slides/Logistic_Regression_Slides.pdf) |

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
