# **07. Support Vector Machine / Support Vector Regression (SVM/SVR)**

## 1. Giới thiệu

Support Vector Machine (SVM) là phương pháp học máy dựa trên nguyên lý tìm siêu phẳng phân tách tối ưu giữa các lớp dữ liệu. SVM có khả năng xử lý các bài toán phân loại tuyến tính và phi tuyến thông qua kỹ thuật Kernel.

Support Vector Regression (SVR) mở rộng ý tưởng của SVM sang bài toán hồi quy, bằng cách xây dựng hàm dự đoán sao cho sai lệch giữa giá trị dự đoán và giá trị thực tế nằm trong một khoảng sai số cho phép.

## 2. Nội dung

- Ý tưởng phân loại bằng siêu phẳng.
- Margin và Support Vectors.
- Hard Margin và Soft Margin.
- Hàm mất mát Hinge Loss.
- Bài toán tối ưu của SVM.
- Kỹ thuật Kernel và Kernel Trick.
- Các hàm Kernel phổ biến: Linear, Polynomial, RBF.
- Support Vector Regression (SVR).
- Tham số C, epsilon và gamma.
- Huấn luyện, lựa chọn tham số và đánh giá SVM/SVR.

## 3. Học liệu

| Thư mục | Nội dung |
|---|---|
| `notebooks/` | Notebook thực hành: [Xem trên GitHub](notebooks/svm_svr_minh_hoa.ipynb) · [Mở bằng Google Colab](https://colab.research.google.com/github/chuminhuneti/Machine-Learning/blob/main/07_SVM_SVR/notebooks/svm_svr_minh_hoa.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chuminhuneti/Machine-Learning/blob/main/07_SVM_SVR/notebooks/svm_svr_minh_hoa.ipynb) |
| `code/` | Mã nguồn Python: [Xem và tải mã nguồn](code/svm_demo.py) |
| `documents/` | Tài liệu học tập: [Đọc tài liệu PDF](documents/SVM.pdf) |
| `html/` | Bài giảng trực tuyến: [Xem minh họa Softmax Regression – Từng bước](https://chuminhuneti.github.io/Machine-Learning/07_SVM_SVR/html/svm_svr_minh_hoa.html) |
| `slides/` | Bài giảng trình chiếu: [Xem slide PDF](slides/) |


## 4. Công cụ sử dụng

- Python
- NumPy, Pandas, Matplotlib
- Scikit-learn

## 5. Mục tiêu học tập

Sau khi hoàn thành bài học, người học có thể giải thích nguyên lý hoạt động của SVM/SVR, hiểu vai trò của Margin, Support Vectors và Kernel, đồng thời xây dựng, lựa chọn tham số và đánh giá mô hình cho các bài toán phân loại và hồi quy.
