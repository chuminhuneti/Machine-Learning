#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
svm_demo.py - Minh họa toàn bộ nội dung bài giảng SVM / SVR (mục 1 đến mục 8)
=============================================================================

Mỗi hàm `phan_K_...` tái hiện các ví dụ ở mục K của bài giảng, in ra số liệu để
đối chiếu với phép tính tay trong tài liệu. Các thuật toán chính được CÀI ĐẶT TỪ ĐẦU
bằng NumPy (LinearSVM, KernelSVM/SMO, LinearSVR, KernelSVR, Platt scaling, đặc trưng
Fourier ngẫu nhiên) và được đối chiếu với scikit-learn.

Cách chạy
---------
    python svm_demo.py                 # chạy tất cả các phần 1..8
    python svm_demo.py --part 3 5      # chỉ chạy mục 3 và mục 5
    python svm_demo.py --plots         # chạy tất cả và lưu hình vào ./hinh_svm
    python svm_demo.py --part 6 --plots --out hinh_svr

Yêu cầu: numpy, scipy, scikit-learn (và matplotlib nếu dùng --plots).

Quy ước ký hiệu (giống tài liệu)
--------------------------------
    X : ma trận đặc trưng (n x d), mỗi hàng là một vector đặc trưng x_i
    y : vector nhãn, y_i thuộc {-1, +1} (phân loại) hoặc số thực (hồi quy)
    w : vector trọng số (d,), b : hệ số chệch (vô hướng)
    m = y * (X @ w + b) : margin;  C : hệ số phạt;  eps : độ rộng ống của SVR
    alpha : hệ số đối ngẫu;  K : ma trận Gram;  Q = diag(y) K diag(y)
"""
import argparse
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")          # tắt cảnh báo hội tụ của sklearn cho gọn màn hình


# =============================================================================
# 0. DỮ LIỆU DÙNG CHUNG
# =============================================================================
def tieu_de(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def du_lieu_6_sinh_vien():
    """6 sinh viên: x1 = số giờ tự học, x2 = số buổi vắng; y = +1 (đậu), -1 (rớt)."""
    X = np.array([[2, 4], [3, 3], [4, 5], [6, 1], [8, 2], [9, 0]], dtype=float)
    y = np.array([-1, -1, -1, 1, 1, 1], dtype=float)
    return X, y


def du_lieu_7_sinh_vien():
    """Thêm sinh viên 7 (7, 1) bị rớt: nằm lẫn giữa các sinh viên đậu -> KHÔNG tách được."""
    X, y = du_lieu_6_sinh_vien()
    return np.vstack([X, [7, 1]]), np.r_[y, -1.0]


def du_lieu_400_sinh_vien(seed=42):
    """400 sinh viên mô phỏng (giống bài Perceptron / Logistic Regression)."""
    rng = np.random.default_rng(seed)
    n_lop = 200
    dau = np.c_[rng.normal(8, 2.2, n_lop), rng.normal(1.5, 1.2, n_lop)]
    rot = np.c_[rng.normal(4, 2.2, n_lop), rng.normal(3.5, 1.5, n_lop)]
    X = np.clip(np.vstack([dau, rot]), 0, None)
    y = np.r_[np.ones(n_lop), -np.ones(n_lop)]
    return X, y


def chia_train_test(X, y, ti_le_test=0.3, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = int(len(y) * ti_le_test)
    return X[idx[n_test:]], X[idx[:n_test]], y[idx[n_test:]], y[idx[:n_test]]


def chuan_hoa(X_train, *others):
    """Chuẩn hóa theo trung bình / độ lệch chuẩn của TẬP HUẤN LUYỆN (tránh rò rỉ dữ liệu)."""
    mu, sd = X_train.mean(0), X_train.std(0)
    return [(X_train - mu) / sd] + [(Z - mu) / sd for Z in others] + [mu, sd]


def du_lieu_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([-1, 1, 1, -1], dtype=float)
    return X, y


def du_lieu_sin_co_ngoai_lai(seed=3):
    """y = sin(x) + nhiễu + 3 ngoại lai (dùng cho SVR)."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 6, 60))
    y = np.sin(x) + rng.normal(0, 0.12, 60)
    y[[10, 30, 45]] += np.array([1.6, -1.8, 1.5])
    return x[:, None], y


# =============================================================================
# 1. LỀ, KHOẢNG CÁCH CÓ DẤU
# =============================================================================
def margin_ham(X, y, w, b):
    """Margin hàm m_i = y_i (x_i w + b). Phụ thuộc thang đo của (w, b)."""
    return y * (X @ w + b)


def khoang_cach_co_dau(X, y, w, b):
    """gamma_i = m_i / ||w||: khoảng cách hình học có dấu, KHÔNG phụ thuộc thang đo."""
    return margin_ham(X, y, w, b) / np.linalg.norm(w)


def le_cua(X, y, w, b):
    """Lề của bộ phân loại = khoảng cách đến mẫu gần nhất = min_i gamma_i."""
    return khoang_cach_co_dau(X, y, w, b).min()


def phan_1():
    tieu_de("MỤC 1. Từ Perceptron / Logistic Regression đến bài toán lề cực đại")
    X, y = du_lieu_6_sinh_vien()
    w, b = np.array([1.0, -1.0]), -2.5
    m = margin_ham(X, y, w, b)
    g = khoang_cach_co_dau(X, y, w, b)
    print("w = (1, -1), b = -2.5  ->  ||w|| =", round(np.linalg.norm(w), 4))
    print("margin hàm m_i      :", m)
    print("khoảng cách có dấu  :", g.round(3), " -> lề =", round(g.min(), 3))

    # Tính bất biến theo thang đo: nhân (w, b) với c > 0
    c = 10
    print(f"\nNhân (w, b) với c = {c}: margin hàm x{c} = {margin_ham(X, y, c*w, c*b)}")
    print("   nhưng khoảng cách có dấu vẫn như cũ:", khoang_cach_co_dau(X, y, c * w, c * b).round(3))

    # Bốn đường của Perceptron (mục 1.1) so với đường lề cực đại (mục 1.3)
    print("\nLề của các đường biên đúng (mục 1.1 và 1.3):")
    duong = {
        "A (Perceptron)": (-1.0, np.array([4.0, -8.0])),
        "B (Perceptron)": (-1.0, np.array([6.0, -11.0])),
        "C (Perceptron)": (-1.0, np.array([5.0, -9.0])),
        "D (Perceptron)": (0.0, np.array([2.0, -4.0])),
        "Lề cực đại": (-19 / 13, np.array([6 / 13, -4 / 13])),
    }
    x_moi = np.array([7.0, 3.6])
    for ten, (b_, w_) in duong.items():
        s = x_moi @ w_ + b_
        print(f"  {ten:16s} lề = {le_cua(X, y, w_, b_):.3f} | sinh viên mới (7; 3,6): "
              f"{'đậu' if s > 0 else 'rớt'}")

    # Mệnh đề ổn định trước nhiễu: dịch chuyển < lề thì không đổi nhãn (Cauchy-Schwarz)
    e = 1.5 * np.array([2, -4]) / np.sqrt(20)        # dịch chuyển dài 1,5 theo hướng pháp tuyến của D
    xn = X[1] + e
    print("\nDịch sinh viên 2 một đoạn dài 1,5:")
    for ten in ["D (Perceptron)", "Lề cực đại"]:
        b_, w_ = duong[ten]
        print(f"  {ten:16s} điểm số mới = {xn @ w_ + b_: .3f}  (nhãn đúng là rớt, cần < 0)")


# =============================================================================
# 2. SVM TUYẾN TÍNH LỀ CỨNG
# =============================================================================
def hard_margin_tu_cap_gan_nhat(X, y):
    """Giải tay lề cứng bằng cách ĐOÁN vector hỗ trợ = cặp mẫu khác lớp gần nhau nhất.

    Nếu x_p (dương), x_q (âm) là vector hỗ trợ thì  (x_p - x_q) w = 2.  Theo Cauchy-Schwarz,
    ||w|| nhỏ nhất khi w cùng phương với (x_p - x_q):   w = 2 (x_p - x_q)^T / ||x_p - x_q||^2.
    Sau đó b = y_p - x_p w. Phải KIỂM TRA mọi ràng buộc còn lại m_i >= 1.
    """
    P, N = np.where(y == 1)[0], np.where(y == -1)[0]
    d2 = ((X[P][:, None, :] - X[N][None, :, :]) ** 2).sum(-1)      # bình phương khoảng cách từng cặp
    ip, iq = np.unravel_index(d2.argmin(), d2.shape)
    p, q = P[ip], N[iq]
    diff = X[p] - X[q]
    w = 2 * diff / (diff @ diff)
    b = y[p] - X[p] @ w
    return w, b, (p, q)


def phan_2():
    tieu_de("MỤC 2. SVM tuyến tính lề cứng")
    from sklearn.svm import SVC
    X, y = du_lieu_6_sinh_vien()
    w, b, (p, q) = hard_margin_tu_cap_gan_nhat(X, y)
    print(f"Cặp gần nhất: sinh viên {q + 1} (rớt) và sinh viên {p + 1} (đậu)")
    print("w =", w.round(4), "(= 6/13, -4/13)   b =", round(b, 4), "(= -19/13)")
    m = margin_ham(X, y, w, b)
    print("margin m_i =", m.round(3), "-> mọi m_i >= 1:", bool((m >= 1 - 1e-9).all()))
    print("vector hỗ trợ (m_i = 1): sinh viên", (np.where(np.isclose(m, 1))[0] + 1).tolist())
    print("lề gamma = 1/||w|| =", round(1 / np.linalg.norm(w), 4), "(= sqrt(13)/2 = 1,8028)",
          "| độ rộng dải lề 2/||w|| =", round(2 / np.linalg.norm(w), 4))

    # Dạng ma trận: min 1/2 u^T P u  với  G u >= 1,  u = [w; b]
    Z = np.diag(y) @ X                                   # hàng i = y_i x_i
    G = np.c_[Z, y]                                      # n x (d+1)
    u = np.r_[w, b]
    print("\nDạng ma trận: G u =", (G @ u).round(3), "(phải >= 1)")

    ref = SVC(kernel="linear", C=1e8, tol=1e-10).fit(X, y)
    print("sklearn SVC(C rất lớn): w =", ref.coef_[0].round(4), " b =", ref.intercept_.round(4),
          " vector hỗ trợ:", (ref.support_ + 1).tolist())

    # Thử bỏ các mẫu: chỉ vector hỗ trợ quyết định đường biên
    for ten, idx in [("chỉ giữ SV 2 và 4", [1, 3]), ("bỏ sinh viên 2", [0, 2, 3, 4, 5])]:
        s = SVC(kernel="linear", C=1e8, tol=1e-10).fit(X[idx], y[idx])
        print(f"  {ten:20s}: w = {s.coef_[0].round(4)}, lề = {1 / np.linalg.norm(s.coef_[0]):.3f}")


# =============================================================================
# 3. SVM LỀ MỀM, HÀM MẤT MÁT HINGE, GRADIENT DƯỚI ĐẠO HÀM
# =============================================================================
class LinearSVM:
    """SVM tuyến tính lề mềm, huấn luyện trên bài toán GỐC không ràng buộc:

        J(w, b) = 1/2 ||w||^2 + C * sum_i max(0, 1 - y_i (x_i w + b))

    method="batch": mỗi bước dùng toàn bộ dữ liệu (gradient dưới đạo hàm).
    method="sgd"  : kiểu Pegasos - mỗi bước chỉ dùng 1 mẫu ngẫu nhiên.
    """

    def __init__(self, C=1.0, n_iter=1000, method="batch", seed=0):
        self.C, self.n_iter, self.method, self.seed = C, n_iter, method, seed

    def objective(self, X, y):
        m = y * (X @ self.w + self.b)
        return 0.5 * self.w @ self.w + self.C * np.maximum(0, 1 - m).sum()

    def fit(self, X, y):
        n, d = X.shape
        # Dạng chuẩn hóa J/(n C) = (lam/2)||w||^2 + trung bình hinge, với lam = 1/(n C).
        # Hàm này lồi chặt với hệ số lam nên bước 1/(lam t) cho đảm bảo hội tụ.
        lam = 1.0 / (n * self.C)
        self.w, self.b, self.history = np.zeros(d), 0.0, []
        rng = np.random.default_rng(self.seed)
        t = 0
        for _ in range(self.n_iter):
            thu_tu = [None] if self.method == "batch" else rng.permutation(n)
            for i in thu_tu:
                t += 1
                eta = 1.0 / (lam * t)                    # tốc độ học GIẢM DẦN
                if i is None:                            # ---- batch
                    # a_i = 1 nếu mẫu i vi phạm lề (m_i < 1), ngược lại 0
                    a = (y * (X @ self.w + self.b) < 1).astype(float)
                    # dưới đạo hàm của f:  lam*w - (1/n) X^T (y ⊙ a)   và   -(1/n) 1^T (y ⊙ a)
                    g_w = lam * self.w - X.T @ (y * a) / n
                    g_b = -np.sum(y * a) / n
                else:                                    # ---- SGD kiểu Pegasos
                    a = float(y[i] * (X[i] @ self.w + self.b) < 1)
                    g_w = lam * self.w - a * y[i] * X[i]
                    g_b = -a * y[i]
                self.w = self.w - eta * g_w
                self.b = self.b - eta * g_b
            self.history.append(self.objective(X, y))
        return self

    def decision_function(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)


def mot_buoc_batch(X, y, w, b, C, t):
    """MỘT bước Gradient Descent dưới đạo hàm trên J (dạng ở mục 3.4 / 3.6), tốc độ học 1/t."""
    m = y * (X @ w + b)
    a = (m < 1).astype(float)                            # vector chỉ báo "vi phạm lề"
    g_w = w - C * X.T @ (y * a)
    g_b = -C * np.sum(y * a)
    return w - g_w / t, b - g_b / t, m, a


def phan_3():
    tieu_de("MỤC 3. SVM lề mềm và hàm mất mát hinge")
    from sklearn.svm import SVC
    X, y = du_lieu_7_sinh_vien()

    print("Ví dụ tính tay một vòng (7 sinh viên, C = 0,2, w = 0, b = 0, bước 1/t):")
    w, b, C = np.zeros(2), 0.0, 0.2
    for t in range(1, 6):
        J = 0.5 * w @ w + C * np.maximum(0, 1 - y * (X @ w + b)).sum()
        w_moi, b_moi, m, a = mot_buoc_batch(X, y, w, b, C, t)
        print(f"  t={t}: J={J:6.3f} | số mẫu vi phạm={int(a.sum())} -> w={w_moi.round(3)}, b={b_moi:.3f}")
        w, b = w_moi, b_moi

    print("\nẢnh hưởng của C (nghiệm chính xác bằng SVC) trên 7 sinh viên:")
    print("      C |     w        |    b   |  lề   | Σξ    | #ξ>0 | #sai")
    for C in [0.01, 0.1, 1, 10]:
        s = SVC(kernel="linear", C=C, tol=1e-10).fit(X, y)
        w_, b_ = s.coef_[0], s.intercept_[0]
        m = y * (X @ w_ + b_)
        xi = np.maximum(0, 1 - m)
        print(f"{C:7.2f} | {np.round(w_, 3)} | {b_:6.3f} | {1 / np.linalg.norm(w_):5.2f} | "
              f"{xi.sum():5.2f} | {(xi > 1e-6).sum():4d} | {(m < 0).sum():3d}")

    # Huấn luyện tự cài đặt, trên dữ liệu ĐÃ CHUẨN HÓA (quan trọng cho tốc độ hội tụ)
    Z, mu, sd = chuan_hoa(X)
    ref = SVC(kernel="linear", C=1, tol=1e-12).fit(Z, y)
    for method, n_iter in [("batch", 1000), ("sgd", 1000)]:
        m_ = LinearSVM(C=1, n_iter=n_iter, method=method).fit(Z, y)
        print(f"\nLinearSVM ({method}): w = {m_.w.round(4)}, b = {m_.b:.4f}, J = {m_.history[-1]:.4f}")
    print(f"sklearn SVC        : w = {ref.coef_[0].round(4)}, b = {ref.intercept_[0]:.4f}")


# =============================================================================
# 4. ĐỐI NGẪU, KKT, VECTOR HỖ TRỢ
# =============================================================================
def alpha_tu_sklearn(svm, n):
    """sklearn lưu dual_coef_ = alpha_i * y_i chỉ cho các vector hỗ trợ -> dựng lại alpha đầy đủ."""
    alpha = np.zeros(n)
    alpha[svm.support_] = np.abs(svm.dual_coef_[0])
    return alpha


def perceptron_dang_doi_ngau(X, y, thu_tu, eta=1.0):
    """Perceptron: n_i = số lần mẫu i gây cập nhật. Khi đó w = sum n_i eta y_i x_i, b = sum n_i eta y_i."""
    w, b, dem = np.zeros(X.shape[1]), 0.0, np.zeros(len(y))
    while True:
        loi = 0
        for i in thu_tu:
            if y[i] * (X[i] @ w + b) <= 0:
                w, b, dem[i], loi = w + eta * y[i] * X[i], b + eta * y[i], dem[i] + 1, loi + 1
        if loi == 0:
            return w, b, dem


def phan_4():
    tieu_de("MỤC 4. Đối ngẫu và vector hỗ trợ")
    from sklearn.svm import SVC
    X, y = du_lieu_7_sinh_vien()
    n = len(y)
    svm = SVC(kernel="linear", C=1.0, tol=1e-12).fit(X, y)
    alpha = alpha_tu_sklearn(svm, n)
    w, b = svm.coef_[0], svm.intercept_[0]
    m = y * (X @ w + b)

    print("alpha =", alpha.round(4), "(lý thuyết: 0, 7/26, 0, 1, 7/26, 0, 1)")
    loai = ["α=0 (ngoài lề)" if a < 1e-8 else ("α=C (bị chặn)" if a > 1 - 1e-6 else "0<α<C (trên biên lề)")
            for a in alpha]
    for i in range(n):
        print(f"  sinh viên {i + 1}: m = {m[i]:6.3f}  -> {loai[i]}")

    # Kiểm tra các công thức của đối ngẫu
    print("\nw = Σ α_i y_i x_i =", (X.T @ (alpha * y)).round(4), "  | w của sklearn =", w.round(4))
    print("Σ α_i y_i =", round(alpha @ y, 8), "(cân bằng lực)")
    Q = np.outer(y, y) * (X @ X.T)                       # Q_ij = y_i y_j x_i x_j^T
    D = alpha.sum() - 0.5 * alpha @ Q @ alpha            # giá trị đối ngẫu
    J = 0.5 * w @ w + np.maximum(0, 1 - m).sum()          # giá trị gốc (C = 1)
    print(f"giá trị đối ngẫu = {D:.4f} | giá trị gốc = {J:.4f} (đối ngẫu mạnh: 127/52 = {127 / 52:.4f})")
    print("m = Q α + b y:", (Q @ alpha + b * y).round(3))

    # Lề cứng trên 6 sinh viên: ngưỡng C* = 2/13
    X6, y6 = du_lieu_6_sinh_vien()
    print("\n6 sinh viên - khi nào lề mềm trùng lề cứng? (C* = 2/13 ≈ 0,1538)")
    for C in [0.05, 0.1, 0.15, 0.16, 1.0]:
        s = SVC(kernel="linear", C=C, tol=1e-12).fit(X6, y6)
        print(f"  C={C:5.2f}: w = {s.coef_[0].round(4)}, α>0: {np.abs(s.dual_coef_[0]).round(4)}")

    # Perceptron dạng đối ngẫu: alpha là SỐ LẦN SAI, khác hẳn alpha của SVM
    w_p, b_p, dem = perceptron_dang_doi_ngau(X6, y6, [0, 1, 2, 3, 4, 5])
    print("\nPerceptron đối ngẫu (thứ tự 1..6): số lần cập nhật n =", dem.astype(int).tolist(),
          "-> w =", w_p, " b =", b_p)


# =============================================================================
# 5. KERNEL
# =============================================================================
def linear_kernel(A, B):
    return A @ B.T


def poly_kernel(A, B, gamma=1.0, coef0=1.0, degree=2):
    return (gamma * (A @ B.T) + coef0) ** degree


def rbf_kernel(A, B, gamma=1.0):
    """Ma trận Gram RBF giữa hai tập điểm KHÔNG DÙNG VÒNG LẶP.

    Dùng khai triển ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b:
      - (A**2).sum(1)[:, None]  : cột các ||a_i||^2, cỡ n1 x 1
      - (B**2).sum(1)[None, :]  : hàng các ||b_j||^2, cỡ 1 x n2
      - A @ B.T                 : mọi tích vô hướng a_i·b_j cùng lúc
    NumPy tự "phát sóng" (broadcast) ba số hạng này thành ma trận n1 x n2.
    np.maximum(., 0) chặn sai số làm tròn khiến bình phương khoảng cách hơi âm.
    """
    sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return np.exp(-gamma * np.maximum(sq, 0))


class KernelSVM:
    """SVM lề mềm có kernel, giải bài toán ĐỐI NGẪU bằng SMO đơn giản hóa.

        max  1^T a - 1/2 a^T Q a     với   y^T a = 0,   0 <= a <= C

    Ý tưởng SMO: ràng buộc y^T a = 0 nên không thể đổi MỘT alpha riêng lẻ; ta đổi CẶP (i, j)
        a_i <- a_i + y_i t,    a_j <- a_j - y_j t
    (giữ nguyên tổng y^T a). Khi đó hàm đối ngẫu chỉ còn là hàm bậc hai theo t -> có nghiệm đóng.
    """

    def __init__(self, kernel, C=1.0, tol=1e-3, max_passes=20, seed=0):
        self.kernel, self.C, self.tol = kernel, C, tol
        self.max_passes, self.seed = max_passes, seed

    def fit(self, X, y):
        n = len(y)
        K = self.kernel(X, X)                            # ma trận Gram n x n, tính MỘT lần
        alpha, b = np.zeros(n), 0.0
        rng = np.random.default_rng(self.seed)
        f = lambda i: (alpha * y) @ K[:, i] + b          # hàm quyết định tại mẫu huấn luyện i
        passes = 0
        while passes < self.max_passes:                  # dừng khi một số lượt liên tiếp không đổi gì
            changed = 0
            for i in range(n):
                E_i = f(i) - y[i]                        # sai số E_i = f(x_i) - y_i
                # Mẫu i VI PHẠM điều kiện KKT? (mục 4.3)
                #   y_i f_i < 1 mà alpha_i < C   hoặc   y_i f_i > 1 mà alpha_i > 0
                # (viết qua E_i: y_i f_i - 1 = y_i E_i)
                if (y[i] * E_i < -self.tol and alpha[i] < self.C) or \
                   (y[i] * E_i > self.tol and alpha[i] > 0):
                    j = rng.integers(n - 1)              # chọn j ngẫu nhiên khác i
                    j = j + 1 if j >= i else j
                    E_j = f(j) - y[j]
                    a_i, a_j = alpha[i], alpha[j]
                    # Cận L, H để cả hai alpha vẫn nằm trong [0, C] sau khi đổi (giữ y^T a = 0)
                    if y[i] != y[j]:
                        L, H = max(0, a_j - a_i), min(self.C, self.C + a_j - a_i)
                    else:
                        L, H = max(0, a_i + a_j - self.C), min(self.C, a_i + a_j)
                    kappa = K[i, i] + K[j, j] - 2 * K[i, j]   # độ cong của hàm đối ngẫu theo hướng đổi
                    if L == H or kappa <= 1e-12:
                        continue
                    # Nghiệm đóng của bài toán một biến, rồi CẮT về [L, H]
                    a_j_new = np.clip(a_j + y[j] * (E_i - E_j) / kappa, L, H)
                    if abs(a_j_new - a_j) < 1e-8:
                        continue
                    a_i_new = a_i + y[i] * y[j] * (a_j - a_j_new)   # suy ra alpha_i để giữ y^T a
                    # Cập nhật b sao cho mẫu i (hoặc j) thỏa KKT khớp: ưu tiên mẫu có 0 < alpha < C
                    b1 = b - E_i - y[i] * (a_i_new - a_i) * K[i, i] - y[j] * (a_j_new - a_j) * K[i, j]
                    b2 = b - E_j - y[i] * (a_i_new - a_i) * K[i, j] - y[j] * (a_j_new - a_j) * K[j, j]
                    if 0 < a_i_new < self.C:
                        b = b1
                    elif 0 < a_j_new < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2
                    alpha[i], alpha[j] = a_i_new, a_j_new
                    changed += 1
            passes = passes + 1 if changed == 0 else 0
        self.alpha, self.b = alpha, b
        sv = alpha > 1e-8
        self.X_sv, self.coef = X[sv], (alpha * y)[sv]    # chỉ giữ vector hỗ trợ (tính thưa)
        self.dual_value = alpha.sum() - 0.5 * (alpha * y) @ K @ (alpha * y)
        return self

    def decision_function(self, X):
        # f(x) = sum_i alpha_i y_i k(x_i, x) + b = K_moi (alpha ⊙ y) + b
        return self.kernel(X, self.X_sv) @ self.coef + self.b

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)


def giai_xor_bang_he_kkt():
    """XOR với kernel (x z^T + 1)^2: nếu cả 4 điểm là vector hỗ trợ (m_i = 1) thì
        Q α + b y = 1   và   y^T α = 0   <=>   [[Q, y], [y^T, 0]] [α; b] = [1; 0].
    Đây là hệ 5 phương trình tuyến tính 5 ẩn: giải thẳng bằng np.linalg.solve."""
    X, y = du_lieu_xor()
    K = poly_kernel(X, X, 1.0, 1.0, 2)
    Q = np.outer(y, y) * K
    n = len(y)
    A = np.block([[Q, y[:, None]], [y[None, :], np.zeros((1, 1))]])
    rhs = np.r_[np.ones(n), 0.0]
    sol = np.linalg.solve(A, rhs)
    return K, Q, sol[:n], sol[n]


def phan_5():
    tieu_de("MỤC 5. SVM phi tuyến - thủ thuật kernel")
    from sklearn.svm import SVC
    from sklearn.datasets import make_circles

    # (a) Đồng nhất thức kernel: (x z^T)^2 = phi(x) phi(z)^T với phi = [x1^2, sqrt2 x1 x2, x2^2]
    x, z = np.array([1.0, 2.0]), np.array([2.0, 0.0])
    phi = lambda v: np.array([1, np.sqrt(2) * v[0], np.sqrt(2) * v[1], v[0] ** 2, np.sqrt(2) * v[0] * v[1], v[1] ** 2])
    print("k(x,z) = (x z^T + 1)^2 =", (x @ z + 1) ** 2, "| phi(x)·phi(z) =", round(phi(x) @ phi(z), 6),
          "(kernel khỏi dựng 6 đặc trưng)")

    # (b) XOR: giải tay bằng hệ KKT
    K, Q, alpha, b = giai_xor_bang_he_kkt()
    print("\nXOR, kernel đa thức bậc 2: Gram K =\n", K)
    print("α =", alpha.round(4), "(lý thuyết 10/3, 8/3, 8/3, 2)   b =", round(b, 4))
    X, y = du_lieu_xor()
    f = lambda p: ((alpha * y) @ poly_kernel(X, np.atleast_2d(p), 1, 1, 2)).item() + b
    print("f(x) tại 4 điểm:", [round(float(f(p)), 3) for p in X], "| f(0,5; 0) =", round(float(f([0.5, 0.0])), 4), "(= -1/6)")

    # (c) SMO tự cài đặt so với sklearn
    m = KernelSVM(lambda A, B: poly_kernel(A, B, 1, 1, 2), C=1e3, max_passes=50).fit(X, y)
    print("\nSMO tự cài đặt trên XOR: α =", m.alpha.round(3), " b =", round(m.b, 3), " dự đoán:", m.predict(X))

    from sklearn.model_selection import train_test_split
    Xc, yc = make_circles(300, noise=0.15, factor=0.45, random_state=1)
    yc = 2.0 * yc - 1
    Xa, Xb, ya, yb = train_test_split(Xc, yc, test_size=0.3, random_state=0)   # trùng với tài liệu
    Za, Zb, _, _ = chuan_hoa(Xa, Xb)
    print("\nHai vòng tròn có nhiễu (210 huấn luyện / 90 kiểm tra):")
    for ten, ker, sk in [
        ("đa thức bậc 2", lambda A, B: poly_kernel(A, B, 1, 1, 2), SVC(kernel="poly", degree=2, gamma=1, coef0=1, C=1)),
        ("RBF γ=1", lambda A, B: rbf_kernel(A, B, 1.0), SVC(kernel="rbf", gamma=1, C=1)),
    ]:
        mine = KernelSVM(ker, C=1.0).fit(Za, ya)
        sk.fit(Za, ya)
        print(f"  {ten:14s} tự cài đặt: test {np.mean(mine.predict(Zb) == yb):.3f}, #SV {len(mine.coef)} | "
              f"sklearn: test {sk.score(Zb, yb):.3f}, #SV {sk.n_support_.sum()}")

    # (d) Vai trò của gamma: gamma lớn -> mô hình thuộc lòng dữ liệu huấn luyện
    print("\nẢnh hưởng của γ (C=1): huấn luyện / kiểm tra / số vector hỗ trợ")
    for g in [0.1, 1, 50]:
        s = SVC(kernel="rbf", gamma=g, C=1).fit(Za, ya)
        print(f"  γ={g:5.1f}: {s.score(Za, ya):.3f} / {s.score(Zb, yb):.3f} / {s.n_support_.sum()}")


# =============================================================================
# 6. SVR
# =============================================================================
class LinearSVR:
    """SVR tuyến tính huấn luyện trên bài toán gốc:
        J(w, b) = 1/2 ||w||^2 + C * sum_i max(0, |y_i - x_i w - b| - eps)
    Giống hệt LinearSVM, chỉ khác điều kiện vi phạm (|r_i| > eps) và hướng đẩy sign(r_i) x_i."""

    def __init__(self, C=1.0, eps=0.1, n_iter=5000):
        self.C, self.eps, self.n_iter = C, eps, n_iter

    def objective(self, X, y):
        r = y - X @ self.w - self.b
        return 0.5 * self.w @ self.w + self.C * np.maximum(0, np.abs(r) - self.eps).sum()

    def fit(self, X, y):
        self.w, self.b = np.zeros(X.shape[1]), 0.0
        for t in range(1, self.n_iter + 1):
            r = y - X @ self.w - self.b                  # phần dư
            a = (np.abs(r) > self.eps).astype(float)     # mẫu NGOÀI ống mới bị phạt
            s = np.sign(r) * a                           # s_i ∈ {-1, 0, +1}
            # r_i = y_i - x_i w - b  =>  d r_i / d w = -x_i^T  nên dưới đạo hàm có dấu trừ như dưới
            self.w -= (self.w - self.C * X.T @ s) / t
            self.b -= (-self.C * s.sum()) / t
        return self

    def predict(self, X):
        return X @ self.w + self.b


class KernelSVR:
    """SVR có kernel, giải bài toán đối ngẫu trên 2n biến g = [alpha; alpha*] bằng SLSQP:

        min 1/2 β^T K β + eps 1^T (α + α*) - y^T β,   β = α - α*
        với  1^T β = 0  và  0 <= α, α* <= C

    SLSQP là bộ giải tối ưu có ràng buộc tổng quát của SciPy (đủ cho vài trăm mẫu);
    thư viện thật dùng SMO. Ta cung cấp luôn gradient để bộ giải nhanh và chính xác.
    """

    def __init__(self, kernel, C=1.0, eps=0.1):
        self.kernel, self.C, self.eps = kernel, C, eps

    def fit(self, X, y):
        from scipy.optimize import minimize
        n = len(y)
        K = self.kernel(X, X)

        def obj(g):
            beta = g[:n] - g[n:]
            return 0.5 * beta @ K @ beta + self.eps * g.sum() - y @ beta

        def grad(g):                                     # đạo hàm theo alpha và theo alpha*
            Kb = K @ (g[:n] - g[n:])
            return np.r_[Kb + self.eps - y, -Kb + self.eps + y]

        rang_buoc = {"type": "eq", "fun": lambda g: g[:n].sum() - g[n:].sum(),      # 1^T β = 0
                     "jac": lambda g: np.r_[np.ones(n), -np.ones(n)]}
        res = minimize(obj, np.zeros(2 * n), jac=grad, bounds=[(0, self.C)] * (2 * n),
                       constraints=[rang_buoc], method="SLSQP", options={"maxiter": 500, "ftol": 1e-12})
        beta = res.x[:n] - res.x[n:]
        self.beta = np.where(np.abs(beta) < 1e-7, 0.0, beta)
        sv = self.beta != 0
        self.X_sv, self.coef = X[sv], self.beta[sv]
        # b từ vector hỗ trợ TỰ DO (0 < |β| < C): phần dư của chúng bằng đúng ±eps
        #   y_k - f0_k - b = eps * sign(β_k)  =>  b = y_k - f0_k - eps * sign(β_k)
        tu_do = (np.abs(self.beta) > 1e-7) & (np.abs(self.beta) < self.C - 1e-7)
        f0 = K @ self.beta
        bs = y[tu_do] - f0[tu_do] - self.eps * np.sign(self.beta[tu_do])
        self.b = bs.mean() if tu_do.any() else np.median(y - f0)
        return self

    def predict(self, X):
        return self.kernel(X, self.X_sv) @ self.coef + self.b


def phan_6():
    tieu_de("MỤC 6. Hồi quy vector hỗ trợ (SVR)")
    from sklearn.svm import SVR
    from sklearn.linear_model import LinearRegression

    # (a) Ngoại lai làm lệch hồi quy tuyến tính nhưng không làm lệch SVR
    x = np.array([1, 2, 3, 4, 5.0])
    y = np.array([1.0, 2.2, 2.9, 4.0, 5.0])
    print("Hệ số góc khi ngoại lai (x=6, y=y6) di chuyển:")
    print("   y6 |  hồi quy MSE (w, b)   |  SVR (C=1, eps=0,2) (w, b)")
    for y6 in [None, 3.0, 2.0, -2.0]:
        xx = x if y6 is None else np.r_[x, 6.0]
        yy = y if y6 is None else np.r_[y, y6]
        ols = LinearRegression().fit(xx[:, None], yy)
        svr = LinearSVR(C=1, eps=0.2, n_iter=20000).fit(xx[:, None], yy)
        print(f"  {'-' if y6 is None else y6:>4} | ({ols.coef_[0]:6.3f}, {ols.intercept_:6.3f}) | "
              f"({svr.w[0]:6.4f}, {svr.b:6.4f})")
    print("   (nghiệm tay với ngoại lai: w = 13/15 = 0,8667, b = 1/3 = 0,3333)")

    # (b) Kiểm tra KKT cho nghiệm tay
    xx, yy = np.r_[x, 6.0], np.r_[y, 2.0]
    w, b, eps = 13 / 15, 1 / 3, 0.2
    r = yy - (w * xx + b)
    print("\nPhần dư r =", r.round(3), "-> trong ống:", (np.abs(r) <= eps + 1e-9).astype(int).tolist())
    xi = np.maximum(0, np.abs(r) - eps)
    print("biến lỏng ξ =", xi.round(4), "| J =", round(0.5 * w * w + xi.sum(), 4), "(= 1729/450 = 3,8422)")

    # (c) ε điều khiển độ phẳng và số vector hỗ trợ (5 sinh viên sạch, C = 1)
    print("\nẢnh hưởng của ε (5 sinh viên sạch, C=1):")
    for e in [0.0, 0.2, 0.5, 1.0, 2.0]:
        s = SVR(kernel="linear", C=1, epsilon=e, tol=1e-12).fit(x[:, None], y)
        print(f"  ε={e:3.1f}: w = {s.coef_[0, 0]:.3f}, b = {s.intercept_[0]:.3f}, "
              f"#SV = {len(s.support_)}   (công thức khi 2 đầu là SV: w = 1 - ε/2 = {1 - e / 2:.3f})")

    # (d) SVR RBF trên y = sin x có ngoại lai: tự cài đặt (đối ngẫu) so với sklearn
    from sklearn.model_selection import train_test_split
    X, yv = du_lieu_sin_co_ngoai_lai()
    X, _, yv, _ = train_test_split(X, yv, test_size=0.3, random_state=0)       # 42 mẫu huấn luyện (như tài liệu)
    mu, sd, ym, sy = X.mean(), X.std(), yv.mean(), yv.std()
    Z, z = (X - mu) / sd, (yv - ym) / sy                  # chuẩn hóa CẢ x và y (để eps có thang đo hợp lý)
    mine = KernelSVR(lambda A, B: rbf_kernel(A, B, 1.0), C=10, eps=0.1).fit(Z, z)
    ref = SVR(kernel="rbf", gamma=1.0, C=10, epsilon=0.1, tol=1e-10).fit(Z, z)
    G = (np.linspace(0, 6, 400)[:, None] - mu) / sd
    sai_so_that = lambda f: np.sqrt(((np.sin(np.linspace(0, 6, 400)) - f) ** 2).mean())
    print(f"\nSVR RBF: #SV tự cài đặt {len(mine.coef)} | sklearn {len(ref.support_)} | "
          f"chênh lệch dự đoán lớn nhất {np.abs(mine.predict(G) - ref.predict(G)).max():.1e}")
    ols = LinearRegression().fit(X, yv)
    print("RMSE so với đường thật sin(x): hồi quy tuyến tính "
          f"{sai_so_that(ols.predict(np.linspace(0, 6, 400)[:, None])):.3f} | "
          f"SVR RBF {sai_so_that(mine.predict(G) * sy + ym):.3f}")


# =============================================================================
# 7. ĐÁNH GIÁ, PLATT SCALING, NHIỀU LỚP, CHỌN SIÊU THAM SỐ
# =============================================================================
def platt_scaling(f, y, n_iter=100):
    """Khớp  P(y=+1 | f) = sigma(A f + B)  bằng phương pháp Newton (Logistic Regression MỘT chiều).

    * Dùng nhãn LÀM MỊN của Platt: t+ = (N+ + 1)/(N+ + 2), t- = 1/(N- + 2) để tránh quá khớp.
    * f phải là điểm số NGOÀI MẪU (ví dụ từ cross_val_predict), nếu không sigmoid sẽ quá dốc.
    Cập nhật Newton:  (A, B) <- (A, B) - H^{-1} g, với
        g = [Σ (p-t) f, Σ (p-t)]      (gradient của cross-entropy)
        H = Σ p(1-p) [[f², f], [f, 1]] (Hessian, nửa xác định dương nên bước Newton ổn định)
    """
    sigma = lambda z: 1 / (1 + np.exp(-z))
    n_pos, n_neg = (y == 1).sum(), (y == -1).sum()
    t = np.where(y == 1, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))
    A, B = 0.0, np.log((n_pos + 1) / (n_neg + 1))
    for _ in range(n_iter):
        p = sigma(A * f + B)
        g = np.array([((p - t) * f).sum(), (p - t).sum()])
        w = p * (1 - p)
        H = np.array([[(w * f * f).sum(), (w * f).sum()], [(w * f).sum(), w.sum()]]) + 1e-9 * np.eye(2)
        buoc = np.linalg.solve(H, g)
        A, B = A - buoc[0], B - buoc[1]
        if np.abs(buoc).max() < 1e-10:
            break
    return A, B


def phan_7():
    tieu_de("MỤC 7. Đánh giá, trực quan hóa và chọn siêu tham số")
    from sklearn.svm import SVC, LinearSVC
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import confusion_matrix, roc_auc_score, log_loss
    from sklearn.model_selection import cross_val_predict, StratifiedKFold, GridSearchCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

    X, y = du_lieu_400_sinh_vien()
    Xtr, Xte, ytr, yte = chia_train_test(X, y)
    Ztr, Zte, mu, sd = chuan_hoa(Xtr, Xte)
    svm = SVC(kernel="linear", C=1).fit(Ztr, ytr)
    f = svm.decision_function(Zte)

    # (a) Các chỉ số phân loại tính từ ma trận nhầm lẫn [[TN, FP], [FN, TP]]
    (tn, fp), (fn, tp) = confusion_matrix(yte, svm.predict(Zte), labels=[-1, 1])
    print(f"Ma trận nhầm lẫn [[TN, FP], [FN, TP]] = [[{tn}, {fp}], [{fn}, {tp}]]")
    print(f"accuracy={(tp + tn) / len(yte):.3f}  precision={tp / (tp + fp):.3f}  recall={tp / (tp + fn):.3f}  "
          f"đặc hiệu={tn / (tn + fp):.3f}  F1={2 * tp / (2 * tp + fp + fn):.3f}  AUC={roc_auc_score(yte, f):.3f}")

    # (b) Platt scaling: học A, B trên điểm số NGOÀI MẪU bằng kiểm định chéo
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    f_cv = cross_val_predict(SVC(kernel="linear", C=1), Ztr, ytr, cv=cv, method="decision_function")
    A, B = platt_scaling(f_cv, ytr)
    sigma = lambda z: 1 / (1 + np.exp(-z))
    yb = (yte + 1) // 2
    print(f"\nPlatt scaling: A = {A:.4f}, B = {B:.4f} | log-loss: sigma(f) ngây thơ "
          f"{log_loss(yb, sigma(f)):.4f}, Platt {log_loss(yb, sigma(A * f + B)):.4f}")
    t_star = (np.log(0.75 / 0.25) - B) / A                # ngưỡng điểm số ứng với τ* = 0,75 (cFP = 3, cFN = 1)
    print(f"Ngưỡng theo chi phí (cFP=3, cFN=1): τ* = 0,75  ->  t* = (ln3 - B)/A = {t_star:.3f}")

    # (c) Nhiều lớp: OvR và OvO trên 3 lớp (yếu / khá / giỏi)
    rng = np.random.default_rng(7)
    means = [(3.5, 4.2), (6.0, 2.6), (8.5, 1.0)]
    Xm = np.clip(np.vstack([np.c_[rng.normal(m[0], 1.6, 100), rng.normal(m[1], 1.2, 100)] for m in means]), 0, None)
    ym = np.repeat([0, 1, 2], 100)
    from sklearn.model_selection import train_test_split
    Xa, Xb, ya, yb3 = train_test_split(Xm, ym, test_size=0.3, random_state=0, stratify=ym)
    Za, Zb, _, _ = chuan_hoa(Xa, Xb)
    ovr = OneVsRestClassifier(SVC(kernel="linear", C=1)).fit(Za, ya)
    ovo = OneVsOneClassifier(SVC(kernel="linear", C=1)).fit(Za, ya)
    print(f"\n3 lớp: One-vs-Rest {ovr.score(Zb, yb3):.3f} | One-vs-One {ovo.score(Zb, yb3):.3f}")
    i = 13                                               # một sinh viên ví dụ (thực tế loại "khá")
    diem_ovr = [e.decision_function(Zb[i:i + 1])[0] for e in ovr.estimators_]
    phieu = np.zeros(3)
    for (a, b_), e in zip([(0, 1), (0, 2), (1, 2)], ovo.estimators_):
        phieu[b_ if e.decision_function(Zb[i:i + 1])[0] > 0 else a] += 1      # >0: nghiêng về lớp thứ hai của cặp
    print(f"  mẫu {i}: điểm OvR = {np.round(diem_ovr, 3)} -> argmax = {int(np.argmax(diem_ovr))}; "
          f"phiếu OvO = {phieu.astype(int).tolist()} -> {int(np.argmax(phieu))} (thực tế {yb3[i]})")

    # (d) Chỉ số hồi quy: ví dụ tay 5 mẫu (một dự đoán sai nặng)
    yt = np.array([2, 3, 5, 7, 9.0])
    for ten, yh in [("5 dự đoán tốt", [2.5, 2.8, 5.4, 6.0, 9.5]), ("có dự đoán sai nặng", [2.5, 2.8, 5.4, 6.0, 14.5])]:
        r = yt - np.array(yh)
        r2 = 1 - (r ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
        print(f"  {ten:20s}: MAE={np.abs(r).mean():.3f}  RMSE={np.sqrt((r ** 2).mean()):.3f}  R2={r2:.3f}")

    # (e) Grid search với pipeline (chuẩn hóa nằm BÊN TRONG để không rò rỉ dữ liệu)
    pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
    luoi = {"svc__C": [0.1, 1, 10, 100], "svc__gamma": [0.001, 0.01, 0.1, 1]}
    gs = GridSearchCV(pipe, luoi, cv=cv).fit(Xtr, ytr)
    print(f"\nGrid search RBF: tốt nhất {gs.best_params_}, CV = {gs.best_score_:.4f}, kiểm tra = {gs.score(Xte, yte):.4f}")

    # (f) So sánh các cách triển khai SVM tuyến tính (cùng nghiệm, khác tốc độ)
    print("\nSVM tuyến tính C=1 trên 400 sinh viên (đặc trưng chuẩn hóa):")
    mine = LinearSVM(C=1, n_iter=1000).fit(Ztr, ytr)
    models = [("LinearSVM tự cài đặt", mine.w, mine.b),
              ("SVC(kernel='linear')", svm.coef_[0], svm.intercept_[0])]
    lin = LinearSVC(C=1, loss="hinge", dual=True, max_iter=200000, tol=1e-8).fit(Ztr, ytr)
    sgd = SGDClassifier(loss="hinge", alpha=1 / len(ytr), max_iter=5000, tol=None, random_state=0).fit(Ztr, ytr)
    models += [("LinearSVC(loss='hinge')", lin.coef_[0], lin.intercept_[0]),
               ("SGDClassifier(hinge)", sgd.coef_[0], sgd.intercept_[0])]
    for ten, w, b in models:
        print(f"  {ten:24s} w = {np.round(w, 4)}, b = {b:.4f}")


# =============================================================================
# 8. TỔNG KẾT VÀ MỞ RỘNG
# =============================================================================
def softplus_dich(m, c, theta):
    """l_{c,θ}(m) = (1/c) ln(1 + exp(c(θ - m))). np.logaddexp(0, z) = ln(1 + e^z) tính ỔN ĐỊNH số học."""
    return np.logaddexp(0, c * (theta - m)) / c


class FourierNgauNhien:
    """Xấp xỉ kernel RBF bằng đặc trưng Fourier ngẫu nhiên (Rahimi & Recht).

    Định lý Bochner: exp(-γ||x-z||²) = E_ω[cos(ω(x-z)^T)] với ω ~ N(0, 2γ I).
    Lấy D mẫu ω rồi dùng đặc trưng  z(x) = sqrt(2/D) cos(xΩ + b)  thì  k(x,z) ≈ z(x)·z(z).
    Sau đó chỉ cần SVM TUYẾN TÍNH trên z(x): chi phí O(nD) thay vì O(n²).
    """

    def __init__(self, gamma=1.0, D=200, seed=0):
        self.gamma, self.D, self.seed = gamma, D, seed

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        self.Omega = rng.normal(0, np.sqrt(2 * self.gamma), (X.shape[1], self.D))   # các cột ω ~ N(0, 2γI)
        self.b = rng.uniform(0, 2 * np.pi, self.D)
        return self

    def transform(self, X):
        return np.sqrt(2 / self.D) * np.cos(X @ self.Omega + self.b)


def phan_8():
    tieu_de("MỤC 8. Tổng kết và mở rộng")
    from sklearn.svm import SVC, LinearSVC, OneClassSVM
    from sklearn.metrics import recall_score, precision_score, roc_auc_score
    from sklearn.datasets import make_circles
    from sklearn.model_selection import train_test_split

    # (a) Họ hàm mất mát: softplus dịch chuyển co về hinge khi c -> vô cùng
    print("Mất mát (1/c)ln(1+e^{c(1-m)}) co về hinge max(0, 1-m) khi c tăng:")
    for m in [0.5, 1.5]:
        print(f"  m={m}: " + ", ".join(f"c={c}: {softplus_dich(m, c, 1.0):.4f}" for c in [1, 5, 20, 100])
              + f"  | hinge = {max(0, 1 - m)}")

    # (b) Lớp mất cân bằng: class_weight = hệ số phạt C riêng cho từng lớp (C_i = C * w_{y_i})
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal([0, 0], 1.2, (950, 2)), rng.normal([1.8, 1.8], 1.0, (50, 2))])
    y = np.r_[-np.ones(950), np.ones(50)]
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    Za, Zb, _, _ = chuan_hoa(Xa, Xb)
    print("\nLớp mất cân bằng 95/5 (test có 15 mẫu dương):")
    for ten, cw in [("không trọng số", None), ("balanced", "balanced")]:
        s = SVC(kernel="linear", C=1, class_weight=cw).fit(Za, ya)
        p = s.predict(Zb)
        print(f"  {ten:15s}: accuracy {np.mean(p == yb):.3f} | recall {recall_score(yb, p):.3f} | "
              f"precision {precision_score(yb, p):.3f} | AUC {roc_auc_score(yb, s.decision_function(Zb)):.3f}")

    # (c) One-class SVM: ν là cận trên tỉ lệ vector bị chặn và cận dưới tỉ lệ vector hỗ trợ
    Xs, ys = du_lieu_400_sinh_vien()
    Xtr, _, ytr, _ = chia_train_test(Xs, ys)
    Zn, _, _ = chuan_hoa(Xtr[ytr == 1])
    print("\nOne-class SVM trên các sinh viên đậu (RBF γ=0,5):")
    print("     ν | vector bị chặn | vector hỗ trợ | (ν phải nằm giữa hai tỉ lệ này)")
    for nu in [0.05, 0.1, 0.2]:
        o = OneClassSVM(kernel="rbf", gamma=0.5, nu=nu, tol=1e-9).fit(Zn)
        a = np.abs(o.dual_coef_[0])
        bi_chan = (a > a.max() * (1 - 1e-6)).sum() / len(Zn)
        print(f"  {nu:5.2f} | {bi_chan:14.3f} | {len(o.support_) / len(Zn):13.3f} |")

    # (d) Xấp xỉ kernel: sai số theo D và tốc độ
    x, z, gamma = np.zeros(2), np.ones(2), 0.5
    k_that = np.exp(-gamma * ((x - z) ** 2).sum())
    print(f"\nFourier ngẫu nhiên: k thật = {k_that:.4f}; sai số trung bình theo D:")
    for D in [10, 100, 1000, 10000]:
        loi = []
        for s in range(20):
            rf = FourierNgauNhien(gamma, D, seed=s).fit(np.zeros((1, 2)))
            loi.append(abs(rf.transform(x[None]) @ rf.transform(z[None]).T - k_that).item())
        print(f"  D = {D:6d}: sai số = {np.mean(loi):.4f}")
    Xc, yc = make_circles(20000, noise=0.15, factor=0.45, random_state=1)
    Xa, Xb, ya, yb = train_test_split(Xc, yc, test_size=0.3, random_state=0)
    Za, Zb, _, _ = chuan_hoa(Xa, Xb)
    t0 = time.time(); exact = SVC(kernel="rbf", gamma=1, C=1).fit(Za, ya); t_exact = time.time() - t0
    rf = FourierNgauNhien(1.0, 200).fit(Za)
    t0 = time.time(); lin = LinearSVC(C=1, max_iter=100000).fit(rf.transform(Za), ya); t_rf = time.time() - t0
    print(f"n = 20000: SVC RBF {t_exact:.2f}s, accuracy {exact.score(Zb, yb):.4f} | "
          f"Fourier+LinearSVC {t_rf:.2f}s, accuracy {lin.score(rf.transform(Zb), yb):.4f}")


# =============================================================================
# HÌNH MINH HỌA (tùy chọn: --plots)
# =============================================================================
def ve_hinh(thu_muc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.svm import SVC, SVR
    from sklearn.linear_model import LinearRegression
    os.makedirs(thu_muc, exist_ok=True)
    DO, XANH = "#d62728", "#1f77b4"

    # Hình 1: đường biên lề cực đại và dải lề trên 6 sinh viên
    X, y = du_lieu_6_sinh_vien()
    w, b, _ = hard_margin_tu_cap_gan_nhat(X, y)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x1 = np.array([0, 10.0])
    for c, ls, lw in [(0, "-", 2), (1, "--", 1), (-1, "--", 1)]:
        ax.plot(x1, -(b - c + w[0] * x1) / w[1], "k", ls=ls, lw=lw)
    ax.scatter(*X[y == -1].T, c=DO, label="rớt"); ax.scatter(*X[y == 1].T, c=XANH, marker="s", label="đậu")
    m = margin_ham(X, y, w, b)
    ax.scatter(*X[np.isclose(m, 1)].T, s=220, facecolors="none", edgecolors="g", lw=2)
    ax.set_xlim(0, 10); ax.set_ylim(-1, 6); ax.set_aspect("equal"); ax.legend()
    ax.set_title("Lề cứng: f=0 và f=±1, vector hỗ trợ khoanh xanh")
    fig.savefig(os.path.join(thu_muc, "hinh_1_le_cung.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

    # Hình 2: các hàm mất mát theo margin
    m = np.linspace(-2.5, 3, 400)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(m, np.maximum(0, -m), label="Perceptron max(0,-m)")
    ax.plot(m, np.log1p(np.exp(-m)), label="Logistic ln(1+e^-m)")
    ax.plot(m, np.maximum(0, 1 - m), label="Hinge max(0,1-m)", lw=2.5)
    ax.set_xlabel("margin m"); ax.set_ylabel("mất mát"); ax.set_ylim(-0.1, 3.5); ax.legend()
    fig.savefig(os.path.join(thu_muc, "hinh_2_ham_mat_mat.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

    # Hình 3: XOR với kernel đa thức bậc 2 và RBF
    X, y = du_lieu_xor()
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (ten, s) in zip(axs, [("kernel (xz^T+1)^2", SVC(kernel="poly", degree=2, gamma=1, coef0=1, C=1e8)),
                                  ("kernel RBF γ=1", SVC(kernel="rbf", gamma=1, C=1e6))]):
        s.fit(X, y)
        g = np.linspace(-0.6, 1.6, 200)
        XX, YY = np.meshgrid(g, g)
        F = s.decision_function(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
        ax.contourf(XX, YY, F, levels=[-1e9, 0, 1e9], colors=["#fbeeee", "#eaf2fa"])
        ax.contour(XX, YY, F, levels=[0], colors="k", linewidths=2)
        ax.scatter(*X[y == -1].T, c=DO, s=80); ax.scatter(*X[y == 1].T, c=XANH, marker="s", s=80)
        ax.set_title("XOR, " + ten); ax.set_aspect("equal")
    fig.savefig(os.path.join(thu_muc, "hinh_3_xor_kernel.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

    # Hình 4: hồi quy MSE và SVR khi ngoại lai di chuyển
    x = np.array([1, 2, 3, 4, 5.0]); y = np.array([1.0, 2.2, 2.9, 4.0, 5.0]); xs = np.linspace(0, 7, 50)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for y6, c in [(3.0, "g"), (2.0, "orange"), (-2.0, "r")]:
        xx, yy = np.r_[x, 6.0], np.r_[y, y6]
        ols = LinearRegression().fit(xx[:, None], yy)
        svr = SVR(kernel="linear", C=1, epsilon=0.2, tol=1e-12).fit(xx[:, None], yy)
        axs[0].plot(xs, ols.coef_[0] * xs + ols.intercept_, c=c)
        axs[1].plot(xs, svr.coef_[0, 0] * xs + svr.intercept_[0], c=c, ls="--" if y6 != 3.0 else "-")
        for ax in axs:
            ax.scatter([6], [y6], c=c, marker="s")
    for ax, t in zip(axs, ["Hồi quy MSE: đường bị kéo theo ngoại lai", "SVR: đường không đổi"]):
        ax.scatter(x, y, c=XANH); ax.set_title(t); ax.set_ylim(-2.8, 6)
    fig.savefig(os.path.join(thu_muc, "hinh_4_svr_ngoai_lai.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\nĐã lưu 4 hình vào thư mục: {thu_muc}/")


# =============================================================================
# ĐIỂM VÀO CHƯƠNG TRÌNH
# =============================================================================
PHAN = {1: phan_1, 2: phan_2, 3: phan_3, 4: phan_4, 5: phan_5, 6: phan_6, 7: phan_7, 8: phan_8}


def main():
    ap = argparse.ArgumentParser(description="Minh họa SVM/SVR (mục 1 đến 8 của bài giảng)")
    ap.add_argument("--part", type=int, nargs="*", choices=sorted(PHAN), help="chỉ chạy các mục này (mặc định: tất cả)")
    ap.add_argument("--plots", action="store_true", help="lưu thêm vài hình minh họa")
    ap.add_argument("--out", default="hinh_svm", help="thư mục lưu hình (mặc định: hinh_svm)")
    args = ap.parse_args()
    for k in (args.part or sorted(PHAN)):
        t0 = time.time()
        PHAN[k]()
        print(f"\n[mục {k} xong trong {time.time() - t0:.1f} giây]")
    if args.plots:
        ve_hinh(args.out)


if __name__ == "__main__":
    main()
