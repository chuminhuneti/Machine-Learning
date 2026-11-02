#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CÁC MÔ HÌNH PHI THAM SỐ - mã minh họa cho toàn bộ tài liệu (mục 1 đến 8)
=========================================================================

Mỗi "phần" của file này ứng với một mục của tài liệu. Chạy:

    python phi_tham_so_demo.py                 # chạy mọi phần (lõi chỉ cần numpy)
    python phi_tham_so_demo.py --phan 2 5      # chỉ chạy phần 2 và 5
    python phi_tham_so_demo.py --ve            # kèm vẽ hình vào thư mục hinh_demo/
    python phi_tham_so_demo.py --seed 7        # đổi hạt giống ngẫu nhiên

Yêu cầu:  numpy (bắt buộc).
          matplotlib (chỉ khi dùng --ve), scikit-learn (chỉ cho phần 9: chữ số viết tay
          và đối chiếu với thư viện; thiếu thì phần đó bị bỏ qua).

Quy ước ký hiệu giống tài liệu: X là ma trận đặc trưng (n x d), Y là ma trận one-hot
(n x K), X0 là ma trận điểm truy vấn (m x d). Tên biến bằng tiếng Việt không dấu.

Lưu ý: dữ liệu mô phỏng ở đây sinh bằng numpy, cách chia train/test cũng tự viết, nên
các con số có thể lệch vài phần trăm so với các bảng trong tài liệu (vốn dùng
scikit-learn để chia). Kết luận định tính thì giữ nguyên.

Những đoạn mã KHÓ được đánh dấu bằng chú thích "GIẢI THÍCH:" ngay bên trên.
"""

import argparse
import os
import time
from math import gamma, pi

import numpy as np


# =============================================================================
# PHẦN 0. TIỆN ÍCH: dữ liệu mô phỏng, chia tập, chuẩn hóa, một-nóng
# =============================================================================

def mot_nong(y, K=None):
    """Nhãn nguyên (n,) -> ma trận one-hot Y (n x K)."""
    K = int(y.max()) + 1 if K is None else K
    return np.eye(K)[y]


def tao_du_lieu_sinh_vien(n_moi_lop=200, seed=0):
    """3 lớp 'rớt/đạt/giỏi', 2 đặc trưng (giờ tự học, số buổi vắng); đám Gauss chồng lấn."""
    rng = np.random.default_rng(seed)
    mu = [(3.5, 4.2), (6.5, 2.4), (9.5, 1.0)]
    sd = [(1.6, 1.3), (1.7, 1.2), (1.6, 1.0)]
    X, y = [], []
    for c in range(3):
        X.append(np.c_[rng.normal(mu[c][0], sd[c][0], n_moi_lop),
                       rng.normal(mu[c][1], sd[c][1], n_moi_lop)])
        y += [c] * n_moi_lop
    X = np.vstack(X)
    X[:, 0] = np.clip(np.round(X[:, 0], 1), 0, None)   # giờ học >= 0, làm tròn 0,1
    X[:, 1] = np.clip(np.round(X[:, 1]), 0, None)      # số buổi vắng là số nguyên >= 0
    return X, np.array(y)


def tao_ba_vong_tron(n_moi_lop=200, seed=0):
    """Lớp 0 là hình tròn đặc bán kính 1; lớp 1, 2 là hai vòng bán kính 3 và 5 (mục 1.1)."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for k, r in enumerate([1, 3, 5]):
        theta = rng.uniform(0, 2 * np.pi, n_moi_lop)
        # GIẢI THÍCH: lớp 0 lấy bán kính = căn(U) để điểm phân bố ĐỀU trên hình tròn đặc
        # (nếu lấy bán kính U thì điểm dồn về tâm). Hai vòng còn lại: bán kính r + nhiễu Gauss.
        rr = np.sqrt(rng.uniform(0, 1, n_moi_lop)) if k == 0 else r + rng.normal(0, 0.3, n_moi_lop)
        X.append(np.c_[rr * np.cos(theta), rr * np.sin(theta)])
        y += [k] * n_moi_lop
    return np.vstack(X), np.array(y)


def ham_that_sin(x):
    return np.sin(1.2 * x) + 0.15 * x


def tao_du_lieu_sin(n=100, sigma=0.3, seed=0):
    """Bài toán hồi quy y = sin(1,2x) + 0,15x + nhiễu, x ~ U(0,10) (mục 3)."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 10, n))
    return x, ham_that_sin(x) + rng.normal(0, sigma, n)


def mat_do_that_hai_dinh(x):
    """Mật độ thật của hỗn hợp 0,6 N(-1,5; 0,7^2) + 0,4 N(2; 1^2) (mục 4)."""
    def gauss(t, mu, sd):
        return np.exp(-0.5 * ((t - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    return 0.6 * gauss(x, -1.5, 0.7) + 0.4 * gauss(x, 2.0, 1.0)


def lay_mau_hai_dinh(n, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.random(n) < 0.6
    return np.where(z, rng.normal(-1.5, 0.7, n), rng.normal(2.0, 1.0, n))


def chia_phan_tang(y, ti_le_test=0.3, seed=0):
    """Chia train/test sao cho mỗi lớp giữ cùng tỉ lệ (phân tầng). Trả về (chỉ số train, chỉ số test)."""
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        n_te = int(round(ti_le_test * len(idx)))
        te += list(idx[:n_te]); tr += list(idx[n_te:])
    return np.array(tr), np.array(te)


class ChuanHoa:
    """Chuẩn hóa z-score. QUAN TRỌNG: chỉ học (mu, sigma) từ tập HUẤN LUYỆN (mục 2.5)."""
    def fit(self, X):
        self.mu, self.sigma = X.mean(axis=0), X.std(axis=0) + 1e-12
        return self

    def transform(self, X):
        return (X - self.mu) / self.sigma


def logsumexp(S):
    """log(sum(exp(S))) theo từng hàng, ổn định số: trừ giá trị lớn nhất trước khi lấy exp."""
    m = S.max(axis=1, keepdims=True)
    return m[:, 0] + np.log(np.exp(S - m).sum(axis=1))


def softmax(S):
    return np.exp(S - logsumexp(S)[:, None])


# =============================================================================
# PHẦN 1. MÔ HÌNH THAM SỐ (Softmax) LÀM MỐC SO SÁNH - mục 1.1
# =============================================================================

class SoftmaxHoiQuy:
    """Softmax Regression huấn luyện bằng Gradient Descent (đại diện mô hình THAM SỐ)."""

    def __init__(self, toc_do_hoc=0.5, so_vong=1500, lam=1e-3):
        self.lr, self.T, self.lam = toc_do_hoc, so_vong, lam

    def fit(self, X, Y):
        self.cn = ChuanHoa().fit(X)
        Xt = np.c_[np.ones(len(X)), self.cn.transform(X)]            # thêm cột 1 (hệ số chặn)
        self.W = np.zeros((Xt.shape[1], Y.shape[1]))
        for _ in range(self.T):
            P = softmax(Xt @ self.W)
            # GIẢI THÍCH: với cross-entropy + softmax, gradient theo W gọn như hồi quy tuyến
            # tính: (1/n) Xt^T (P - Y). Hệ số chặn (hàng 0 của W) không bị điều chuẩn.
            grad = Xt.T @ (P - Y) / len(Xt)
            grad[1:] += self.lam * self.W[1:]
            self.W -= self.lr * grad
        return self

    def xac_suat(self, X0):
        return softmax(np.c_[np.ones(len(X0)), self.cn.transform(X0)] @ self.W)

    def du_doan(self, X0):
        return self.xac_suat(X0).argmax(axis=1)


# =============================================================================
# PHẦN 2. k-NN PHÂN LOẠI - mục 2
# =============================================================================

def binh_phuong_khoang_cach(X0, X):
    """Ma trận D2 (m x n) với D2[j, i] = ||X0[j] - X[i]||^2, không dùng vòng lặp.

    GIẢI THÍCH: khai triển ||u - v||^2 = ||u||^2 - 2 u.v + ||v||^2. Ba thành phần gom thành
    một biểu thức ma trận: a (m x 1) cộng b (1 x n) nhờ "broadcasting" của numpy (cột a được
    lặp ngang, hàng b được lặp dọc), trừ 2 * X0 @ X.T (phép nhân ma trận m x d với d x n,
    phần nặng nhất nhưng thư viện đại số tuyến tính làm rất nhanh).
    """
    a = np.sum(X0 ** 2, axis=1, keepdims=True)    # (m, 1)
    b = np.sum(X ** 2, axis=1)[None, :]           # (1, n)
    D2 = a - 2.0 * X0 @ X.T + b
    return np.maximum(D2, 0.0)                    # sai số làm tròn có thể cho số âm ~ -1e-16


def knn_xac_suat(X0, X, Y, k=3, co_trong_so=False, eps=1e-12):
    """Ma trận xác suất lớp P_hat (m x K) = (1/k) N Y  (hoặc N_w Y nếu có trọng số)."""
    m, n = X0.shape[0], X.shape[0]
    D2 = binh_phuong_khoang_cach(X0, X)
    # GIẢI THÍCH: argsort(stable) cho thứ tự lặp lại được khi có hai điểm cách đều (hòa).
    # Có thể thay bằng np.argpartition(D2, k-1)[:, :k] (O(n) thay vì O(n log n)) nhưng khi
    # hòa thì chọn láng giềng nào là tùy ý.
    idx = np.argsort(D2, axis=1, kind="stable")[:, :k]       # (m, k) chỉ số k láng giềng
    N = np.zeros((m, n))
    hang = np.arange(m)[:, None]                              # (m, 1)
    # GIẢI THÍCH: "fancy indexing" - cặp (hang, idx) có hình (m,1) và (m,k) được broadcast
    # thành (m,k): ô N[j, idx[j, t]] được gán, tức đánh dấu k láng giềng của mỗi hàng.
    N[hang, idx] = 1.0 / (D2[hang, idx] + eps) if co_trong_so else 1.0
    return (N @ Y) / N.sum(axis=1, keepdims=True)             # chia tổng hàng = chuẩn hóa


def knn_du_doan(X0, X, y, k=3, co_trong_so=False):
    return knn_xac_suat(X0, X, mot_nong(y), k, co_trong_so).argmax(axis=1)


def chon_k_kiem_dinh_cheo(X, y, ds_k, V=5, seed=0):
    """Chọn k bằng kiểm định chéo V phần. Chuẩn hóa được HỌC LẠI trong từng phần (tránh rò)."""
    rng = np.random.default_rng(seed)
    nhom = rng.permutation(len(y)) % V                        # gán mỗi mẫu vào một trong V phần
    diem = {k: [] for k in ds_k}
    for v in range(V):
        tr, va = nhom != v, nhom == v
        cn = ChuanHoa().fit(X[tr])                            # chỉ dùng V-1 phần để chuẩn hóa
        Ztr, Zva = cn.transform(X[tr]), cn.transform(X[va])
        for k in ds_k:
            diem[k].append((knn_du_doan(Zva, Ztr, y[tr], k) == y[va]).mean())
    cv = {k: float(np.mean(v)) for k, v in diem.items()}
    return max(cv, key=cv.get), cv


# =============================================================================
# PHẦN 3. HỒI QUY: k-NN, NADARAYA-WATSON, LOESS - mục 3
# =============================================================================

def knn_hoi_quy(X0, X, y, k=3):
    D2 = binh_phuong_khoang_cach(X0, X)
    idx = np.argsort(D2, axis=1, kind="stable")[:, :k]
    return y[idx].mean(axis=1)                                # (1/k) N y


def chon_k_hoi_quy_bo_mot_ra(x, y, ds_k):
    """Chọn k cho k-NN hồi quy một chiều bằng bỏ-một-ra (bỏ thật từng mẫu, vì k-NN không có
    công thức đóng như bộ làm trơn có băng thông cố định)."""
    D = np.abs(x[:, None] - x[None, :])
    np.fill_diagonal(D, np.inf)                       # GIẢI THÍCH: đặt khoảng cách tới chính mình = vô cực
    thu_tu = np.argsort(D, axis=1, kind="stable")     # -> mẫu i không bao giờ là láng giềng của chính nó
    loi = {k: np.mean((y - y[thu_tu[:, :k]].mean(axis=1)) ** 2) for k in ds_k}
    return min(loi, key=loi.get)


def ham_nhan(u, loai="gauss"):
    if loai == "gauss":
        return np.exp(-0.5 * u ** 2)
    if loai == "epanechnikov":
        return np.maximum(1.0 - u ** 2, 0.0)
    if loai == "deu":
        return (np.abs(u) <= 1.0).astype(float)
    raise ValueError(loai)


def nadaraya_watson(X0, X, y, h, loai="gauss"):
    G = ham_nhan(np.sqrt(binh_phuong_khoang_cach(X0, X)) / h, loai)   # (m, n)
    return (G @ y) / G.sum(axis=1)                            # = diag(G 1)^-1 G y


def loess(X0, X, y, h, loai="gauss", lam=1e-8):
    """Hồi quy tuyến tính trọng số địa phương: w(x0) = (X~^T V X~)^-1 X~^T V y."""
    Xt = np.c_[np.ones(len(X)), X]                            # X~: thêm cột 1
    Gv = ham_nhan(np.sqrt(binh_phuong_khoang_cach(X0, X)) / h, loai)  # (m, n): v_i(x0)
    ket_qua = np.empty(len(X0))
    for j in range(len(X0)):                                  # mỗi điểm truy vấn giải một hệ riêng
        v = Gv[j][:, None]                                    # (n, 1) = đường chéo của V
        # GIẢI THÍCH: không dựng ma trận đường chéo V (n x n) mà nhân từng hàng: (v * Xt)
        # bằng V @ Xt nhưng chỉ tốn O(n d). lam*I giúp ma trận không suy biến khi ít mẫu.
        A = Xt.T @ (v * Xt) + lam * np.eye(Xt.shape[1])
        w = np.linalg.solve(A, Xt.T @ (v[:, 0] * y))          # solve ổn định hơn inv(A) @ b
        ket_qua[j] = np.r_[1.0, X0[j]] @ w                    # x0~ w(x0)
    return ket_qua


def ma_tran_lam_muot(x, h, kieu="nw"):
    """Ma trận làm trơn S (n x n) sao cho y_hat = S y trên chính các mẫu huấn luyện."""
    n = len(x)
    G = ham_nhan(np.abs(x[:, None] - x[None, :]) / h)
    if kieu == "nw":
        return G / G.sum(axis=1, keepdims=True)
    Xa = np.c_[np.ones(n), x]
    S = np.zeros((n, n))
    for j in range(n):                                        # hàng j của S = x_j~ (X~^T V X~)^-1 X~^T V
        v = G[j][:, None]
        A = np.linalg.inv(Xa.T @ (v * Xa) + 1e-8 * np.eye(2))
        S[j] = (np.array([1.0, x[j]]) @ A @ Xa.T) * G[j]
    return S


def loocv_bo_mot_ra(S, y):
    """Sai số bỏ-một-ra của bộ làm trơn tuyến tính mà KHÔNG phải huấn luyện lại n lần.

    GIẢI THÍCH: với y_hat = S y, dự đoán khi bỏ mẫu i là (y_hat_i - S_ii y_i)/(1 - S_ii),
    nên sai số bỏ-một-ra là (y_i - y_hat_i)/(1 - S_ii). Công thức này CHÍNH XÁC (không xấp xỉ)
    với bộ làm trơn tuyến tính có băng thông cố định (mục 3.4).
    """
    r = (y - S @ y) / (1.0 - np.diag(S))
    return float(np.mean(r ** 2))


# =============================================================================
# PHẦN 4. ƯỚC LƯỢNG MẬT ĐỘ: HISTOGRAM, KDE, k LÁNG GIỀNG, GAUSS - mục 4
# =============================================================================

def histogram_mat_do(x0, x, delta, goc=0.0):
    """f(x) = n_j / (n * delta) với j là ô chứa x."""
    chi_so_x = np.floor((x - goc) / delta).astype(int)
    chi_so_0 = np.floor((np.atleast_1d(x0) - goc) / delta).astype(int)
    return np.array([(chi_so_x == b).sum() for b in chi_so_0]) / (len(x) * delta)


def kde(X0, X, h):
    """KDE nhân Gauss: f = (1/n) (2 pi)^(-d/2) h^(-d) * G 1_n, G[j,i] = exp(-D2[j,i]/(2h^2))."""
    n, d = X.shape
    G = np.exp(-binh_phuong_khoang_cach(X0, X) / (2.0 * h ** 2))
    return G.sum(axis=1) / (n * (2.0 * pi) ** (d / 2) * h ** d)


def the_tich_cau(d, r):
    return pi ** (d / 2) / gamma(d / 2 + 1) * r ** d          # omega_d * r^d


def knn_mat_do(X0, X, k):
    """f(x) = k / (n V_k(x)); V_k là thể tích hình cầu bán kính = khoảng cách tới láng giềng thứ k."""
    n, d = X.shape
    r = np.sqrt(np.sort(binh_phuong_khoang_cach(X0, X), axis=1)[:, k - 1])
    return k / (n * the_tich_cau(d, r))


def silverman(x):
    """Quy tắc Silverman: h = 0,9 min(sigma, IQR/1,34) n^(-1/5) (một chiều)."""
    q75, q25 = np.percentile(x, [75, 25])
    return 0.9 * min(x.std(ddof=1), (q75 - q25) / 1.34) * len(x) ** (-0.2)


def log_kha_nang_bo_mot_ra(X, h):
    """LCV(h) = (1/n) sum_i ln f_{-i}(x_i): mật độ tại x_i tính KHÔNG dùng chính x_i."""
    n, d = X.shape
    G = np.exp(-binh_phuong_khoang_cach(X, X) / (2.0 * h ** 2))
    # GIẢI THÍCH: đường chéo của G ứng với khoảng cách 0 (mẫu so với chính nó), giá trị 1.
    # Nếu giữ lại thì f(x_i) có thêm số hạng không đổi và LCV luôn tăng khi h -> 0 (học thuộc
    # cực đoan). Đặt đường chéo = 0 và chia cho (n - 1) chính là "bỏ một ra".
    np.fill_diagonal(G, 0.0)
    f = G.sum(axis=1) / ((n - 1) * (2.0 * pi) ** (d / 2) * h ** d)
    return float(np.mean(np.log(f + 1e-300)))


# =============================================================================
# PHẦN 5. BAYES NGÂY THƠ - mục 5
# =============================================================================

class GaussNB:
    """Bayes ngây thơ Gauss. 'Huấn luyện' = vài phép nhân ma trận với Y^T (công thức đóng)."""
    def __init__(self, lam=1e-9):
        self.lam = lam

    def fit(self, X, Y):
        t = Y.sum(axis=0)                                     # t_c = số mẫu mỗi lớp
        self.pi = t / len(X)
        self.mu = (Y.T @ X) / t[:, None]                      # M = diag(t)^-1 Y^T X   (K x d)
        E2 = (Y.T @ (X * X)) / t[:, None]                     # trung bình của bình phương
        # phương sai = E[x^2] - (E[x])^2; cộng sàn nhỏ để không chia cho 0 (mục 7.4)
        self.var = E2 - self.mu ** 2 + self.lam * X.var(axis=0).max()
        return self

    def diem_so(self, X0):
        """s_c = ln pi_c + sum_j ln N(x_j; mu_cj, var_cj)."""
        S = np.log(self.pi)[None, :] + np.zeros((len(X0), 1))
        for c in range(len(self.pi)):
            S[:, c] += np.sum(-0.5 * np.log(2 * pi * self.var[c])
                              - 0.5 * (X0 - self.mu[c]) ** 2 / self.var[c], axis=1)
        return S

    def xac_suat(self, X0):
        return softmax(self.diem_so(X0))

    def du_doan(self, X0):
        return self.diem_so(X0).argmax(axis=1)


class MultiNB:
    """Bayes ngây thơ Multinomial (đếm từ) với làm trơn Laplace alpha."""
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, Y):
        self.pi = Y.sum(axis=0) / len(X)
        C = Y.T @ X                                           # ma trận đếm K x d
        self.theta = (C + self.alpha) / (C.sum(axis=1, keepdims=True) + self.alpha * X.shape[1])
        return self

    def diem_so(self, X0):
        # GIẢI THÍCH: S = 1 ln(pi)^T + X0 ln(Theta)^T  -  CÙNG DẠNG với Softmax Regression
        # S = X~ W, chỉ khác W được tính bằng đếm (hợp lý đồng thời) chứ không bằng GD (mục 5.5).
        return np.log(self.pi)[None, :] + X0 @ np.log(self.theta).T

    def xac_suat(self, X0):
        return softmax(self.diem_so(X0))

    def du_doan(self, X0):
        return self.diem_so(X0).argmax(axis=1)


# =============================================================================
# PHẦN 6. THIÊN LỆCH - PHƯƠNG SAI VÀ LỜI NGUYỀN CHIỀU - mục 6
# =============================================================================

def phan_ra_sai_so(S, f_tren_x, f_tren_luoi, sigma):
    """Thiên lệch^2 và phương sai trung bình của bộ làm trơn y_hat = S y (thiết kế cố định).

    GIẢI THÍCH: với y = f(x) + nhiễu độc lập, E[y_hat] = S f nên thiên lệch = S f - f;
    Var[y_hat_j] = sigma^2 * sum_i S_ji^2 (mỗi y_i đóng góp S_ji * nhiễu_i, độc lập nhau).
    Với k-NN, S_ji = 1/k trên k mẫu nên phương sai = sigma^2 / k đúng như mục 6.1.
    """
    bias2 = np.mean((S @ f_tren_x - f_tren_luoi) ** 2)
    var = sigma ** 2 * np.mean(np.sum(S ** 2, axis=1))
    return float(bias2), float(var)


def ma_tran_trong_so_tai_luoi(x, luoi, tham_so, kieu):
    """Ma trận S_luoi (m x n): hàng j là trọng số hiệu dụng tại điểm lưới j."""
    D = np.abs(luoi[:, None] - x[None, :])
    if kieu == "knn":
        k = tham_so
        S = np.zeros_like(D)
        idx = np.argsort(D, axis=1, kind="stable")[:, :k]
        np.put_along_axis(S, idx, 1.0 / k, axis=1)
        return S
    G = ham_nhan(D / tham_so)
    return G / G.sum(axis=1, keepdims=True)


def ty_le_tap_trung(d, n=1000, seed=0):
    """(d_xa_nhat - d_gan_nhat)/d_gan_nhat với n điểm đều trong [0,1]^d (mục 4.6, 6.3)."""
    rng = np.random.default_rng(seed)
    P, q = rng.random((n, d)), rng.random((1, d))
    dist = np.linalg.norm(P - q, axis=1)
    return float((dist.max() - dist.min()) / dist.min())


# =============================================================================
# PHẦN 7. ĐÁNH GIÁ - mục 7
# =============================================================================

def ma_tran_nham_lan(y_that, y_du_doan, K):
    """M = Y^T Y_hat: M[a, b] = số mẫu thật là a nhưng bị dự đoán là b."""
    return mot_nong(y_that, K).T @ mot_nong(y_du_doan, K)


def precision_recall_f1(M):
    """Theo từng lớp. Precision = M_cc / tổng CỘT c; Recall = M_cc / tổng HÀNG c."""
    tp = np.diag(M).astype(float)
    p = tp / np.maximum(M.sum(axis=0), 1)
    r = tp / np.maximum(M.sum(axis=1), 1)
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    return p, r, f1


def log_loss(y, P, alpha_lam_tron=0.0, k=None, K=None):
    """-(1/n) sum ln P[i, y_i].

    Nếu alpha_lam_tron > 0 (cho k-NN, P là tỉ lệ phiếu v_c/k): dùng p~ = (v + alpha)/(k + K alpha)
    để không có xác suất 0 (mục 7.2). Nếu không, cắt dưới ở 1e-15 như scikit-learn.
    """
    if alpha_lam_tron > 0:
        P = (P * k + alpha_lam_tron) / (k + K * alpha_lam_tron)
    P = np.clip(P, 1e-15, 1.0)
    return float(-np.mean(np.log(P[np.arange(len(y)), y])))


# =============================================================================
# PHẦN 8. MỞ RỘNG: CÂY QUYẾT ĐỊNH (tìm điểm chia), HỖN HỢP GAUSS (EM) - mục 8
# =============================================================================

def gini(nhan, K):
    if len(nhan) == 0:
        return 0.0
    p = np.bincount(nhan, minlength=K) / len(nhan)
    return 1.0 - float(np.sum(p ** 2))


def cac_cach_chia_gini(X, y, K):
    """Duyệt mọi (đặc trưng j, ngưỡng t = trung điểm các giá trị liền kề); trả về danh sách
    (độ tạp có trọng số, j, t) đã sắp tăng dần. Điểm chia tốt nhất là phần tử đầu (mục 8.3)."""
    ket_qua = []
    for j in range(X.shape[1]):
        gia_tri = np.unique(X[:, j])
        for t in (gia_tri[:-1] + gia_tri[1:]) / 2:
            trai, phai = y[X[:, j] <= t], y[X[:, j] > t]
            do_tap = (len(trai) * gini(trai, K) + len(phai) * gini(phai, K)) / len(y)
            ket_qua.append((do_tap, j + 1, float(t)))
    return sorted(ket_qua)


def log_gauss_nhieu_chieu(X, mu, Sigma):
    """ln N(x; mu, Sigma) cho mọi hàng của X, tính qua phân rã Cholesky (ổn định hơn nghịch đảo)."""
    d = X.shape[1]
    L = np.linalg.cholesky(Sigma)
    # GIẢI THÍCH: Sigma = L L^T nên (x-mu) Sigma^-1 (x-mu)^T = ||L^-1 (x-mu)^T||^2, và
    # ln|Sigma| = 2 sum ln L_ii. Giải hệ tam giác thay vì nghịch đảo cả ma trận.
    z = np.linalg.solve(L, (X - mu).T)
    return -0.5 * (d * np.log(2 * pi) + 2 * np.sum(np.log(np.diag(L))) + np.sum(z ** 2, axis=0))


def gmm_em(X, J, so_vong=300, dung=1e-7, seed=0, reg=1e-6):
    """Hỗn hợp Gauss bằng EM. Bước M dùng đúng công thức của Bayes ngây thơ nhưng Y -> Gamma."""
    n, d = X.shape
    rng = np.random.default_rng(seed)
    mu = X[rng.choice(n, J, replace=False)]                   # khởi tạo tâm = J mẫu ngẫu nhiên
    Sigma = np.array([np.cov(X.T).reshape(d, d) + reg * np.eye(d)] * J)
    pi_j = np.full(J, 1.0 / J)
    ll_cu = -np.inf
    for it in range(so_vong):
        # --- Bước E: trách nhiệm Gamma (n x J) ---
        logp = np.column_stack([np.log(pi_j[j]) + log_gauss_nhieu_chieu(X, mu[j], Sigma[j])
                                for j in range(J)])
        ll = float(np.mean(logsumexp(logp)))                  # log-hợp lý trung bình
        Gamma = np.exp(logp - logsumexp(logp)[:, None])
        # --- Bước M: giống mục 5.4 với Y thay bằng Gamma ("one-hot mềm") ---
        t = Gamma.sum(axis=0)                                 # số mẫu "mềm" mỗi thành phần
        pi_j = t / n
        mu = (Gamma.T @ X) / t[:, None]                       # M = diag(t)^-1 Gamma^T X
        for j in range(J):
            Z = X - mu[j]
            Sigma[j] = (Gamma[:, j][:, None] * Z).T @ Z / t[j] + reg * np.eye(d)
        if abs(ll - ll_cu) < dung:
            break
        ll_cu = ll
    return dict(pi=pi_j, mu=mu, Sigma=Sigma, ll=ll, so_vong=it + 1)


def gmm_log_mat_do(X, mo_hinh):
    logp = np.column_stack([np.log(mo_hinh["pi"][j]) + log_gauss_nhieu_chieu(X, mo_hinh["mu"][j], mo_hinh["Sigma"][j])
                            for j in range(len(mo_hinh["pi"]))])
    return logsumexp(logp)


def bic_gmm(ll_tb, n, J, d):
    so_tham_so = (J - 1) + J * d + J * d * (d + 1) // 2
    return -2.0 * ll_tb * n + so_tham_so * np.log(n)


# =============================================================================
# CÁC BÀI DEMO (mỗi bài ứng với một mục của tài liệu)
# =============================================================================

def tieu_de(s):
    print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78)


def demo_1_tham_so_vs_phi_tham_so(seed):
    tieu_de("MỤC 1-2: Softmax (tham số) vs k-NN (phi tham số) trên ba vòng tròn")
    X, y = tao_ba_vong_tron(200, seed)
    tr, te = chia_phan_tang(y, 0.3, seed)
    cn = ChuanHoa().fit(X[tr])
    Ztr, Zte = cn.transform(X[tr]), cn.transform(X[te])
    sm = SoftmaxHoiQuy().fit(X[tr], mot_nong(y[tr]))
    print(f"Softmax trên (x1, x2): train {np.mean(sm.du_doan(X[tr]) == y[tr]):.3f}"
          f" | test {np.mean(sm.du_doan(X[te]) == y[te]):.3f}   (biên là đường thẳng -> thiên lệch cao)")
    for k in (1, 15):
        acc = np.mean(knn_du_doan(Zte, Ztr, y[tr], k) == y[te])
        print(f"k-NN k={k:<2}            : test {acc:.3f}")

    tieu_de("MỤC 2.4-2.5: chọn k bằng kiểm định chéo, và vai trò của chuẩn hóa (dữ liệu sinh viên)")
    X, y = tao_du_lieu_sinh_vien(200, seed)
    tr, te = chia_phan_tang(y, 0.3, seed)
    k_tot, cv = chon_k_kiem_dinh_cheo(X[tr], y[tr], [1, 3, 5, 9, 15, 25, 50])
    print("CV theo k:", {k: round(v, 3) for k, v in cv.items()}, "-> k tốt nhất =", k_tot)
    cn = ChuanHoa().fit(X[tr])
    acc_chuan = np.mean(knn_du_doan(cn.transform(X[te]), cn.transform(X[tr]), y[tr], k_tot) == y[te])
    X_phut = X.copy(); X_phut[:, 0] *= 60                     # đổi đơn vị giờ -> phút
    acc_phut_tho = np.mean(knn_du_doan(X_phut[te], X_phut[tr], y[tr], k_tot) == y[te])
    print(f"test với chuẩn hóa: {acc_chuan:.3f} | đổi 'giờ' thành 'phút' mà KHÔNG chuẩn hóa: {acc_phut_tho:.3f}")
    sm = SoftmaxHoiQuy().fit(X[tr], mot_nong(y[tr]))
    print(f"Softmax (dạng hàm đúng với đám Gauss): test {np.mean(sm.du_doan(X[te]) == y[te]):.3f}")

    print("\nVí dụ tính tay (mục 2.6): 9 sinh viên, 3 điểm truy vấn, k = 3")
    S9 = np.array([[2, 4, 0], [3, 3, 0], [4, 5, 0], [6, 3, 1], [7, 2, 1], [6, 4, 1],
                   [9, 1, 2], [10, 0, 2], [8, 0, 2]], float)
    X9, y9 = S9[:, :2], S9[:, 2].astype(int)
    X0 = np.array([[5, 3], [8, 2], [3, 2]], float)
    print("D^(2) =\n", binh_phuong_khoang_cach(X0, X9).astype(int))
    print("P_hat (bỏ phiếu đa số) =\n", np.round(knn_xac_suat(X0, X9, mot_nong(y9), 3), 3))
    print("P_hat (trọng số 1/d^2) =\n", np.round(knn_xac_suat(X0, X9, mot_nong(y9), 3, True), 3))


def demo_2_hoi_quy(seed):
    tieu_de("MỤC 3: k-NN hồi quy, Nadaraya-Watson, LOESS trên dữ liệu sin")
    x, y = tao_du_lieu_sin(100, 0.3, seed)
    xt = np.random.default_rng(seed + 1).uniform(0, 10, 400)
    yt = ham_that_sin(xt) + np.random.default_rng(seed + 2).normal(0, 0.3, 400)
    X, Xt = x[:, None], xt[:, None]
    rmse = lambda p: float(np.sqrt(np.mean((yt - p) ** 2)))
    print(f"Hồi quy tuyến tính  : RMSE kiểm tra {rmse(np.polyval(np.polyfit(x, y, 1), xt)):.3f}")
    print(f"Đa thức bậc 7       : RMSE kiểm tra {rmse(np.polyval(np.polyfit(x / 10, y, 7), xt / 10)):.3f}")
    print(f"(mức sàn do nhiễu sigma = 0,300)")

    hs = np.round(np.arange(0.1, 2.01, 0.05), 2)
    cv_nw = [loocv_bo_mot_ra(ma_tran_lam_muot(x, h, "nw"), y) for h in hs]
    cv_lo = [loocv_bo_mot_ra(ma_tran_lam_muot(x, h, "loess"), y) for h in hs]
    h_nw, h_lo = hs[int(np.argmin(cv_nw))], hs[int(np.argmin(cv_lo))]
    k_tot = chon_k_hoi_quy_bo_mot_ra(x, y, range(1, 31))
    print(f"k-NN hồi quy (k={k_tot})            : RMSE {rmse(knn_hoi_quy(Xt, X, y, k_tot)):.3f}")
    print(f"Nadaraya-Watson (h={h_nw})       : RMSE {rmse(nadaraya_watson(Xt, X, y, h_nw)):.3f}  (h chọn bằng bỏ-một-ra)")
    print(f"LOESS            (h={h_lo})       : RMSE {rmse(loess(Xt, X, y, h_lo)):.3f}")

    # kiểm chứng công thức bỏ-một-ra bằng cách bỏ thật từng mẫu
    h0 = 0.6
    thuc = np.mean([(y[i] - nadaraya_watson(X[i:i + 1], np.delete(X, i, 0), np.delete(y, i), h0)[0]) ** 2
                    for i in range(len(x))])
    print(f"Kiểm tra LOOCV (NW, h=0,6): công thức đóng {loocv_bo_mot_ra(ma_tran_lam_muot(x, h0), y):.6f}"
          f" | bỏ thật từng mẫu {thuc:.6f}  (phải trùng nhau)")

    print("\nVí dụ tính tay (mục 3.6): 6 sinh viên, điểm truy vấn x0 = 3,7 và x0 = 6 (biên)")
    xs = np.array([1, 2, 3, 4, 5, 6.]); ys = np.array([4.0, 5.2, 5.8, 7.1, 8.0, 9.1])
    Q = np.array([[3.7], [6.0]])
    print("k-NN (k=3)       :", np.round(knn_hoi_quy(Q, xs[:, None], ys, 3), 3))
    print("Nadaraya-Watson  :", np.round(nadaraya_watson(Q, xs[:, None], ys, 1.0), 3))
    print("LOESS            :", np.round(loess(Q, xs[:, None], ys, 1.0), 3), " (quan sát tại x=6: 9,1 -> LOESS ít lệch ở biên)")


def demo_3_mat_do(seed):
    tieu_de("MỤC 4: ước lượng mật độ trên mẫu hai đỉnh")
    x = lay_mau_hai_dinh(300, seed)
    xt = lay_mau_hai_dinh(2000, seed + 1)
    X, Xt = x[:, None], xt[:, None]
    luoi = np.linspace(-5, 6, 1101)
    dl = luoi[1] - luoi[0]
    ise = lambda f: float(np.sum((f - mat_do_that_hai_dinh(luoi)) ** 2) * dl)
    ll = lambda f_te: float(np.mean(np.log(np.maximum(f_te, 1e-6))))
    print(f"Mật độ thật: log-hợp lý kiểm tra = {np.mean(np.log(mat_do_that_hai_dinh(xt))):.3f}")
    mu, sd = x.mean(), x.std()
    gauss = lambda t: np.exp(-0.5 * ((t - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    print(f"Gauss hợp lý cực đại : ISE {ise(gauss(luoi)):.4f} | log-hợp lý {ll(gauss(xt)):.3f}")
    for delta in (0.1, 0.5, 2.0):
        print(f"Histogram Δ={delta:<3}      : ISE {ise(histogram_mat_do(luoi, x, delta)):.4f}"
              f" | ô rỗng trong tập kiểm tra: {(histogram_mat_do(xt, x, delta) == 0).sum()}/2000")
    h_s = silverman(x)
    hs = np.round(np.arange(0.05, 1.5, 0.01), 2)
    h_l = hs[int(np.argmax([log_kha_nang_bo_mot_ra(X, h) for h in hs]))]
    for ten, h in (("Silverman", h_s), ("hợp lý cực đại có KĐC", h_l)):
        print(f"KDE h={h:.3f} ({ten:<22}): ISE {ise(kde(luoi[:, None], X, h)):.4f} | log-hợp lý {ll(kde(Xt, X, h)):.3f}")
    for k in (3, 15, 60):
        f = knn_mat_do(luoi[:, None], X, k)
        print(f"k láng giềng k={k:<2}: ISE {ise(f):.4f} | tích phân trên [-200,200] = "
              f"{np.sum(knn_mat_do(np.linspace(-200, 200, 200001)[:, None], X, k)) * 0.002:.2f} (> 1: không phải mật độ hợp lệ)")

    print("\nVí dụ tính tay (mục 4.7): 8 sinh viên, điểm truy vấn 2,5 và 5")
    xs = np.array([1, 2, 2.5, 3, 7, 8, 8.5, 9.])
    Q = np.array([[2.5], [5.0]])
    print("Histogram (Δ=2) :", np.round(histogram_mat_do(Q[:, 0], xs, 2.0), 4))
    print("KDE (h=1)       :", np.round(kde(Q, xs[:, None], 1.0), 4))
    print("k láng giềng k=3:", np.round(knn_mat_do(Q, xs[:, None], 3), 4))
    print("Silverman       :", round(silverman(xs), 3), "| LCV (h=0,5; 1; 2):",
          [round(log_kha_nang_bo_mot_ra(xs[:, None], h), 3) for h in (0.5, 1.0, 2.0)])


def demo_4_bayes_ngay_tho(seed):
    tieu_de("MỤC 5: Bayes ngây thơ")
    X, y = tao_du_lieu_sinh_vien(200, seed)
    tr, te = chia_phan_tang(y, 0.3, seed)
    nb = GaussNB().fit(X[tr], mot_nong(y[tr]))
    P = nb.xac_suat(X[te])
    print(f"Bayes ngây thơ Gauss trên sinh viên: test {np.mean(P.argmax(1) == y[te]):.3f},"
          f" log-loss {log_loss(y[te], P):.3f}")
    # chống đặc trưng nhiễu so với k-NN (mục 5.7, 6.3)
    rng = np.random.default_rng(seed)
    print("Thêm q đặc trưng nhiễu thuần túy (độ chính xác kiểm tra):")
    for q in (0, 20, 100):
        XX = np.c_[X, rng.normal(size=(len(X), q))]
        cn = ChuanHoa().fit(XX[tr])
        a_nb = np.mean(GaussNB().fit(XX[tr], mot_nong(y[tr])).du_doan(XX[te]) == y[te])
        a_knn = np.mean(knn_du_doan(cn.transform(XX[te]), cn.transform(XX[tr]), y[tr], 18) == y[te])
        print(f"  q={q:<3}: Bayes ngây thơ {a_nb:.3f} | k-NN(k=18) {a_knn:.3f}")
    # đặc trưng bị sao chép -> xác suất quá tự tin (mục 5.6)
    print("Sao chép đặc trưng x1 thành m bản: Bayes ngây thơ ngày càng 'tự tin' (log-loss tăng):")
    for m in (1, 2, 5, 10):
        A = np.c_[X] if m == 1 else np.c_[X, np.repeat(X[:, :1], m - 1, axis=1)]
        Pm = GaussNB().fit(A[tr], mot_nong(y[tr])).xac_suat(A[te])
        print(f"  m={m:<2}: độ chính xác {np.mean(Pm.argmax(1) == y[te]):.3f},"
              f" độ tin cậy TB {Pm.max(1).mean():.3f}, log-loss {log_loss(y[te], Pm):.3f}")

    print("\nVí dụ tính tay (mục 5.7): Multinomial NB với làm trơn Laplace")
    # cột: [thưởng, miễn_phí, họp, bài]; lớp 1 = rác, lớp 0 = bình thường
    Xt = np.array([[2, 1, 0, 0], [1, 2, 0, 1], [0, 0, 1, 2], [1, 0, 1, 1], [0, 0, 2, 1.]])
    Yt = mot_nong(np.array([1, 1, 0, 0, 0]))
    x0 = np.array([[1, 1, 1, 0.]])
    for a in (1.0,):
        m = MultiNB(a).fit(Xt, Yt)
        print(f"alpha={a}: điểm số (bt, rác) = {np.round(m.diem_so(x0)[0], 3)} -> "
              f"P(bt, rác) = {np.round(m.xac_suat(x0)[0], 3)}")
    with np.errstate(divide="ignore"):
        m0 = MultiNB(0.0).fit(Xt, Yt)
        print("alpha=0 (không làm trơn): điểm số =", m0.diem_so(x0)[0], "-> cả hai lớp bị loại (-inf)")


def demo_5_thien_lech_phuong_sai(seed):
    tieu_de("MỤC 6.1: phân rã sai số = nhiễu + thiên lệch^2 + phương sai (công thức đóng)")
    sigma = 0.3
    x, _ = tao_du_lieu_sin(100, sigma, seed)
    luoi = np.linspace(0, 10, 201)
    print(f"{'mô hình':<22}{'thiên lệch^2':>14}{'phương sai':>12}{'tổng (+0,09)':>14}")
    for ten, kieu, ts in [("k-NN k=1", "knn", 1), ("k-NN k=6", "knn", 6), ("k-NN k=20", "knn", 20),
                          ("k-NN k=50", "knn", 50), ("NW h=0,05", "nw", 0.05), ("NW h=0,3", "nw", 0.3),
                          ("NW h=0,8", "nw", 0.8), ("NW h=2", "nw", 2.0)]:
        S = ma_tran_trong_so_tai_luoi(x, luoi, ts, kieu)
        b2, v = phan_ra_sai_so(S, ham_that_sin(x), ham_that_sin(luoi), sigma)
        print(f"{ten:<22}{b2:>14.4f}{v:>12.4f}{sigma ** 2 + b2 + v:>14.4f}")
    S6 = ma_tran_trong_so_tai_luoi(x, luoi, 6, "knn")
    print(f"Kiểm tra: phương sai k-NN k=6 = sigma^2/k = {sigma ** 2 / 6:.4f}  (công thức cho"
          f" {phan_ra_sai_so(S6, ham_that_sin(x), ham_that_sin(luoi), sigma)[1]:.4f})")

    tieu_de("MỤC 6.3: lời nguyền chiều - khoảng cách tập trung khi số chiều tăng")
    print(f"{'d':>6}{'(xa nhất - gần nhất) / gần nhất':>36}")
    for d in (2, 10, 100, 1000):
        print(f"{d:>6}{ty_le_tap_trung(d, 1000, seed):>36.3f}")
    print("Cạnh hình lập phương con chứa 1% dữ liệu đều trong [0,1]^d:",
          {d: round(0.01 ** (1 / d), 3) for d in (1, 2, 10, 50, 100)})

    tieu_de("MỤC 6.4: tính nhất quán - 1-NN kẹt ở sai số > Bayes, k ~ căn(n) tiến về Bayes")
    rng = np.random.default_rng(seed)
    def lay(n):
        y = rng.integers(0, 2, n); return rng.normal(2.0 * y, 1.0)[:, None], y
    Xt, yt = lay(4000)
    print("sai số Bayes = Phi(-1) = 0,1587; giới hạn lý thuyết 1-NN = 0,2248")
    for n in (10, 100, 1000, 3000):
        X, y = lay(n)
        e1 = np.mean(knn_du_doan(Xt, X, y, 1) != yt)
        ks = max(1, int(np.sqrt(n))) | 1
        eks = np.mean(knn_du_doan(Xt, X, y, ks) != yt)
        print(f"  n={n:<5}: 1-NN {e1:.3f} | k={ks:<3} {eks:.3f}")


def demo_6_danh_gia(seed):
    tieu_de("MỤC 7.2: ma trận nhầm lẫn, precision / recall / F1, log-loss có làm trơn")
    X, y = tao_du_lieu_sinh_vien(200, seed)
    tr, te = chia_phan_tang(y, 0.3, seed)
    cn = ChuanHoa().fit(X[tr])
    Ztr, Zte = cn.transform(X[tr]), cn.transform(X[te])
    k = 18
    P = knn_xac_suat(Zte, Ztr, mot_nong(y[tr]), k)
    yh = P.argmax(1)
    M = ma_tran_nham_lan(y[te], yh, 3)
    p, r, f1 = precision_recall_f1(M)
    print("Ma trận nhầm lẫn M = Y^T Y_hat (hàng: thật; cột: dự đoán)\n", M.astype(int))
    print("precision:", np.round(p, 3), "| recall:", np.round(r, 3), "| F1:", np.round(f1, 3))
    print(f"độ chính xác = tr(M)/n = {np.trace(M) / M.sum():.3f} | macro-F1 = {f1.mean():.3f}")
    print("log-loss của k-NN theo hệ số làm trơn alpha (alpha=0: cắt ở 1e-15, phụ thuộc sàn tùy ý):")
    for a in (0.0, 0.01, 0.1, 1.0):
        print(f"  alpha={a:<5}: {log_loss(y[te], P, a, k, 3):.3f}")
    print("Ví dụ lệch lớp: 1000 mẫu, 50 dương, mô hình luôn đoán 'âm' -> accuracy 0,95 nhưng recall 0, F1 0.")


def demo_7_gmm_va_cay(seed):
    tieu_de("MỤC 8.3: cây quyết định (điểm chia Gini) và hỗn hợp Gauss (EM)")
    S9 = np.array([[2, 4, 0], [3, 3, 0], [4, 5, 0], [6, 3, 1], [7, 2, 1], [6, 4, 1],
                   [9, 1, 2], [10, 0, 2], [8, 0, 2]], float)
    X9, y9 = S9[:, :2], S9[:, 2].astype(int)
    print(f"Độ tạp Gini ban đầu của 9 sinh viên = {gini(y9, 3):.3f}")
    print("Các cách chia tốt nhất (độ tạp có trọng số, đặc trưng, ngưỡng):")
    for do_tap, j, t in cac_cach_chia_gini(X9, y9, 3)[:4]:
        print(f"   {do_tap:.3f}  x{j} <= {t}")
    print("   (ba cách đầu HÒA nhau ở 0,333 -> cài đặt cụ thể chọn theo thứ tự duyệt)")

    x = lay_mau_hai_dinh(300, seed)[:, None]
    xt = lay_mau_hai_dinh(2000, seed + 1)[:, None]
    print("\nChọn số thành phần J bằng BIC (dữ liệu thật có 2 thành phần):")
    for J in range(1, 5):
        mh = gmm_em(x, J, seed=seed)
        print(f"  J={J}: BIC {bic_gmm(mh['ll'], len(x), J, 1):8.1f} | log-hợp lý kiểm tra {np.mean(gmm_log_mat_do(xt, mh)):.4f}"
              f" | số vòng EM {mh['so_vong']}")
    mh = gmm_em(x, 2, seed=seed)
    o = np.argsort(mh["mu"][:, 0])
    print("EM J=2:  pi =", np.round(mh["pi"][o], 3), " mu =", np.round(mh["mu"][o, 0], 3),
          " sigma =", np.round(np.sqrt(mh["Sigma"][o, 0, 0]), 3), "  (thật: pi=0,6/0,4, mu=-1,5/2, sigma=0,7/1)")
    print("KDE h=0,22 để so sánh: log-hợp lý kiểm tra =",
          round(float(np.mean(np.log(kde(xt, x, 0.22)))), 4))


def demo_8_chu_so(seed):
    tieu_de("MỤC 7.5-7.6: chữ số viết tay - cài đặt từ đầu so với scikit-learn")
    try:
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB, MultinomialNB
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        print("Chưa cài scikit-learn: bỏ qua phần này."); return
    X, y = load_digits(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    Ytr = mot_nong(ytr, 10)
    print(f"{len(Xtr)} ảnh huấn luyện, {len(Xte)} ảnh kiểm tra, d = {X.shape[1]}")
    t = time.perf_counter(); yh_knn = knn_du_doan(Xte, Xtr, ytr, 3); t_knn = time.perf_counter() - t
    sk_knn = KNeighborsClassifier(3).fit(Xtr, ytr).predict(Xte)
    gnb = GaussNB(lam=1e-2).fit(Xtr, Ytr); mnb = MultiNB(1.0).fit(Xtr, Ytr)
    print(f"{'mô hình':<34}{'độ chính xác':>14}")
    print(f"{'k-NN (k=3), cài từ đầu':<34}{np.mean(yh_knn == yte):>14.4f}   (trùng nhãn với sklearn: {np.mean(yh_knn == sk_knn):.3f}; {t_knn * 1000:.1f} ms)")
    print(f"{'Bayes ngây thơ Gauss (làm trơn .01)':<34}{np.mean(gnb.du_doan(Xte) == yte):>14.4f}")
    print(f"{'Bayes ngây thơ Multinomial':<34}{np.mean(mnb.du_doan(Xte) == yte):>14.4f}")
    sm = LogisticRegression(C=0.011, max_iter=3000).fit(Xtr, ytr)
    print(f"{'Softmax Regression (sklearn)':<34}{sm.score(Xte, yte):>14.4f}")
    sk_g = GaussianNB(var_smoothing=1e-9).fit(Xtr, ytr); g0 = GaussNB(lam=1e-9).fit(Xtr, Ytr)
    sk_m = MultinomialNB(alpha=1.0).fit(Xtr, ytr)
    print(f"Đối chiếu: GaussNB ta vs sklearn: nhãn trùng {np.mean(g0.du_doan(Xte) == sk_g.predict(Xte)):.3f},"
          f" |ΔP| tối đa {np.abs(g0.xac_suat(Xte) - sk_g.predict_proba(Xte)).max():.1e}")
    print(f"           MultiNB ta vs sklearn: nhãn trùng {np.mean(mnb.du_doan(Xte) == sk_m.predict(Xte)):.3f},"
          f" |ΔP| tối đa {np.abs(mnb.xac_suat(Xte) - sk_m.predict_proba(Xte)).max():.1e}")
    M = ma_tran_nham_lan(yte, yh_knn, 10)
    p, r, f1 = precision_recall_f1(M)
    print("k-NN: macro-F1 =", round(float(f1.mean()), 4), "| recall thấp nhất ở chữ số", int(np.argmin(r)), "=", round(float(r.min()), 3))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    print("Kiểm định chéo 5 phần theo k (k-NN):",
          {k: round(float(cross_val_score(KNeighborsClassifier(k), Xtr, ytr, cv=cv).mean()), 4) for k in (1, 3, 5, 10, 20)})


# =============================================================================
# VẼ HÌNH (tùy chọn --ve)
# =============================================================================

def ve_hinh(thu_muc, seed):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Chưa cài matplotlib: bỏ qua việc vẽ hình."); return
    os.makedirs(thu_muc, exist_ok=True)
    # 1) vùng quyết định: Softmax vs k-NN trên ba vòng tròn
    X, y = tao_ba_vong_tron(200, seed)
    cn = ChuanHoa().fit(X); Z = cn.transform(X)
    gx, gy = np.meshgrid(np.linspace(-6.5, 6.5, 250), np.linspace(-6.5, 6.5, 250))
    L = cn.transform(np.c_[gx.ravel(), gy.ravel()])
    sm = SoftmaxHoiQuy().fit(X, mot_nong(y))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
    for a, ten, Zg in [(ax[0], "Softmax (tham số)", sm.du_doan(np.c_[gx.ravel(), gy.ravel()])),
                       (ax[1], "k-NN, k=15 (phi tham số)", knn_du_doan(L, Z, y, 15))]:
        a.contourf(gx, gy, Zg.reshape(gx.shape), levels=[-.5, .5, 1.5, 2.5], colors=["#fbd9bf", "#bfe8dc", "#d9d6ec"])
        a.scatter(X[:, 0], X[:, 1], c=y, cmap="Dark2", s=6); a.set_title(ten); a.set_aspect("equal")
    fig.tight_layout(); fig.savefig(os.path.join(thu_muc, "01_vung_quyet_dinh.png"), dpi=130); plt.close(fig)
    # 2) hồi quy
    x, yy = tao_du_lieu_sin(100, 0.3, seed); g = np.linspace(0, 10, 300)
    fig, a = plt.subplots(figsize=(7, 4))
    a.scatter(x, yy, s=8, c="#999"); a.plot(g, ham_that_sin(g), "--k", label="hàm thật")
    a.plot(g, nadaraya_watson(g[:, None], x[:, None], yy, 0.3), label="Nadaraya-Watson h=0,3")
    a.plot(g, loess(g[:, None], x[:, None], yy, 0.35), label="LOESS h=0,35"); a.legend()
    fig.tight_layout(); fig.savefig(os.path.join(thu_muc, "02_hoi_quy.png"), dpi=130); plt.close(fig)
    # 3) mật độ
    xs = lay_mau_hai_dinh(300, seed); g = np.linspace(-5, 6, 400)
    fig, a = plt.subplots(figsize=(7, 4))
    a.plot(g, mat_do_that_hai_dinh(g), "--k", label="mật độ thật")
    for h in (0.05, 0.22, 1.5):
        a.plot(g, kde(g[:, None], xs[:, None], h), label=f"KDE h={h}")
    a.legend(); fig.tight_layout(); fig.savefig(os.path.join(thu_muc, "03_kde.png"), dpi=130); plt.close(fig)
    print("Đã lưu 3 hình vào", thu_muc)


# =============================================================================
# CHƯƠNG TRÌNH CHÍNH
# =============================================================================

CAC_PHAN = {
    "1": demo_1_tham_so_vs_phi_tham_so, "2": demo_2_hoi_quy, "3": demo_3_mat_do,
    "4": demo_4_bayes_ngay_tho, "5": demo_5_thien_lech_phuong_sai, "6": demo_6_danh_gia,
    "7": demo_7_gmm_va_cay, "8": demo_8_chu_so,
}


def main():
    ap = argparse.ArgumentParser(description="Minh họa các mô hình phi tham số")
    ap.add_argument("--phan", nargs="*", default=["all"], help="các phần cần chạy: 1..8 hoặc all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ve", action="store_true", help="vẽ hình vào thư mục hinh_demo/")
    args = ap.parse_args()
    chon = list(CAC_PHAN) if "all" in args.phan else args.phan
    t0 = time.perf_counter()
    for ma in chon:
        CAC_PHAN[ma](args.seed)
    if args.ve:
        ve_hinh("hinh_demo", args.seed)
    print(f"\nHoàn thành sau {time.perf_counter() - t0:.1f} giây.")


if __name__ == "__main__":
    main()
