import numpy as np
from logistic import *
np.set_printoptions(precision=4, suppress=True)

# dữ liệu của bài Perceptron (mục 5.2), nhãn đổi sang 0/1
rng = np.random.default_rng(42)
n_lop = 200
dau = np.c_[rng.normal(8, 2.2, n_lop), rng.normal(1.5, 1.2, n_lop)]   # lớp 1 (đậu)
rot = np.c_[rng.normal(4, 2.2, n_lop), rng.normal(3.5, 1.5, n_lop)]   # lớp 0 (rớt)
X = np.clip(np.vstack([dau, rot]), 0, None)
y = np.r_[np.ones(n_lop), np.zeros(n_lop)].astype(int)
X_tr, X_te, y_tr, y_te = chia_train_test(X, y, ti_le_test=0.3, seed=0)
Xb_tr, Xb_te = them_bias(X_tr), them_bias(X_te)
print(Xb_tr.shape, Xb_te.shape)

# 1) kiểm tra gradient giải tích bằng đạo hàm số
w = np.array([-1.0, 0.3, -0.2]); eps = 1e-6
g_so = np.array([(mat_mat(Xb_tr, y_tr, w + eps * e) - mat_mat(Xb_tr, y_tr, w - eps * e)) / (2 * eps)
                 for e in np.eye(3)])
print("gradient giai tich:", gradient(Xb_tr, y_tr, w))
print("gradient so       :", g_so)

# 2) huấn luyện bằng ba cách
cau_hinh = {"batch (b = n = 280)":  dict(eta=0.2,  kich_thuoc_batch=None),
            "mini-batch (b = 32)":  dict(eta=0.1,  kich_thuoc_batch=32),
            "SGD (b = 1)":          dict(eta=0.01, kich_thuoc_batch=1)}
ket_qua = {}
for ten, cfg in cau_hinh.items():
    w_k, ls = huan_luyen(Xb_tr, y_tr, so_epoch=200, **cfg)
    ket_qua[ten] = (w_k, ls)
    acc_tr = np.mean(du_doan(Xb_tr, w_k) == y_tr); acc_te = np.mean(du_doan(Xb_te, w_k) == y_te)
    print(f"{ten:22s} J = {ls[-1]:.4f} | w = {w_k} | train {acc_tr:.4f} | test {acc_te:.4f}")
print("J ban dau =", round(ket_qua["batch (b = n = 280)"][1][0], 4), "= ln 2 =", round(np.log(2), 4))
