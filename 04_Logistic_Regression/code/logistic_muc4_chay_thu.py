import numpy as np
from logistic import *
np.set_printoptions(precision=4, suppress=True)

# dữ liệu 400 sinh viên (nhãn 0/1), chia train/test như mục 3.7
rng = np.random.default_rng(42)
n_lop = 200
dau = np.c_[rng.normal(8, 2.2, n_lop), rng.normal(1.5, 1.2, n_lop)]
rot = np.c_[rng.normal(4, 2.2, n_lop), rng.normal(3.5, 1.5, n_lop)]
X = np.clip(np.vstack([dau, rot]), 0, None)
y = np.r_[np.ones(n_lop), np.zeros(n_lop)].astype(int)
X_tr, X_te, y_tr, y_te = chia_train_test(X, y, ti_le_test=0.3, seed=0)

# --- mục 4.2: chuẩn hóa đặc trưng ---
Z_tr, Z_te, mu, s = chuan_hoa(X_tr, X_te)
print("mu =", mu, "| s =", s)
Zb_tr = them_bias(Z_tr)
w_z, ls = huan_luyen(Zb_tr, y_tr, eta=5.0, so_epoch=300)
print("w' (chuan hoa) =", w_z, "| J =", round(ls[-1], 6))
w_goc = np.r_[w_z[0] - np.sum(w_z[1:] * mu / s), w_z[1:] / s]
print("w (thang goc)  =", w_goc)

# --- mục 4.4: điều chuẩn hóa L2 trên dữ liệu phân tách hoàn hảo ---
X_sv = np.array([[2, 4], [3, 3], [4, 5], [6, 1], [8, 2], [9, 0]], dtype=float)
y_sv = np.array([0, 0, 0, 1, 1, 1])
Xb_sv = them_bias(X_sv)
for lam in (0.0, 0.001, 0.01, 0.1):
    w_l, ls = huan_luyen(Xb_sv, y_sv, eta=0.5, so_epoch=200000, lam=lam)
    p_B = sigmoid(np.array([1, 5, 2]) @ w_l)
    print(f"lam = {lam:<6} w = {w_l} | |w~| = {np.linalg.norm(w_l[1:]):.3f} | p(sv B) = {p_B:.3f}")
