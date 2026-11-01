import numpy as np
from perceptron import *
np.set_printoptions(precision=3, suppress=True)
print("=== 5.1 ===")
X_sv = np.array([[2, 4], [3, 3], [4, 5], [6, 1], [8, 2], [9, 0]], dtype=float)
y_sv = np.array([-1, -1, -1, 1, 1, 1])
Xb_sv = them_bias(X_sv)
w, lich_su, quy_dao = huan_luyen(Xb_sv, y_sv)
print("w =", w)
print("So cap nhat moi epoch:", lich_su)
print("y_hat =", du_doan(Xb_sv, w))
print("margin m =", y_sv * (Xb_sv @ w))
print("quy dao:", [list(q) for q in quy_dao])
print("=== 5.2 ===")
rng = np.random.default_rng(42)
n_lop = 200
dau = np.c_[rng.normal(8, 2.2, n_lop), rng.normal(1.5, 1.2, n_lop)]
rot = np.c_[rng.normal(4, 2.2, n_lop), rng.normal(3.5, 1.5, n_lop)]
X = np.clip(np.vstack([dau, rot]), 0, None)
y = np.r_[np.ones(n_lop), -np.ones(n_lop)].astype(int)
X_tr, X_te, y_tr, y_te = chia_train_test(X, y, ti_le_test=0.3, seed=0)
Xb_tr, Xb_te = them_bias(X_tr), them_bias(X_te)
print(X_tr.shape, X_te.shape, "so mau dau (train/test):", (y_tr == 1).sum(), (y_te == 1).sum())
w_1, lich_su_1, _ = huan_luyen(Xb_tr, y_tr, t_max=100)
print("so epoch da chay:", len(lich_su_1), "| so cap nhat 5 epoch cuoi:", lich_su_1[-5:])
w_cuoi, w_tui, lich_su_tui = huan_luyen_tui(Xb_tr, y_tr, t_max=100)
print("w_cuoi =", w_cuoi, "| w_tui =", w_tui)
print("loi tren train: w_cuoi =", lich_su_tui[-1][0], "| w_tui =", lich_su_tui[-1][1], "/", len(y_tr))
acc = lambda a, b: np.mean(a == b)
for ten, ww in [("w (khong xao tron)", w_1), ("w_cuoi (xao tron)", w_cuoi), ("w_tui", w_tui)]:
    print(f"{ten:20s} train: {acc(y_tr, du_doan(Xb_tr, ww)):.4f}  test: {acc(y_te, du_doan(Xb_te, ww)):.4f}")
print("=== 5.3 ===")
y_hat_te = du_doan(Xb_te, w_tui)
C = ma_tran_nham_lan(y_te, y_hat_te)
print(C)
tn, fp, fn, tp = C.ravel()
print("accuracy =", (tp + tn) / C.sum(), "precision =", tp / (tp + fp), "recall =", tp / (tp + fn), "F1 =", 2 * tp / (2 * tp + fp + fn))
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
clf = Perceptron(eta0=1.0, max_iter=20, tol=None, shuffle=False)
clf.fit(X_sv, y_sv)
print("sklearn 6 SV: intercept_ =", clf.intercept_, "coef_ =", clf.coef_)
clf = Perceptron(eta0=1.0, max_iter=100, tol=None, shuffle=False)
clf.fit(X_tr, y_tr)
print("sklearn (khop code): intercept_ =", clf.intercept_, "coef_ =", clf.coef_)
print("code tu cai dat     : w =", w_1)
clf_mac_dinh = Perceptron(random_state=0).fit(X_tr, y_tr)
print("sklearn mac dinh: n_iter_ =", clf_mac_dinh.n_iter_)
hq = LinearRegression().fit(X_tr, y_tr)
du_doan_hq = lambda X_: np.where(hq.predict(X_) >= 0, 1, -1)
rows = [("Perceptron tu cai dat (w_tui)", du_doan(Xb_tr, w_tui), du_doan(Xb_te, w_tui)),
        ("sklearn Perceptron (mac dinh)", clf_mac_dinh.predict(X_tr), clf_mac_dinh.predict(X_te)),
        ("Hoi quy tuyen tinh + nguong 0", du_doan_hq(X_tr), du_doan_hq(X_te))]
for ten, a, b in rows: print(f"{ten:32s} train {accuracy_score(y_tr, a):.4f}  test {accuracy_score(y_te, b):.4f}")
print(confusion_matrix(y_te, y_hat_te, labels=[-1, 1]))
print(classification_report(y_te, y_hat_te, labels=[-1, 1], target_names=["rot", "dau"], digits=3))
np.savez('res.npz',X_tr=X_tr,X_te=X_te,y_tr=y_tr,y_te=y_te,w_tui=w_tui,w_cuoi=w_cuoi,w_1=w_1,lich=np.array(lich_su_tui),C=C)


import matplotlib.pyplot as plt

def ve_duong_bien(ax, X, y, w, x1_max=10, x2_max=6):
    """Vẽ dữ liệu, đường biên w0 + w1*x1 + w2*x2 = 0 và hai vùng dự đoán."""
    u, v = np.meshgrid(np.linspace(0, x1_max, 200), np.linspace(-1, x2_max, 200))
    s = w[0] + w[1] * u + w[2] * v
    ax.contourf(u, v, np.where(s >= 0, 1, -1),
                levels=[-2, 0, 2], colors=["tab:red", "tab:blue"], alpha=0.08)
    ax.contour(u, v, s, levels=[0], colors="k")           # đường biên s = 0
    ax.scatter(*X[y == -1].T, c="tab:red", label="Rớt")
    ax.scatter(*X[y == 1].T, c="tab:blue", marker="s", label="Đậu")
    ax.set_xlim(0, x1_max)
    ax.set_ylim(-1, x2_max)

fig, axes = plt.subplots(1, 5, figsize=(18, 3.6))
for ax, w_k in zip(axes, quy_dao):
    ve_duong_bien(ax, X_sv, y_sv, w_k)
plt.show()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6))
ve_duong_bien(ax1, X_tr, y_tr, w_tui, x1_max=14, x2_max=8)
ve_duong_bien(ax2, X_te, y_te, w_tui, x1_max=14, x2_max=8)
plt.show()