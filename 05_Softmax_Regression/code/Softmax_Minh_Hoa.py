#!/usr/bin/env python
# coding: utf-8

# # MÔ HÌNH SOFTMAX REGRESSION — Notebook minh họa
# 
# Notebook minh họa bằng mã Python toàn bộ nội dung tài liệu *Mô hình Softmax Regression* (phân loại **nhiều lớp** bằng xác suất), theo mạch **từ Logistic Regression → mô hình → huấn luyện → cơ chế học → đánh giá → mở rộng**:
# 
# | Mục | Nội dung | Phần minh họa |
# | --- | --- | --- |
# | 1 | Từ Logistic đến nhiều lớp | Hồi quy trên nhãn số phụ thuộc cách đánh số, one-hot, one-vs-rest và 3 nhược điểm của nó |
# | 2 | Mô hình Softmax | $\mathbf S=\mathbf{XW}$, $\mathbf P=\mathrm{softmax}(\mathbf S)$, bất biến khi dịch, $K=2$ ⇒ sigmoid, vùng quyết định, tham số dư thừa |
# | 3 | Huấn luyện | Cross-entropy nhiều lớp, gradient $\frac1n\mathbf X^T(\mathbf P-\mathbf Y)$, ví dụ tính tay, tốc độ học, Batch/SGD/Mini-batch |
# | 4 | Cơ chế học | Kéo–đẩy các cột của $\mathbf W$, Hessian **suy biến**, dữ liệu phân tách hoàn hảo + L2, ba vòng tròn đồng tâm |
# | 5 | Đánh giá | Ma trận nhầm lẫn $K\times K$, F1 theo lớp, ma trận chi phí, log-loss, ECE, AUC OvR, scikit-learn, kiểm định chéo |
# | 6 | Tổng kết và mở rộng | Odds ratio, IIA, so sánh OvR/Softmax, nhiệt độ, **mạng nơ-ron** và lan truyền ngược, bài tập |
# 
# **Quy ước ký hiệu:** $\mathbf x_i$ là vector hàng (phần tử đầu bằng 1); $\mathbf W$ là ma trận $(d{+}1)\times K$ (mỗi **cột** là tham số của một lớp); nhãn $y_i\in\{0,\dots,K-1\}$; $\mathbf Y$ là ma trận **one-hot** $n\times K$; $\mathbf S=\mathbf{XW}$ là ma trận điểm số; $\mathbf P=\mathrm{softmax}(\mathbf S)$ theo từng hàng.
# 
# > **Softmax Regression = $K$ hồi quy tuyến tính chạy song song + hàm softmax.**
# 
# Những đoạn mã khó có khung **🔍 Giải thích mã** ngay bên dưới.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
plt.rcParams.update({
    "figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})

TEN_LOP = ["rớt", "đạt", "giỏi"]
MAU_LOP = ["tab:red", "tab:blue", "tab:green"]


# ## Các hàm nền tảng (dùng xuyên suốt)
# 
# Để các mục sau dùng được ngay, ta khai báo trước các hàm cơ bản — chúng chính là các hàm của mục 3.7 trong tài liệu.

# In[ ]:


def them_bias(X):
    # Thêm cột 1 vào đầu ma trận đặc trưng: (n, d) -> (n, d+1).
    return np.hstack([np.ones((X.shape[0], 1)), X])

def one_hot(y, K):
    # Vector nhãn (n,) -> ma trận one-hot Y (n, K).
    return np.eye(K)[y]

def softmax(S):
    # Softmax theo từng hàng; trừ giá trị lớn nhất của hàng để không tràn số.
    S = S - S.max(axis=1, keepdims=True)
    E = np.exp(S)
    return E / E.sum(axis=1, keepdims=True)

def mat_mat(Xb, Y, W):
    # Cross-entropy trung bình J(W), tính qua log-sum-exp: l_i = -s_ic + ln sum_j e^{s_ij}.
    S = Xb @ W
    m = S.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(S - m).sum(axis=1))
    return np.mean(lse - np.sum(Y * S, axis=1))

def gradient(Xb, Y, W):
    # Gradient của J theo W: (1/n) * X^T (P - Y), kích thước (d+1, K).
    P = softmax(Xb @ W)
    return Xb.T @ (P - Y) / len(Xb)

def du_doan_xac_suat(Xb, W):
    return softmax(Xb @ W)

def du_doan(Xb, W):
    return du_doan_xac_suat(Xb, W).argmax(axis=1)

def huan_luyen(Xb, Y, eta=0.5, so_epoch=1000, kich_thuoc_batch=None, seed=0):
    # Gradient Descent: toàn bộ (batch), từng mẫu (SGD) hoặc mini-batch.
    n, d1 = Xb.shape
    b = n if kich_thuoc_batch is None else kich_thuoc_batch
    rng = np.random.default_rng(seed)
    W = np.zeros((d1, Y.shape[1]))                     # khởi tạo W = 0
    lich_su = [mat_mat(Xb, Y, W)]                      # J sau mỗi epoch
    for epoch in range(so_epoch):
        idx = rng.permutation(n) if b < n else np.arange(n)
        for k in range(0, n, b):
            lo = idx[k:k + b]                          # chỉ số của mini-batch hiện tại
            W = W - eta * gradient(Xb[lo], Y[lo], W)
        lich_su.append(mat_mat(Xb, Y, W))
    return W, lich_su

def sigmoid(s):
    return 0.5 * (1.0 + np.tanh(0.5 * s))              # sigmoid ổn định số học


# > ### 🔍 Giải thích mã
# > **`np.eye(K)[y]` (one-hot)** — `np.eye(K)` là ma trận đơn vị $K\times K$; mỗi *hàng* $k$ của nó chính là vector one-hot của lớp $k$. Viết `np.eye(K)[y]` với `y` là mảng nguyên là *lập chỉ mục nâng cao (fancy indexing)*: với mỗi $y_i$ ta lấy hàng thứ $y_i$ → ma trận $n\times K$. Ví dụ `np.eye(3)[[0, 2, 1]]` cho 3 hàng $[1,0,0],[0,0,1],[0,1,0]$.
# >
# > **`S.max(axis=1, keepdims=True)`** — tính giá trị lớn nhất **theo từng hàng** (`axis=1`). `keepdims=True` giữ kết quả dạng cột $n\times1$ thay vì vector $n$ để phép trừ `S - ...` *broadcasting* đúng: mỗi hàng bị trừ đúng giá trị lớn nhất của chính nó. Vì softmax **bất biến khi dịch** nên kết quả không đổi, nhưng số mũ lớn nhất luôn bằng 0 nên không bao giờ tràn.
# >
# > **`E / E.sum(axis=1, keepdims=True)`** — chia mỗi hàng của $\mathbf E$ cho tổng của hàng đó ⇒ mỗi hàng cộng lại bằng 1.
# >
# > **`mat_mat` dạng log-sum-exp** — Công thức $l_i=-s_{ic}+\ln\sum_je^{s_{ij}}$. `np.sum(Y * S, axis=1)` là phép nhân từng phần tử rồi cộng theo hàng: vì $\mathbf Y$ one-hot nên nó *chọn đúng* $s_{ic}$ (điểm số của lớp thật). `lse` được tính ổn định bằng cách trừ `m` (giá trị lớn nhất) trước khi lấy mũ, rồi cộng `m` lại. Không bao giờ phải tính `log(0)` nên *không cần `np.clip`* như ở Logistic Regression.
# >
# > **`Xb.T @ (P - Y) / len(Xb)`** — chính là $\frac1n\mathbf X^T(\mathbf P-\mathbf Y)$: $(d{+}1)\times n$ nhân $n\times K$ ra $(d{+}1)\times K$, **cùng kích thước với `W`**. Một phép nhân ma trận cho gradient của toàn bộ $K$ cột.
# >
# > **Vòng lặp mini-batch** — `rng.permutation(n)` xáo trộn mỗi epoch; `range(0, n, b)` nhảy từng bước $b$; `idx[k:k+b]` cắt ra các chỉ số của mini-batch hiện tại (batch cuối có thể ngắn hơn). `b = n` cho batch, `b = 1` cho SGD.

# ---
# # 1. Từ Logistic Regression đến bài toán phân loại nhiều lớp
# 
# ## 1.1. Hồi quy trên nhãn số: kết quả phụ thuộc cách đánh số lớp
# 
# Ba cụm dữ liệu nằm dọc trục $x_1$: trái ($x_1\approx-4$), giữa ($\approx0$), phải ($\approx4$), mỗi cụm 150 điểm. Thử hai cách đặt số: **A** (trái=0, giữa=1, phải=2) và **B** (trái=0, giữa=2, phải=1), hồi quy tuyến tính rồi làm tròn. *(Tài liệu không công bố dữ liệu nên notebook mô phỏng; con số có thể lệch nhẹ nhưng hiện tượng giống hệt.)*

# In[ ]:


def ba_cum(seed=0, n=150):
    rng = np.random.default_rng(seed)
    X = np.vstack([np.c_[rng.normal(c, 1.0, n), rng.normal(0, 1.5, n)] for c in (-4, 0, 4)])
    return X, np.repeat([0, 1, 2], n)                   # chỉ số cụm: trái, giữa, phải

def hoi_quy_lam_tron(X, so_cho_cum):
    # Hồi quy tuyến tính (phương trình chuẩn) trên nhãn là SỐ, rồi làm tròn và kẹp về [0, 2].
    Xb = them_bias(X)
    muc_tieu = np.array(so_cho_cum, dtype=float)[ba_cum_nhan]
    w = np.linalg.solve(Xb.T @ Xb, Xb.T @ muc_tieu)
    du_doan_so = Xb @ w
    return du_doan_so, np.clip(np.rint(du_doan_so), 0, 2).astype(int), muc_tieu

X3, ba_cum_nhan = ba_cum()
ket_qua_ab = {}
print(f"{'Cách đặt nhãn':<28}{'Trái':>6}{'Giữa':>6}{'Phải':>6}{'Độ chính xác':>15}   Độ chính xác từng cụm (trái, giữa, phải)")
for ten, so in [("A: trái=0, giữa=1, phải=2", (0, 1, 2)), ("B: trái=0, giữa=2, phải=1", (0, 2, 1))]:
    du_so, du_nguyen, muc_tieu = hoi_quy_lam_tron(X3, so)
    dung = du_nguyen == muc_tieu
    ket_qua_ab[ten] = (du_so, muc_tieu)
    print(f"{ten:<28}{so[0]:>6}{so[1]:>6}{so[2]:>6}{dung.mean():>15.1%}   " + ", ".join(f"{dung[ba_cum_nhan == c].mean():.0%}" for c in range(3)))
print("(tài liệu: A = 96,7% ; B = 34,7% với cụm giữa sai 100%)")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for a, (ten, (du_so, muc_tieu)) in zip(ax, ket_qua_ab.items()):
    for c in range(3):
        m = ba_cum_nhan == c
        a.scatter(X3[m, 0], muc_tieu[m], s=10, alpha=0.4, color=MAU_LOP[c])
    idx = np.argsort(X3[:, 0]); a.plot(X3[idx, 0], du_so[idx], "k", lw=2, label="hồi quy tuyến tính")
    a.set(title=ten, xlabel="x1", ylabel="số gán cho lớp"); a.legend()
plt.suptitle("Hình 1: cùng một dữ liệu, chỉ đổi cách đặt số cho lớp", y=1.02); plt.tight_layout(); plt.show()


# > ### 🔍 Giải thích mã
# > **`muc_tieu = np.array(so_cho_cum)[ba_cum_nhan]`** — `so_cho_cum` là bảng tra "cụm → số" (ví dụ `(0, 2, 1)`); lập chỉ mục bằng mảng `ba_cum_nhan` (mỗi phần tử là 0/1/2 cho biết điểm thuộc cụm nào) cho ra số được gán cho *từng điểm*. Đây là cách đổi tên nhãn chỉ bằng một dòng.
# >
# > **`np.rint(...)` rồi `np.clip(..., 0, 2)`** — làm tròn đến số nguyên gần nhất, rồi ép về các nhãn hợp lệ $\{0,1,2\}$ (hồi quy có thể dự đoán $-0{,}3$ hay $2{,}4$ mà không phải nhãn nào).
# >
# > **Kết quả:** ở cách B không có hàm tuyến tính nào đi qua $0$ (trái), $2$ (giữa), $1$ (phải) nên cụm giữa sai gần như hoàn toàn. Hồi quy *ngầm áp đặt thứ tự và khoảng cách đều* giữa các lớp.

# In[ ]:


# Softmax Regression trên cùng dữ liệu: KHÔNG phụ thuộc cách đặt số cho lớp
Z3 = (X3 - X3.mean(0)) / X3.std(0)
Zb3 = them_bias(Z3)
for ten, so in [("A", (0, 1, 2)), ("B", (0, 2, 1))]:
    nhan = np.array(so)[ba_cum_nhan]
    W_, _ = huan_luyen(Zb3, one_hot(nhan, 3), eta=0.5, so_epoch=1500)
    print(f"Softmax, cách đặt {ten}: độ chính xác = {np.mean(du_doan(Zb3, W_) == nhan):.1%}   (tài liệu: 96,9% cho cả hai)")


# ## 1.2. One-hot, dữ liệu 9 sinh viên và bộ dữ liệu 600 sinh viên
# 
# Mã hóa **one-hot** biểu diễn nhãn bằng vector $K$ phần tử (chỉ lớp đúng bằng 1): các lớp bình đẳng, mỗi hàng của $\mathbf Y$ là một phân phối xác suất ("xác suất mục tiêu"). Với $K=2$, cột thứ hai của $\mathbf Y$ trùng nhãn $\{0,1\}$ của bài Logistic.

# In[ ]:


print("one_hot([2,0,3,1], K=4) =\n", one_hot(np.array([2, 0, 3, 1]), 4).astype(int),
      "\nTổng mỗi hàng:", one_hot(np.array([2, 0, 3, 1]), 4).sum(axis=1), "| tổng mỗi cột:", one_hot(np.array([2, 0, 3, 1]), 4).sum(axis=0))
print("\nK=2: one_hot([0,1,1,0], 2) =\n", one_hot(np.array([0, 1, 1, 0]), 2).astype(int), "-> cột thứ hai trùng nhãn nhị phân")

# ---- 9 sinh viên (tính tay) ----
X_sv = np.array([[2, 4], [3, 3], [4, 5], [6, 3], [7, 2], [6, 4], [9, 1], [10, 0], [8, 0]], dtype=float)
y_sv = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
Xb_sv, Y_sv = them_bias(X_sv), one_hot(y_sv, 3)
print("\nXb_sv:\n", Xb_sv.astype(int).T, " (mỗi cột là một sinh viên)")
print("Y_sv:\n", Y_sv.astype(int).T, " (mỗi cột là một sinh viên)")


# In[ ]:


# ---- 600 sinh viên mô phỏng (đúng mã của tài liệu, mục 3.7) ----
def sinh_du_lieu(seed=42, n_lop=200):
    rng = np.random.default_rng(seed)
    def cum(m1, m2, s1, s2):
        return np.c_[rng.normal(m1, s1, n_lop), rng.normal(m2, s2, n_lop)]
    X = np.vstack([cum(3.0, 3.75, 1.55, 1.55),      # lớp 0: rớt
                   cum(6.25, 2.0, 1.55, 1.395),     # lớp 1: đạt
                   cum(9.75, 1.3, 1.55, 1.24)])     # lớp 2: giỏi
    return np.clip(X, 0, None), np.repeat([0, 1, 2], n_lop)

def chia_train_test(X, y, ti_le_test=0.3, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = int(len(y) * ti_le_test)
    return X[idx[n_test:]], X[idx[:n_test]], y[idx[n_test:]], y[idx[:n_test]]

def chuan_hoa(X_tr, X_te):
    mu, s = X_tr.mean(axis=0), X_tr.std(axis=0)     # chỉ tính từ tập huấn luyện
    return (X_tr - mu) / s, (X_te - mu) / s, mu, s

X, y = sinh_du_lieu()
X_tr, X_te, y_tr, y_te = chia_train_test(X, y)
Z_tr, Z_te, mu, s = chuan_hoa(X_tr, X_te)
Zb_tr, Zb_te = them_bias(Z_tr), them_bias(Z_te)
Y_tr, Y_te = one_hot(y_tr, 3), one_hot(y_te, 3)
print(Zb_tr.shape, Zb_te.shape, "| mu =", mu, "| s =", s, "  (tài liệu: [6,2951 2,3943], [3,1081 1,685])")
print("Số mẫu mỗi lớp trong tập huấn luyện:", np.bincount(y_tr), "| tập kiểm tra:", np.bincount(y_te))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for k in range(3):
    ax[0].scatter(*X_sv[y_sv == k].T, s=70, color=MAU_LOP[k], edgecolor="k", label=TEN_LOP[k])
    ax[1].scatter(*X[y == k].T, s=10, alpha=0.5, color=MAU_LOP[k], label=TEN_LOP[k])
ax[0].set(title="(a) 9 sinh viên (tính tay)", xlabel="x1 (giờ học)", ylabel="x2 (buổi vắng)"); ax[0].legend()
ax[1].set(title="(b) 600 sinh viên mô phỏng: lớp 'đạt' nằm GIỮA", xlabel="x1 (giờ học)"); ax[1].legend()
plt.tight_layout(); plt.show()


# ## 1.3. One-vs-rest (OvR) và ba nhược điểm của nó
# 
# OvR huấn luyện $K$ bộ phân loại Logistic **độc lập**: bộ thứ $k$ trả lời "lớp $k$ hay không?" với nhãn nhị phân là *cột $k$ của $\mathbf Y$*. Ta huấn luyện ba bộ trên đặc trưng chuẩn hóa ($\eta=0{,}5$, 2000 epoch):

# In[ ]:


def huan_luyen_logistic(Xb, t, eta=0.5, so_epoch=2000):
    w = np.zeros(Xb.shape[1])
    for _ in range(so_epoch):
        w = w - eta * Xb.T @ (sigmoid(Xb @ w) - t) / len(t)
    return w

# Mỗi cột của W_ovr là một bộ phân loại Logistic độc lập: nhãn là cột k của Y
W_ovr = np.column_stack([huan_luyen_logistic(Zb_tr, Y_tr[:, k]) for k in range(3)])

def j_nhi_phan(Xb, t, w):
    p = np.clip(sigmoid(Xb @ w), 1e-12, 1 - 1e-12)
    return -np.mean(t * np.log(p) + (1 - t) * np.log(1 - p))

print("Nhược điểm 1 — lớp ở giữa bị che khuất:")
for k in range(3):
    print(f"  bộ phân loại '{TEN_LOP[k]} chống phần còn lại': J = {j_nhi_phan(Zb_tr, Y_tr[:, k], W_ovr[:, k]):.3f}   w = {np.round(W_ovr[:, k], 3)}")
p_hang = Y_tr[:, 1].mean()
print(f"  mốc 'chỉ biết tỉ lệ lớp' (p = 1/3): J = {-(p_hang*np.log(p_hang) + (1-p_hang)*np.log(1-p_hang)):.4f}  (≈ 0,6365)")
print("  -> bộ 'đạt' có J ≈ mốc: nó GẦN NHƯ KHÔNG HỌC ĐƯỢC GÌ (lớp giữa không tách được bằng một đường thẳng)")

P_ovr = sigmoid(Zb_te @ W_ovr)                       # xác suất độc lập của từng bộ
tong = P_ovr.sum(axis=1)
print("\nNhược điểm 2 — vùng 'không bộ nào nhận':")
print(f"  không bộ nào cho p >= 0,5: {(P_ovr < 0.5).all(axis=1).sum()}/{len(y_te)} mẫu ({(P_ovr < 0.5).all(axis=1).mean():.1%})  (tài liệu: 64 mẫu — số liệu phụ thuộc η, epoch)")
print("Nhược điểm 3 — xác suất không cộng lại bằng 1:")
print(f"  tổng p0+p1+p2: trung bình {tong.mean():.3f}, nhỏ nhất {tong.min():.3f}, lớn nhất {tong.max():.3f}; {np.mean(np.abs(tong - 1) > 0.2):.1%} mẫu lệch quá 0,2")
x_6_3 = (np.array([[6.0, 3.0]]) - mu) / s
print("  sinh viên học 6 giờ, vắng 3 buổi: p =", np.round(sigmoid(them_bias(x_6_3) @ W_ovr)[0], 3), "-> tổng =", round(float(sigmoid(them_bias(x_6_3) @ W_ovr).sum()), 3))

y_hat_ovr = P_ovr.argmax(axis=1)
C_ovr = np.zeros((3, 3), dtype=int)
for t_, p_ in zip(y_te, y_hat_ovr): C_ovr[t_, p_] += 1
print(f"\nOvR: độ chính xác kiểm tra = {np.mean(y_hat_ovr == y_te):.1%}; ma trận nhầm lẫn (hàng: thật, cột: dự đoán):\n{C_ovr}")
print("  (tài liệu: mục 5.5/6.2 nêu OvR = 80,6% — khớp kết quả này; mục 1.3 nêu 83,9% với cấu hình huấn luyện khác)")
print("  lớp 'đạt' (giữa) được nhận đúng", f"{C_ovr[1,1]}/60 = {C_ovr[1,1]/60:.1%}", "— lớp ở giữa chịu thiệt nhất")


# In[ ]:


# Hình 3: (a) đường p=0,5 của các bộ OvR, (b) vùng không bộ nào nhận, (c) tổng xác suất
u, v = np.meshgrid(np.linspace(-0.5, 14, 300), np.linspace(-1, 7, 300))
G = (np.c_[u.ravel(), v.ravel()] - mu) / s
P_g = sigmoid(them_bias(G) @ W_ovr)
khong_ai = (P_g < 0.5).all(axis=1).reshape(u.shape)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for k in range(3):
    ax[0].scatter(*X_te[y_te == k].T, s=12, color=MAU_LOP[k], alpha=0.6)
    ax[0].contour(u, v, P_g[:, k].reshape(u.shape), levels=[0.5], colors=MAU_LOP[k], linewidths=2)
ax[0].set(title="(a) đường p = 0,5 của từng bộ OvR (bộ 'đạt' hầu như không có)", xlabel="x1", ylabel="x2")
ax[1].contourf(u, v, khong_ai.astype(int), levels=[-0.5, 0.5, 1.5], colors=["white", "lightgray"])
for k in range(3): ax[1].scatter(*X_te[y_te == k].T, s=12, color=MAU_LOP[k], alpha=0.7)
ax[1].set(title=f"(b) vùng xám = không bộ nào nhận (~{khong_ai.mean():.0%} khung hình)", xlabel="x1")
ax[2].hist(tong, bins=25, color="tab:orange", alpha=0.8); ax[2].axvline(1, color="k", ls="--", label="tổng = 1")
ax[2].set(title="(c) tổng 3 xác suất của OvR dao động quanh 1", xlabel="p0 + p1 + p2"); ax[2].legend()
plt.tight_layout(); plt.show()


# > ### 🔍 Giải thích mã
# > **`W_ovr = np.column_stack([... for k in range(3)])`** — mỗi phần tử trong danh sách là một vector tham số $w_k$ (độ dài $d{+}1$) của bộ phân loại độc lập thứ $k$; `column_stack` xếp chúng thành các **cột** của một ma trận $(d{+}1)\times K$ — *cùng hình thức* với $\mathbf W$ của Softmax. Điểm khác nhau không nằm ở cách tính điểm số mà ở (i) cách đổi điểm số thành xác suất (sigmoid độc lập hay softmax chung) và (ii) cách huấn luyện (từng cột riêng hay cả ma trận đồng thời).
# >
# > **`Y_tr[:, k]`** — cột $k$ của $\mathbf Y$ chính là nhãn nhị phân "lớp $k$ hay không".
# >
# > **`(P_g < 0.5).all(axis=1)`** — `P_g < 0.5` cho ma trận Boolean $N\times3$; `.all(axis=1)` kiểm tra *cả ba* bộ đều trả lời "không" tại điểm lưới đó ⇒ vùng "không bộ nào nhận".
# >
# > **Lưới điểm:** `np.meshgrid` tạo toạ độ lưới, `np.c_[u.ravel(), v.ravel()]` ghép thành ma trận $N\times2$ các điểm; ta chuẩn hóa bằng đúng `mu, s` của tập huấn luyện rồi mới đưa vào mô hình.

# **Hướng 2 — mô hình xác suất chung (Softmax):** $K$ điểm số $s_{ik}=\mathbf x_i\mathbf w_k$ được đưa qua **một** hàm chung $p_{ik}=\dfrac{e^{s_{ik}}}{\sum_je^{s_{ij}}}$ nên $p_{ik}\ge0$ và $\sum_kp_{ik}=1$ ngay từ đầu; các lớp **cạnh tranh** vì mẫu số chứa điểm số của mọi lớp. Kết quả sơ bộ cho thấy lớp "đạt" được nhận ra tốt hơn rõ rệt (ta sẽ huấn luyện ở mục 3):

# In[ ]:


W_tam, _ = huan_luyen(Zb_tr, Y_tr, eta=0.5, so_epoch=2000)
yh = du_doan(Zb_te, W_tam)
C_sm = np.zeros((3, 3), dtype=int)
for t_, p_ in zip(y_te, yh): C_sm[t_, p_] += 1
print(f"Softmax: độ chính xác kiểm tra = {np.mean(yh == y_te):.1%} (tài liệu: 85,0%)\n{C_sm}   (tài liệu: [[49 11 0],[9 48 3],[0 4 56]])")
print(f"Lớp 'đạt' (giữa): Softmax {C_sm[1,1]}/60 = {C_sm[1,1]/60:.1%}  so với OvR {C_ovr[1,1]}/60 = {C_ovr[1,1]/60:.1%}")
print("\n--- Kiểm tra nhanh mục 1 ---")
print("5) OvR p = [0.62, 0.35, 0.41]: tổng =", 0.62 + 0.35 + 0.41, "| lớp dự đoán = 0 | chia cho tổng:", np.round(np.array([0.62, 0.35, 0.41]) / 1.38, 3))
print(f"6) mốc cross-entropy khi chỉ biết tỉ lệ 1/3: {-(1/3*np.log(1/3) + 2/3*np.log(2/3)):.4f}")
print("4) K = 10: tỉ lệ lớp dương của mỗi bài OvR = 10% | số mô hình OvR = 10 | OvO = 10·9/2 =", 10 * 9 // 2)


# ---
# # 2. Mô hình Softmax Regression
# 
# ## 2.2. Ma trận tham số $\mathbf W$ và điểm số $\mathbf S=\mathbf{XW}$
# 
# Mỗi lớp $k$ có điểm số tuyến tính $s_k=\mathbf x\mathbf w_k$; xếp $K$ vector thành các **cột** của $\mathbf W$ ($(d{+}1)\times K$), một phép nhân ma trận cho $\mathbf S$ ($n\times K$). Mô hình có $K(d{+}1)$ tham số — với $d=2$, $K=3$ là 9. Ta chọn tay một $\mathbf W$ cho 9 sinh viên: $s_0=1{,}5-x_1+x_2$, $s_1=0$, $s_2=-6{,}5+x_1-x_2$.

# In[ ]:


W_tay = np.array([[ 1.5, 0.0, -6.5],
                  [-1.0, 0.0,  1.0],
                  [ 1.0, 0.0, -1.0]])                    # cột k là w_k
S = Xb_sv @ W_tay
P = softmax(S)
print("S =\n", S); print("P =\n", P.round(3))
print("Tổng mỗi hàng của P:", P.sum(axis=1))
print("y_hat =", P.argmax(axis=1), "| y =", y_sv, "-> đúng cả 9")
print("\nSinh viên 6 (6 giờ, vắng 4): s =", S[5], "-> p =", P[5].round(3), "(dự đoán 'đạt' với độ tin cậy 61,8%)")


# ## 2.3. Hàm softmax: định nghĩa, tính chất, ổn định số học
# 
# $p_k=\dfrac{e^{s_k}}{\sum_je^{s_j}}$. Hai bước: lấy mũ (để dương), chia cho tổng (để cộng bằng 1). Tính chất then chốt: **bất biến khi dịch** — $\mathrm{softmax}(\mathbf s+c)=\mathrm{softmax}(\mathbf s)$ — nên có thể trừ giá trị lớn nhất để tính ổn định:

# In[ ]:


s_vd = np.array([[0.5, 1.5, 2.5]])
print("softmax([0.5, 1.5, 2.5]) =", softmax(s_vd).round(4), " (tài liệu: 0,090 0,245 0,665)")
print("log-odds: ln(p2/p1) =", round(float(np.log(softmax(s_vd)[0, 2] / softmax(s_vd)[0, 1])), 4), "= s2 - s1 =", 2.5 - 1.5)

s_lon = np.array([[1000., 1001., 1002.]])
with np.errstate(over="ignore", invalid="ignore"):
    print("\nnaive :", np.exp(s_lon) / np.exp(s_lon).sum(axis=1, keepdims=True), " (inf/inf = nan)")
print("on dinh:", softmax(s_lon).round(4), " (giống hệt [0.5,1.5,2.5] vì chỉ khác một hằng số)")
print("Bất biến khi dịch:", np.allclose(softmax(s_vd + 3), softmax(s_vd)))

print(f"\n{'c':>5}{'p0':>8}{'p1':>8}{'p2':>8}   (nhân điểm số với c: c lớn -> quyết liệt)")
for c in (0.5, 1, 2, 5):
    print(f"{c:>5}" + "".join(f"{v:>8.3f}" for v in softmax(c * s_vd)[0]))


# ### Khi $K=2$, softmax là sigmoid; đạo hàm của softmax
# 
# $p_1=\dfrac{e^{s_1}}{e^{s_0}+e^{s_1}}=\sigma(s_1-s_0)$ — **Softmax bao hàm Logistic Regression**. Đạo hàm: $\dfrac{\partial p_k}{\partial s_j}=p_k(\mathbb 1_{k=j}-p_j)$, tức Jacobian $=\mathrm{diag}(\mathbf p)-\mathbf p^T\mathbf p$ (mỗi hàng và cột cộng bằng 0 — biểu hiện của sự *cạnh tranh*).

# In[ ]:


s2 = np.array([[0.2, 1.7]])
print(f"K=2: softmax -> p1 = {softmax(s2)[0, 1]:.4f} | sigmoid(s1 - s0) = {sigmoid(1.7 - 0.2):.4f}  (tài liệu: 0,8176)")

p = softmax(np.array([[0.5, 1.5, 2.5]]))[0]
Jac = np.diag(p) - np.outer(p, p)
print("\nJacobian diag(p) - p^T p =\n", Jac.round(3))
print("Tổng mỗi hàng:", Jac.sum(axis=1).round(12), "| tổng mỗi cột:", Jac.sum(axis=0).round(12))

# Kiểm tra bằng đạo hàm số
s0_ = np.array([0.5, 1.5, 2.5]); h = 1e-6
Jac_so = np.column_stack([(softmax((s0_ + h * e)[None])[0] - softmax((s0_ - h * e)[None])[0]) / (2 * h) for e in np.eye(3)])
print("Sai khác tối đa với đạo hàm số:", np.abs(Jac - Jac_so).max())


# > ### 🔍 Giải thích mã
# > **`np.diag(p) - np.outer(p, p)`** — `np.diag(p)` tạo ma trận chéo với $p_k$ trên đường chéo; `np.outer(p, p)` là tích ngoài ($K\times K$) với phần tử $p_kp_j$. Hiệu của chúng có phần tử $p_k\mathbb 1_{k=j}-p_kp_j=p_k(\mathbb 1_{k=j}-p_j)$ — đúng Jacobian của softmax.
# >
# > **Đạo hàm số cho Jacobian:** mỗi *cột* $j$ của Jacobian là $\partial\mathbf p/\partial s_j$; ta tính bằng sai phân trung tâm khi dịch riêng $s_j$ một đoạn $\pm h$ (`h * e` với `e` là hàng của `np.eye(3)`), rồi `column_stack` các cột lại. `(...)[None]` thêm một chiều để `softmax` (yêu cầu ma trận 2 chiều) nhận một "mẫu" duy nhất.

# ## 2.4. Biểu diễn ma trận và xác suất gán cho lớp thật
# 
# Xác suất mô hình gán cho lớp **thật** của mẫu $i$: $q_i=\sum_ky_{ik}p_{ik}$, tức `(Y * P).sum(axis=1)`. Tích $\prod q_i$ là xác suất mô hình gán cho toàn bộ nhãn quan sát — điểm xuất phát của hợp lý cực đại.

# In[ ]:


q = (Y_sv * P).sum(axis=1)
print("q =", q.round(3))
print("Tích các q_i =", round(float(np.prod(q)), 3), "| -trung bình ln q_i =", round(float(-np.mean(np.log(q))), 3), "  (tài liệu: ≈ 0,187 và 0,186)")


# ## 2.5. Vùng quyết định, odds ratio và **tính dư thừa tham số**
# 
# Ranh giới giữa lớp $k$ và $l$: $s_k=s_l\Leftrightarrow\mathbf x(\mathbf w_k-\mathbf w_l)=0$ — vùng của lớp $k$ là **giao của $K{-}1$ nửa không gian** nên **lồi**. Lớp "đạt" nằm giữa được *kẹp* bởi hai ranh giới (Hình 6):

# In[ ]:


u, v = np.meshgrid(np.linspace(0, 12, 300), np.linspace(-1, 7, 300))
Gb = np.c_[np.ones(u.size), u.ravel(), v.ravel()]
Pg = softmax(Gb @ W_tay)
vung = Pg.argmax(axis=1).reshape(u.shape)

fig, ax = plt.subplots(1, 4, figsize=(17, 3.9))
ax[0].contourf(u, v, vung, levels=[-0.5, 0.5, 1.5, 2.5], colors=MAU_LOP, alpha=0.25)
for k in range(3): ax[0].scatter(*X_sv[y_sv == k].T, s=60, color=MAU_LOP[k], edgecolor="k")
ax[0].set(title="(a) ba vùng quyết định (giao của nửa không gian)", xlabel="x1", ylabel="x2")
for k in range(3):
    a = ax[k + 1]
    im = a.contourf(u, v, Pg[:, k].reshape(u.shape), levels=np.linspace(0, 1, 11), cmap="Blues")
    a.contour(u, v, Pg[:, k].reshape(u.shape), levels=[0.5], colors="k", linestyles="--")
    a.scatter(*X_sv[y_sv == k].T, s=40, color=MAU_LOP[k], edgecolor="k")
    a.set(title=f"({'bcd'[k]}) xác suất p{k} ({TEN_LOP[k]})", xlabel="x1")
plt.suptitle("Hình 6", y=1.03); plt.tight_layout(); plt.show()
print("Ranh giới rớt/đạt: x1 - x2 = 1,5 ; đạt/giỏi: x1 - x2 = 6,5 ; rớt/giỏi (x1 - x2 = 4) nằm trong vùng của 'đạt' nên không hiện.")


# In[ ]:


# Odds ratio giữa hai lớp: khi x_j tăng 1, p_k / p_l nhân với e^{w_jk - w_jl}
w_h = W_tay
print("Mỗi giờ học thêm nhân  p_giỏi/p_đạt  với e^(w12-w11) =", round(float(np.exp(w_h[1, 2] - w_h[1, 1])), 3),
      "| p_đạt/p_rớt với e^(w11-w10) =", round(float(np.exp(w_h[1, 1] - w_h[1, 0])), 3))
x6 = Xb_sv[5]; x7 = x6 + np.array([0, 1, 0])
r6 = softmax((x6 @ w_h)[None])[0]; r7 = softmax((x7 @ w_h)[None])[0]
print(f"SV 6: p_giỏi/p_đạt = {r6[2]/r6[1]:.4f} -> học thêm 1 giờ: {r7[2]/r7[1]:.4f} = {r6[2]/r6[1]:.4f} × {np.exp(1):.3f}  (kiểm chứng)")


# ### Tính dư thừa tham số
# 
# Cộng **cùng một vector $\mathbf v$ vào mọi cột** của $\mathbf W$ ($\mathbf W'=\mathbf W+\mathbf v\mathbf 1^T$) làm mọi điểm số của mẫu $i$ dịch cùng một lượng $\mathbf x_i\mathbf v$ ⇒ softmax không đổi. Vậy chỉ có $(K{-}1)(d{+}1)$ tham số độc lập, và chỉ các **hiệu** $\mathbf w_k-\mathbf w_l$ mới có ý nghĩa:

# In[ ]:


v = np.array([[-1.5], [1.0], [-1.0]])
W2 = W_tay + v                                       # cộng cùng một vector v vào MỌI cột
print("W' =\n", W2)
print("max |P' - P| =", np.abs(softmax(Xb_sv @ W2) - P).max(), "-> cùng mô hình")
print("Với W' (cột đầu bằng 0): ln(p_đạt/p_rớt) của SV 6 =", round(float(Xb_sv[5] @ W2[:, 1]), 3), "= ln", round(float(P[5, 1] / P[5, 0]), 3), "→", round(float(np.log(P[5, 1] / P[5, 0])), 3))
print(f"Số tham số: tổng {W_tay.size}, độc lập (K-1)(d+1) = {2 * 3}")


# > ### 🔍 Giải thích mã
# > **`W_tay + v`** — `v` có dạng cột $(d{+}1)\times1$, cộng vào `W_tay` $(d{+}1)\times K$ nhờ *broadcasting*: cột `v` được "trải" qua mọi cột của `W_tay`, đúng bằng $\mathbf W+\mathbf v\mathbf 1^T$ (tích ngoài của $\mathbf v$ với vector toàn số 1).
# >
# > **Lớp tham chiếu:** nếu chọn $\mathbf v=-\mathbf w_0$ thì cột đầu thành 0 và mỗi cột còn lại có nghĩa rất rõ: $\ln\dfrac{p_k}{p_0}=\mathbf x\mathbf w_k'$ — log-odds so với lớp tham chiếu (với $K=2$ chính là Logistic Regression).

# In[ ]:


print("--- Kiểm tra nhanh mục 2 ---")
p1 = softmax(np.array([[2, -1, 0.5]]))[0]; print("1) softmax([2,-1,0.5]) =", p1.round(3), "| ln(p0/p2) =", round(float(np.log(p1[0]/p1[2])), 3), "= s0 - s2 =", 1.5)
print("3) K=2, s=[1,3]: softmax p1 =", round(float(softmax(np.array([[1., 3.]]))[0, 1]), 3), "| sigmoid(3-1) =", round(float(sigmoid(2.0)), 3))
print("4) K=4, d=5: W là 6×4, tổng 24 tham số, độc lập (K-1)(d+1) = 18")
W5 = np.array([[0., -1., -3.], [0., 1., 2.]])        # cột k là w_k = (chệch, hệ số)
p5 = softmax(np.array([[1, 2.]]) @ W5)[0]
print("5) tại x=2:", p5.round(3), "-> lớp 1 và 2 có xác suất bằng nhau: x=2 nằm trên ranh giới lớp 1/lớp 2")


# ---
# # 3. Huấn luyện mô hình Softmax Regression
# 
# ## 3.1. Cross-entropy nhiều lớp (và vì sao không dùng MSE)
# 
# Với MSE $\|\mathbf p-\mathbf y\|^2$, gradient theo điểm số chứa Jacobian của softmax nên *tắt* khi mô hình sai chắc chắn. Mẫu thuộc lớp 2 nhưng mô hình có $\mathbf s=[4,0,-2]$:

# In[ ]:


s_sai = np.array([[4.0, 0.0, -2.0]]); y_sai = np.array([[0.0, 0.0, 1.0]])
p_sai = softmax(s_sai)
Jac = np.diag(p_sai[0]) - np.outer(p_sai[0], p_sai[0])
g_mse = 2 * (p_sai - y_sai) @ Jac
g_ce = p_sai - y_sai
print("p =", p_sai.round(3), "| ||p - y|| =", round(float(np.linalg.norm(p_sai - y_sai)), 3), " (sai rất nặng)")
print("Gradient theo điểm số:  MSE ->", np.round(g_mse, 4), "độ lớn", round(float(np.linalg.norm(g_mse)), 3), " (≈ 0,056: gần như 'tắt')")
print("                        CE  ->", np.round(g_ce, 4),  "độ lớn", round(float(np.linalg.norm(g_ce)), 3), " (lớn đúng ở vùng sai nặng)")


# **Từ 2 lớp lên $K$ lớp:** $l_i=-\sum_ky_{ik}\ln p_{ik}=-\ln q_i$ và
# $$J(\mathbf W)=-\frac1n\sum_{i}\sum_{k}y_{ik}\ln p_{ik}.$$
# Kiểm tra: $J(\mathbf 0)=\ln K$; $W_{tay}$ trên 9 sinh viên cho $J\approx0{,}186$:

# In[ ]:


print(f"J(W = 0) = {mat_mat(Xb_sv, Y_sv, np.zeros((3, 3))):.4f}  = ln 3 = {np.log(3):.4f}")
print("Mất mát từng sinh viên -ln q_i:", (-np.log(q)).round(3))
print(f"J(W_tay) = {mat_mat(Xb_sv, Y_sv, W_tay):.4f}  (tài liệu: 0,186)\n")

# Dạng log-sum-exp khớp công thức gốc
print("Công thức gốc  -mean(sum(Y*ln P)):", round(float(-np.mean(np.sum(Y_sv * np.log(P), axis=1))), 6))
print("Log-sum-exp (mat_mat)            :", round(float(mat_mat(Xb_sv, Y_sv, W_tay)), 6))


# ## 3.2. Hợp lý cực đại, bất biến theo cách đặt tên lớp
# 
# So sánh vài bộ tham số trên 9 sinh viên (hoán vị cột của $\mathbf W$ mà giữ nguyên $\mathbf Y$ là *đổi mô hình* chứ không đổi tên lớp):

# In[ ]:


def L_lnL_J(W):
    J = mat_mat(Xb_sv, Y_sv, W)
    return np.exp(-9 * J), -9 * J, J

W_bo_vang = W_tay.copy(); W_bo_vang[2] = 0
W_bo_gio = W_tay.copy(); W_bo_gio[1] = 0
W_doi_cot = W_tay[:, [2, 1, 0]]
cac_W = [("W = 0", np.zeros((3, 3))), ("W_tay bỏ số buổi vắng", W_bo_vang), ("W_tay bỏ số giờ học", W_bo_gio),
         ("W_tay đổi chỗ cột w0, w2", W_doi_cot), ("W_tay", W_tay), ("2 W_tay", 2 * W_tay)]
print(f"{'Bộ tham số':<28}{'L(W)':>12}{'ln L':>10}{'J(W)':>8}")
for ten, W_ in cac_W:
    L, lnL, J = L_lnL_J(W_)
    print(f"{ten:<28}{L:>12.3e}{lnL:>10.3f}{J:>8.3f}")
print("(tài liệu J: 1,099; 0,845; 4,449; 6,631; 0,186; 0,063)")

# Hoán vị ĐỒNG THỜI cột của Y và W: J không đổi
pi = [2, 0, 1]
print("\nHoán vị đồng thời cột Y và cột W: J =", round(float(mat_mat(Xb_sv, Y_sv[:, pi], W_tay[:, pi])), 6), "= J gốc", round(float(mat_mat(Xb_sv, Y_sv, W_tay)), 6))


# ## 3.3. Gradient $\nabla_{\mathbf W}J=\frac1n\mathbf X^T(\mathbf P-\mathbf Y)$ và quy tắc cập nhật
# 
# Chuỗi $\mathbf W\to s_{ij}\to p_{ik}\to l_i$: $\ \dfrac{\partial l_i}{\partial\mathbf s_i}=\mathbf p_i-\mathbf y_i$ (thừa số $p_{ik}$ triệt tiêu mẫu số), và $\nabla_{\mathbf W}l_i=\mathbf x_i^T(\mathbf p_i-\mathbf y_i)$ (tích ngoài, ma trận hạng một). Cập nhật: $\mathbf W\leftarrow\mathbf W-\frac\eta n\mathbf X^T(\mathbf P-\mathbf Y)$ — **kéo** cột lớp đúng, **đẩy** các cột lớp sai theo trọng số $p_{ik}$.
# 
# Hệ quả đẹp: $(\mathbf P-\mathbf Y)\mathbf 1_K=\mathbf 0$ nên **mỗi hàng của gradient cộng bằng 0** ⇒ tổng các cột của $\mathbf W$ **không đổi** qua mọi bước.

# In[ ]:


# Kiểm tra gradient bằng đạo hàm số trên ma trận W ngẫu nhiên
rng = np.random.default_rng(0)
W_t = rng.normal(0, 0.3, size=(3, 3)); eps = 1e-6
g_so = np.zeros_like(W_t)
for i in range(3):
    for j in range(3):
        E = np.zeros_like(W_t); E[i, j] = eps
        g_so[i, j] = (mat_mat(Xb_sv, Y_sv, W_t + E) - mat_mat(Xb_sv, Y_sv, W_t - E)) / (2 * eps)
g_gt = gradient(Xb_sv, Y_sv, W_t)
print("gradient giải tích:\n", g_gt, "\nsai khác tối đa so với đạo hàm số:", np.abs(g_gt - g_so).max())
print("Tổng mỗi hàng của gradient:", g_gt.sum(axis=1).round(12), "(= 0)")

# Ví dụ trong tài liệu: s = [0.5, 1.5, 2.5], lớp thật = 1
p_e = softmax(np.array([[0.5, 1.5, 2.5]]))[0]; y_e = np.array([0, 1, 0])
print("\nVí dụ: p - y =", (p_e - y_e).round(3), "| mất mát = -ln 0,245 =", round(float(-np.log(p_e[1])), 3))


# > ### 🔍 Giải thích mã (đạo hàm số cho ma trận)
# > Với tham số là **ma trận**, ta chỉ có thể dịch *một phần tử* $W_{ij}$ mỗi lần: `E = np.zeros_like(W_t); E[i, j] = eps` tạo ma trận "đơn vị" tại ô $(i,j)$, rồi tính $\dfrac{J(\mathbf W+\epsilon\mathbf E_{ij})-J(\mathbf W-\epsilon\mathbf E_{ij})}{2\epsilon}$. Hai vòng `for i`, `for j` duyệt qua cả $9$ phần tử và điền vào `g_so` cùng kích thước với `W_t`. Nếu `g_gt` (giải tích) và `g_so` (số) trùng nhau cỡ $10^{-10}$ thì công thức $\frac1n\mathbf X^T(\mathbf P-\mathbf Y)$ đúng — kỹ thuật **gradient check**.

# ## 3.6. Ví dụ tính tay một vòng huấn luyện ($K=3$, $\eta=0{,}1$, $\mathbf W^{(0)}=\mathbf 0$)
# 
# Bước 1: mọi $p_{ik}=\frac13$ nên $J=\ln3$; hàng đầu của gradient bằng 0 (ba lớp cân bằng).

# In[ ]:


W = np.zeros((3, 3)); eta = 0.1
P0 = softmax(Xb_sv @ W)
print("P0 toàn 1/3:", np.allclose(P0, 1 / 3), "| J0 =", round(float(mat_mat(Xb_sv, Y_sv, W)), 4))
XtP = Xb_sv.T @ P0; XtY = Xb_sv.T @ Y_sv
print("X^T P0 =\n", XtP.round(3), "\nX^T Y  =\n", XtY, "\nX^T (P0 - Y) =\n", (XtP - XtY).round(3))
g1 = gradient(Xb_sv, Y_sv, W)
print("Gradient tại W0 =\n", g1.round(4), "\nW1 = W0 - 0.1·gradient =\n", (W - eta * g1).round(4))

print("\nBảng J qua các bước (đối chiếu tài liệu: 1,0986; 0,9217; 0,8159; 0,7429; 0,6872; 0,6437; 0,6082):")
W = np.zeros((3, 3))
for t in range(0, 7):
    dung = int(np.sum(du_doan(Xb_sv, W) == y_sv))
    print(f"  t = {t}: J = {mat_mat(Xb_sv, Y_sv, W):.4f} | đúng {dung}/9 | tổng mỗi hàng của W = {W.sum(axis=1).round(10)}")
    W = W - eta * gradient(Xb_sv, Y_sv, W)


# In[ ]:


# Dùng huan_luyen (batch) và đối chiếu với mô tả của tài liệu
for k in (1, 2, 6):
    W_k, ls = huan_luyen(Xb_sv, Y_sv, eta=0.1, so_epoch=k)
    print(f"sau {k} buoc: J = {ls[-1]:.4f}\n{W_k}")
print("J ban dau =", round(ls[0], 4), "= ln 3 =", round(np.log(3), 4))


# Nhận xét: $J$ giảm đều; sau bước 1 chỉ 3/9 mẫu đúng, sau 6 bước đã 9/9 nhưng thuật toán *không dừng* vì xác suất vẫn chưa đủ "chắc chắn". Mỗi hàng của $\mathbf W_t$ cộng bằng 0 ở mọi bước.
# 
# ## 3.4. Tốc độ học và chuẩn hóa đặc trưng
# 
# Softmax **rất nhạy** với $\eta$. Mức tối ưu $J^*$ tìm bằng L-BFGS; đặc trưng chuẩn hóa chạy batch GD từ $\mathbf W=\mathbf 0$:

# In[ ]:


from scipy.optimize import minimize

def toi_uu_chinh_xac(Xb, Y):
    d1, K = Xb.shape[1], Y.shape[1]
    f = lambda v: mat_mat(Xb, Y, v.reshape(d1, K))
    g = lambda v: gradient(Xb, Y, v.reshape(d1, K)).ravel()
    r = minimize(f, np.zeros(d1 * K), jac=g, method="L-BFGS-B", options=dict(maxiter=20000, ftol=1e-16, gtol=1e-12))
    return r.x.reshape(d1, K), r.fun

W_opt, J_opt = toi_uu_chinh_xac(Zb_tr, Y_tr)
print(f"J* = {J_opt:.4f}  (tài liệu: ≈ 0,3730) | J0 = ln 3 = {np.log(3):.4f}")

moc = (10, 100, 500, 2000)
def bang_toc_do_hoc(Xb, etas):
    lich = {}
    print(f"{'η':<7}" + "".join(f"{'sau ' + str(m):>14}" for m in moc))
    for eta_ in etas:
        _, ls = huan_luyen(Xb, Y_tr, eta=eta_, so_epoch=2000)
        lich[eta_] = ls
        print(f"{eta_:<7}" + "".join(f"{ls[m]:>14.3f}" for m in moc))
    return lich

print("\n--- Đặc trưng ĐÃ chuẩn hóa ---")
lich_chuan = bang_toc_do_hoc(Zb_tr, (0.01, 0.1, 0.5, 1, 2, 5, 10))
print("\n--- Đặc trưng GỐC (chưa chuẩn hóa) ---")
Xb_tr_goc = them_bias(X_tr)
lich_goc = bang_toc_do_hoc(Xb_tr_goc, (0.001, 0.01, 0.1, 0.2, 0.5))


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for eta_, ls in lich_chuan.items(): ax[0].plot(ls[:300], label=f"η = {eta_}")
for eta_, ls in lich_goc.items():   ax[1].plot(ls, label=f"η = {eta_}")
for a, ten in zip(ax, ("(a) đặc trưng chuẩn hóa", "(b) đặc trưng gốc")):
    a.axhline(J_opt, color="k", ls=":"); a.set(xlabel="epoch", ylabel="J", title=ten, ylim=(0.3, 1.2)); a.legend(fontsize=7, ncol=2)
plt.suptitle("Hình 8: hàm mất mát theo epoch ứng với các tốc độ học", y=1.02); plt.tight_layout(); plt.show()

# Điều kiện đủ η < 4/λmax(X^T X / n)
for ten, Xb_ in (("gốc", Xb_tr_goc), ("chuẩn hóa", Zb_tr)):
    lam = np.linalg.eigvalsh(Xb_.T @ Xb_ / len(Xb_))
    print(f"Đặc trưng {ten:<10}: trị riêng của X^T X / n = {np.round(lam, 3)} | tỉ số lớn/nhỏ = {lam.max()/lam.min():.0f} | 4/λmax = {4/lam.max():.3f}")
print("(tài liệu: gốc 0,069; 5,24; 53,5 và 4/λmax ≈ 0,075; chuẩn hóa 0,46; 1,00; 1,54 và ≈ 2,6)")


# **Nhận xét:** với đặc trưng chuẩn hóa, $\eta$ từ 0,5 đến 5 hội tụ nhanh về $J^*\approx0{,}373$ (η = 10 dao động); với đặc trưng gốc, $\eta$ phải rất nhỏ và vẫn hội tụ chậm. Chuẩn hóa dùng $\mu,\sigma$ **chỉ từ tập huấn luyện**, và không chuẩn hóa cột 1 của hệ số chệch (độ lệch chuẩn bằng 0).
# 
# ## 3.5 & 3.7. Batch, SGD, Mini-batch trên 420 mẫu huấn luyện (Hình 9)

# In[ ]:


cau_hinh = {"batch (b = n = 420)": dict(eta=0.5, kich_thuoc_batch=None),
            "mini-batch (b = 32)": dict(eta=0.1, kich_thuoc_batch=32),
            "SGD (b = 1)":         dict(eta=0.01, kich_thuoc_batch=1)}
ket_qua = {}
for ten, cfg in cau_hinh.items():
    W_k, ls = huan_luyen(Zb_tr, Y_tr, so_epoch=200, **cfg)
    ket_qua[ten] = ls
    acc_tr = np.mean(du_doan(Zb_tr, W_k) == y_tr); acc_te = np.mean(du_doan(Zb_te, W_k) == y_te)
    print(f"{ten:22s} J = {ls[-1]:.4f} | train {acc_tr:.4f} | test {acc_te:.4f}")
print("Số lần cập nhật mỗi epoch: batch = 1 | mini-batch =", int(np.ceil(420 / 32)), "(13 batch đủ 32 + 1 batch 4 mẫu) | SGD = 420")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for a, n_ep in zip(ax, (100, 40)):
    for ten, ls in ket_qua.items(): a.plot(ls[:n_ep + 1], label=ten)
    a.axhline(J_opt, color="k", ls=":"); a.set(xlabel="epoch", ylabel="J", ylim=(0.35, 0.8)); a.legend(fontsize=8)
plt.suptitle("Hình 9: ba biến thể Gradient Descent", y=1.02); plt.tight_layout(); plt.show()


# In[ ]:


# Huấn luyện batch eta = 0.5 trong 2000 epoch -> ma trận W gần tối ưu (mô hình chính của các mục sau)
W, ls = huan_luyen(Zb_tr, Y_tr, eta=0.5, so_epoch=2000)
print("J cuoi =", round(ls[-1], 4)); print("W =\n", W)
print("tong moi hang cua W:", W.sum(axis=1).round(10), "(khởi tạo 0 + tính chất tổng cột không đổi)")
print("||grad|| =", np.linalg.norm(gradient(Zb_tr, Y_tr, W)))
y_hat = du_doan(Zb_te, W)
print("acc train =", np.mean(du_doan(Zb_tr, W) == y_tr).round(4), "| acc test =", np.mean(y_hat == y_te).round(4))
C = np.zeros((3, 3), dtype=int)
for t_, p_ in zip(y_te, y_hat): C[t_, p_] += 1
print(C)
print("\nĐọc W (đặc trưng chuẩn hóa): hàng giờ học =", W[1].round(2), "-> giờ học tăng: điểm 'rớt' giảm mạnh, 'giỏi' tăng mạnh, 'đạt' gần như đứng yên")

print("\n--- Kiểm tra nhanh mục 3 ---")
p_ = np.array([0.2, 0.5, 0.3]); print("1) lớp thật = 2: mất mát =", round(float(-np.log(p_[2])), 3), "| gradient p - y =", p_ - np.array([0, 0, 1]))
print("3) K=10, W=0: J = ln 10 =", round(float(np.log(10)), 3))
print("5) x=[1,2,1], lớp 1, W=0, SGD η=0.3:")
x5 = np.array([[1, 2, 1.]]); Y5 = one_hot(np.array([1]), 3); W5 = np.zeros((3, 3))
W5n = W5 - 0.3 * gradient(x5, Y5, W5); print("   p mới =", softmax(x5 @ W5n).round(3), "| mất mát mới =", round(float(mat_mat(x5, Y5, W5n)), 3))
print("7) z = (10 - 6,295)/3,108 =", round((10 - 6.295) / 3.108, 3))


# ---
# # 4. Phân tích cơ chế học của Softmax Regression
# 
# Mục này đặt cạnh Logistic Regression để thấy điểm giống và khác. Ta dùng lại 600 sinh viên và ma trận tối ưu $\mathbf W^*$ (đặc trưng chuẩn hóa, $J(\mathbf W^*)\approx0{,}3730$). Trước hết định nghĩa batch GD **có điều chuẩn hóa L2** và khởi tạo tùy ý (dùng từ đây về sau):

# In[ ]:


def huan_luyen_l2(Xb, Y, eta=1.0, so_epoch=1000, lam=0.0, W0=None):
    # Batch Gradient Descent với điều chuẩn hóa L2 (không phạt hàng đầu của W, tức hệ số chệch).
    n, d1 = Xb.shape
    W = np.zeros((d1, Y.shape[1])) if W0 is None else W0.copy()
    D0 = np.eye(d1); D0[0, 0] = 0.0                    # ma trận D0 = diag(0, 1, ..., 1)
    for epoch in range(so_epoch):
        W = W - eta * (gradient(Xb, Y, W) + lam * D0 @ W)
    return W

def hessian(Xb, W):
    # Hessian của J theo vec(W) (xếp các cột w_0, w_1, ... nối nhau), kích thước (d+1)K x (d+1)K.
    P = softmax(Xb @ W)
    H = np.zeros((Xb.shape[1] * W.shape[1],) * 2)
    for i in range(len(Xb)):
        A = np.diag(P[i]) - np.outer(P[i], P[i])        # diag(p_i) - p_i^T p_i, kích thước K x K
        H += np.kron(A, np.outer(Xb[i], Xb[i]))         # A (x) x_i^T x_i
    return H / len(Xb)

W_star = huan_luyen_l2(Zb_tr, Y_tr, eta=2.0, so_epoch=8000)
print("J(W*) =", round(float(mat_mat(Zb_tr, Y_tr, W_star)), 6), "| W* =\n", W_star.round(4))


# > ### 🔍 Giải thích mã
# > **`D0 = np.eye(d1); D0[0, 0] = 0`** — $\mathbf D_0=\mathrm{diag}(0,1,\dots,1)$. `D0 @ W` giữ nguyên các hàng của $d$ đặc trưng và *xóa hàng đầu* (hệ số chệch) ⇒ số hạng phạt $\lambda\mathbf D_0\mathbf W$ chỉ kéo các **trọng số đặc trưng** về 0, không động đến hệ số chệch.
# >
# > **`gradient(...) + lam * D0 @ W`** — gradient của $J_\lambda=J+\frac\lambda2\|\tilde{\mathbf W}\|_F^2$ (với $\tilde{\mathbf W}$ là $\mathbf W$ bỏ hàng đầu). Viết lại: $\mathbf W\leftarrow(\mathbf I-\eta\lambda\mathbf D_0)\mathbf W-\eta\nabla J$ — mỗi bước các trọng số co lại hệ số $1-\eta\lambda$ (*weight decay*).
# >
# > **`hessian`** — Hessian theo $\mathrm{vec}(\mathbf W)$ (các cột xếp nối nhau, kích thước $(d{+}1)K\times(d{+}1)K=9\times9$) là $\frac1n\sum_i\mathbf A_i\otimes\mathbf x_i^T\mathbf x_i$ với $\mathbf A_i=\mathrm{diag}(\mathbf p_i)-\mathbf p_i^T\mathbf p_i$ (Jacobian softmax, $K\times K$). `np.kron(A, B)` là **tích Kronecker**: ma trận khối mà khối $(k,l)$ bằng `A[k, l] * B`. Với `B = np.outer(x, x)` (= $\mathbf x^T\mathbf x$ cỡ $(d{+}1)^2$), khối $(k,l)$ chính là $A_{kl}\,\mathbf x^T\mathbf x$ — đúng đạo hàm bậc hai theo $\mathbf w_k,\mathbf w_l$.

# ## 4.1. Điều gì xảy ra khi một mẫu bị phân loại sai?
# 
# Gradient một mẫu là **tích ngoài** $\nabla_{\mathbf W}l_i=\mathbf x_i^T(\mathbf p_i-\mathbf y_i)$ (hạng một) nên $\|\nabla l_i\|_F=\|\mathbf x_i\|\cdot\|\mathbf p_i-\mathbf y_i\|$. Mẫu đúng và chắc chắn ≈ không đóng góp; mẫu sát biên đóng góp vừa; mẫu sai nặng đóng góp lớn nhất:

# In[ ]:


P_tr = softmax(Zb_tr @ W_star)
dong_gop = np.linalg.norm(Zb_tr, axis=1) * np.linalg.norm(P_tr - Y_tr, axis=1)       # ||x_i|| * ||p_i - y_i||
mau_chon = [((2.27, 4.16), 0, "Đúng, rất chắc chắn"), ((4.48, 2.59), 1, "Đúng nhưng sát biên"),
            ((5.74, 4.11), 0, "Sai nhẹ, sát biên"), ((5.67, 0.17), 0, "Sai nặng"), ((10.00, 0.73), 1, "Sai nặng và ở xa tâm")]
print(f"{'Mẫu x1;x2':<14}{'thật':>6}  {'p (rớt, đạt, giỏi)':<26}{'||p-y||':>8}{'||x||':>7}{'||∇l||':>8}  Tình trạng")
idx_chon = []
for (a, b), k, tt in mau_chon:
    ung_vien = np.where(y_tr == k)[0]
    i = ung_vien[np.argmin(np.linalg.norm(X_tr[ung_vien] - np.array([a, b]), axis=1))]
    idx_chon.append(i)
    print(f"{X_tr[i, 0]:.2f}; {X_tr[i, 1]:.2f}".ljust(14) + f"{TEN_LOP[k]:>6}  {str(P_tr[i].round(3)):<26}{np.linalg.norm(P_tr[i] - Y_tr[i]):>8.3f}{np.linalg.norm(Zb_tr[i]):>7.2f}{dong_gop[i]:>8.3f}  {tt}")

dung = P_tr.argmax(axis=1) == y_tr
print(f"\nSố mẫu sai: {(~dung).sum()}/{len(y_tr)} | độ lớn gradient trung bình: đúng {dong_gop[dung].mean():.3f}, sai {dong_gop[~dung].mean():.3f} "
      f"(gấp {dong_gop[~dung].mean() / dong_gop[dung].mean():.1f} lần)   (tài liệu: 58/420; 0,280 và 1,423)")


# ### Kéo lớp đúng, đẩy các lớp còn lại
# 
# Một bước SGD theo mẫu $i$ (lớp thật $c$): $\mathbf w_c'=\mathbf w_c+\eta(1-p_{ic})\mathbf x_i^T$ (**kéo**), $\mathbf w_k'=\mathbf w_k-\eta\,p_{ik}\mathbf x_i^T$ (**đẩy**, $k\neq c$), và điểm số đổi $\mathbf s_i'=\mathbf s_i+\eta\|\mathbf x_i\|^2(\mathbf y_i-\mathbf p_i)$. Tổng các thay đổi bằng 0. Ví dụ mẫu lớp rớt bị đoán thành đạt:

# In[ ]:


i = idx_chon[3]                                      # mẫu (≈5,67; 0,17), lớp thật = rớt
x_i = Zb_tr[i]; eta_s = 0.3
s_truoc = x_i @ W_star; p_truoc = softmax(s_truoc[None])[0]
delta = eta_s * (x_i @ x_i) * (Y_tr[i] - p_truoc)
W_sau = W_star - eta_s * np.outer(x_i, p_truoc - Y_tr[i])      # W' = W - η x^T (p - y)
s_sau = x_i @ W_sau; p_sau = softmax(s_sau[None])[0]
print("x (chuẩn hóa) =", x_i.round(4), "| ||x||² =", round(float(x_i @ x_i), 3))
print("s  trước =", s_truoc.round(3), "-> p =", p_truoc.round(3))
print("Δs (công thức) =", delta.round(3), "| tổng =", round(float(delta.sum()), 12))
print("s  sau   =", s_sau.round(3), "-> p' =", p_sau.round(3), "| kiểm tra s + Δs =", (s_truoc + delta).round(3))
print(f"Mất mát mẫu: {-np.log(p_truoc[0]):.3f} -> {-np.log(p_sau[0]):.3f} | J toàn tập huấn luyện: {mat_mat(Zb_tr, Y_tr, W_star):.4f} -> {mat_mat(Zb_tr, Y_tr, W_sau):.4f} (tăng: SGD dao động quanh tối ưu)")

# Hình 10a: kích thước điểm ∝ 1 - q_i^2 và vùng quyết định
q_tr = (Y_tr * P_tr).sum(axis=1)
u, v = np.meshgrid(np.linspace(-0.5, 14, 300), np.linspace(-1, 7, 300))
Gz = them_bias((np.c_[u.ravel(), v.ravel()] - mu) / s)
fig, ax = plt.subplots(figsize=(7, 4.8))
ax.contourf(u, v, (Gz @ W_star).argmax(axis=1).reshape(u.shape), levels=[-0.5, 0.5, 1.5, 2.5], colors=MAU_LOP, alpha=0.15)
for k in range(3):
    m = y_tr == k; ax.scatter(*X_tr[m].T, s=8 + 160 * (1 - q_tr[m] ** 2), color=MAU_LOP[k], alpha=0.5, label=TEN_LOP[k])
ax.set(xlabel="x1", ylabel="x2", title="Hình 10a: kích thước điểm ∝ 1 − q²  (đóng góp vào gradient)"); ax.legend(); plt.show()


# > ### 🔍 Giải thích mã
# > **`np.outer(x_i, p_truoc - Y_tr[i])`** — tích ngoài của vector đặc trưng $\mathbf x_i^T$ ($d{+}1$) với sai số $\mathbf p_i-\mathbf y_i$ ($K$) cho ma trận $(d{+}1)\times K$: **cột $k$** của nó là $(p_{ik}-y_{ik})\mathbf x_i^T$. Cột của lớp đúng nhận hệ số âm ($p-1<0$) nên $\mathbf W-\eta\cdot(\dots)$ **cộng** vào ⇒ *kéo*; các cột lớp sai nhận hệ số dương ⇒ *đẩy*, lớp nào đang có $p_{ik}$ lớn bị đẩy mạnh nhất.
# >
# > **`np.linalg.norm(Zb_tr, axis=1)`** — chuẩn Euclid **của từng hàng** (`axis=1`) ⇒ vector $\|\mathbf x_i\|$ cho mọi mẫu; nhân với `np.linalg.norm(P - Y, axis=1)` được $\|\nabla l_i\|_F$ cho mọi mẫu cùng lúc, không vòng lặp.
# >
# > **`ung_vien[np.argmin(...)]`** — trong các mẫu thuộc lớp $k$, chọn mẫu gần nhất (theo khoảng cách) với điểm toạ độ cho trước; để lấy "mẫu điển hình" tương ứng bảng của tài liệu.

# ## 4.2. Chuẩn hóa, khởi tạo và chuyển ngược về thang đo gốc
# 
# Số epoch để $J$ cách $J^*$ không quá $0{,}001$ (batch GD từ $\mathbf W=\mathbf 0$):

# In[ ]:


def so_epoch_hoi_tu(Xb, Y, eta, J_star, dung_sai=1e-3, toi_da=40000):
    W_ = np.zeros((Xb.shape[1], Y.shape[1]))
    for t in range(1, toi_da + 1):
        W_ = W_ - eta * gradient(Xb, Y, W_)
        J = mat_mat(Xb, Y, W_)
        if J - J_star < dung_sai: return t
        if not np.isfinite(J) or J > 20: return None
    return None

J_star = mat_mat(Zb_tr, Y_tr, W_star)
print(f"J* = {J_star:.4f}\n{'η':<7}{'Đặc trưng gốc':>26}{'Chuẩn hóa':>26}")
for eta_ in (0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10):
    a = so_epoch_hoi_tu(Xb_tr_goc, Y_tr, eta_, J_star); b = so_epoch_hoi_tu(Zb_tr, Y_tr, eta_, J_star)
    fa = str(a) if a else (">40000" if eta_ <= 0.01 else "không hội tụ (dao động)")
    fb = str(b) if b else "không hội tụ (dao động)"
    print(f"{eta_:<7}{fa:>26}{fb:>26}")
print("(tài liệu — gốc: >40000, 16341, 8170, dao động…; chuẩn hóa: 24569, 4912, 2455, 1227, 490, 244, 121, 45, dao động)")


# In[ ]:


# Chuyển ngược W' (đặc trưng chuẩn hóa) về thang đo gốc:  w_0k = w'_0k - sum_j (mu_j/s_j) w'_jk ;  w_jk = w'_jk / s_j
W_goc = np.vstack([W_star[0] - (mu / s) @ W_star[1:], W_star[1:] / s[:, None]])
print("W (thang đo gốc) =\n", W_goc.round(3), "\n  (tài liệu: hàng chệch [5,633 1,876 -7,509], hàng x1 [-1,409 0,081 1,328], hàng x2 [0,813 -0,244 -0,569])")
print("Điểm số không đổi:", np.allclose(Zb_tr @ W_star, Xb_tr_goc @ W_goc))
print("Mỗi giờ học thêm làm điểm lớp rớt giảm", abs(W_goc[1, 0]).round(3), "và điểm lớp giỏi tăng", W_goc[1, 2].round(3))


# > ### 🔍 Giải thích mã
# > **`np.vstack([hàng_chệch, W_star[1:] / s[:, None]])`** — ghép hàng chệch mới lên trên các hàng trọng số mới. `W_star[1:] / s[:, None]` chia **hàng $j$** của $\mathbf W'$ cho $\sigma_j$ (`s[:, None]` biến vector $\sigma$ thành cột $d\times1$ để broadcast theo hàng). Hàng chệch: `(mu / s) @ W_star[1:]` là vector độ dài $K$ với phần tử $k$ là $\sum_j(\mu_j/\sigma_j)w'_{jk}$ — đúng công thức $w_{0k}=w'_{0k}-\sum_j\mu_jw'_{jk}/\sigma_j$.
# >
# > **Vì sao đúng?** Điểm số phải *không đổi*: $\mathbf z\mathbf W'=\mathbf x\mathbf W$ với $z_j=(x_j-\mu_j)/\sigma_j$. Khai triển theo $x_j$ cho các công thức trên — dòng `np.allclose(...)` kiểm chứng điều đó trên toàn tập huấn luyện.

# ### Khởi tạo: P và J không đổi, nhưng $\mathbf W$ thì **có** đổi
# 
# Khác Logistic Regression, từ các khởi tạo khác nhau ta thu được các $\mathbf W$ **khác nhau** nhưng cùng $J$ và cùng $\mathbf P$. Vì $\nabla J\,\mathbf 1_K=\mathbf 0$ nên $\mathbf W\mathbf 1_K$ không đổi, suy ra $\mathbf W_\infty=\mathbf W^*_{tâm}+\mathbf v_0\mathbf 1_K^T$ với $\mathbf v_0=\mathbf W_0\mathbf 1_K/K$:

# In[ ]:


W0s = {"A": np.array([[2, -1, 0], [1, 3, -2], [-3, 1, 2.]]), "B": np.array([[1, 1, 1], [-2, 0, 2], [2, -2, 0.]])}
Wc_star = W_star - W_star.mean(axis=1, keepdims=True)         # nghiệm có các hàng cộng lại bằng 0
P_star = softmax(Zb_tr @ W_star)
print(f"{'Khởi tạo':<10}{'J(W0)':>8}{'J cuối':>12}{'max|P-P*|':>12}{'max|W - (W*_tâm + TB hàng của W0)|':>38}")
for ten, W0 in W0s.items():
    Wf = huan_luyen_l2(Zb_tr, Y_tr, eta=2.0, so_epoch=8000, W0=W0)
    print(f"{ten:<10}{mat_mat(Zb_tr, Y_tr, W0):>8.3f}{mat_mat(Zb_tr, Y_tr, Wf):>12.6f}{np.abs(softmax(Zb_tr @ Wf) - P_star).max():>12.1e}"
          f"{np.abs(Wf - (Wc_star + W0.mean(axis=1, keepdims=True))).max():>38.1e}")
print("\nW cuối từ khởi tạo A (khác W* từ 0):\n", huan_luyen_l2(Zb_tr, Y_tr, eta=2.0, so_epoch=8000, W0=W0s["A"]).round(3))


# > ### 🔍 Giải thích mã
# > **`W_star.mean(axis=1, keepdims=True)`** — trung bình **theo từng hàng** (cột $d{+}1$ phần tử). Trừ nó khỏi `W_star` cho ma trận có *các cột trung bình bằng 0* — đại diện "đã căn giữa" `Wc_star` của họ nghiệm $\mathbf W^*+\mathbf v\mathbf 1^T$.
# >
# > Công thức dự đoán nói: nghiệm cuối từ khởi tạo $\mathbf W_0$ phải bằng `Wc_star + W0.mean(axis=1, keepdims=True)` (cộng lại phần trung bình-theo-hàng của khởi tạo, thứ mà gradient *không bao giờ* thay đổi). Cột cuối bảng (`~1e-14`) xác nhận điều đó.

# ## 4.3. Lồi nhưng **suy biến**: phân tích Hessian
# 
# $\mathbf H=\frac1n\sum_i\mathbf A_i\otimes\mathbf x_i^T\mathbf x_i$ nửa xác định dương (dạng toàn phương là *phương sai* nên $\ge0$) ⇒ $J$ **lồi**. Nhưng $\mathbf H\,\mathrm{vec}(\mathbf v\mathbf 1^T)=0$ nên có đúng $d{+}1$ trị riêng bằng 0: nghiệm $\mathbf W$ **không duy nhất** (chỉ $\mathbf P$ duy nhất).

# In[ ]:


H = hessian(Zb_tr, W_star)
tri_rieng = np.linalg.eigvalsh(H)
print("Trị riêng của H:", tri_rieng.round(4), "\n  (tài liệu: 0; 0; 0; 0,0058; 0,0135; 0,0295; 0,0738; 0,1333; 0,2286)")
print(f"Số trị riêng ≈ 0: {(np.abs(tri_rieng) < 1e-9).sum()} = d+1 = 3 | số trị riêng dương: {(tri_rieng > 1e-9).sum()} = (K-1)(d+1) = 6 | hạng H = {np.linalg.matrix_rank(H, tol=1e-9)}")

v = np.array([0.3, -1.2, 0.7]); Zv = np.outer(v, np.ones(3))
print("|| H vec(v 1^T) || =", np.linalg.norm(H @ Zv.T.ravel()), " (≈ 0: các hướng v 1^T nằm trong nhân của H)")

# Cố định lớp tham chiếu w0 = 0: bỏ khối đầu của Hessian -> xác định dương
H_gon = H[3:, 3:]
print("Hessian thu gọn (bỏ khối của cột w0): trị riêng =", np.linalg.eigvalsh(H_gon).round(4), "(tài liệu: 0,0020; 0,0081; 0,0113; 0,0577; 0,0820; 0,2265 — tất cả dương)")

# Hình 11a: lát cắt của J quanh W*
rng = np.random.default_rng(0)
huong_phang = np.outer(rng.normal(size=3), np.ones(3)); huong_khac = rng.normal(size=(3, 3)); huong_khac -= huong_khac.mean(axis=1, keepdims=True)
ts = np.linspace(-3, 3, 121)
J_phang = [mat_mat(Zb_tr, Y_tr, W_star + t * huong_phang) for t in ts]
J_khac = [mat_mat(Zb_tr, Y_tr, W_star + t * huong_khac) for t in ts]
fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.9))
ax[0].plot(ts, J_phang, label="dọc v 1^T (phẳng)"); ax[0].plot(ts, J_khac, label="hướng khác (parabol)")
ax[0].set(xlabel="t", ylabel="J(W* + t·hướng)", title="(a) lát cắt hàm mất mát quanh W*"); ax[0].legend()
for ten, W0 in [("0", np.zeros((3, 3)))] + list(W0s.items()):
    ls_ = [mat_mat(Zb_tr, Y_tr, W0)]; Wt = W0.copy()
    for _ in range(100): Wt = Wt - 2.0 * gradient(Zb_tr, Y_tr, Wt); ls_.append(mat_mat(Zb_tr, Y_tr, Wt))
    ax[1].plot(ls_, label=f"khởi tạo {ten}")
ax[1].set(xlabel="epoch", ylabel="J", yscale="log", title="(b) mọi khởi tạo hội tụ về cùng J"); ax[1].legend(); plt.tight_layout(); plt.show()


# > ### 🔍 Giải thích mã
# > **`Zv.T.ravel()`** — `vec(W)` xếp **các cột** nối nhau; `ravel()` của NumPy duyệt theo *hàng*, nên ta lấy chuyển vị `Zv.T` trước (hàng của $\mathbf Z^T$ = cột của $\mathbf Z$) rồi `ravel()`. Đây là quy ước thứ tự khớp với `np.kron(A, ·)` trong `hessian`.
# >
# > **`np.outer(v, np.ones(3))`** — ma trận $(d{+}1)\times K$ mà *mọi cột đều bằng $\mathbf v$*, tức $\mathbf v\mathbf 1_K^T$: hướng "cộng cùng một vector vào mọi cột".
# >
# > **`H[3:, 3:]`** — Hessian thu gọn khi cố định cột $\mathbf w_0=0$: bỏ khối hàng/cột ứng với cột đầu (3 phần tử đầu của vec). Khi đó không còn hướng phẳng nên xác định dương và nghiệm duy nhất *trong cách tham số hóa này*.
# >
# > **Hình (a):** dọc hướng $\mathbf v\mathbf 1^T$, $J$ phẳng tuyệt đối; dọc hướng khác, $J$ là parabol.

# In[ ]:


# Điều kiện tối ưu: X^T (P - Y) = 0  =>  trung bình xác suất mỗi lớp = tỉ lệ lớp
print("trung bình P =", P_star.mean(axis=0).round(4), "| tỉ lệ lớp =", Y_tr.mean(axis=0).round(4))
print("max |X^T (P - Y)| / n =", np.abs(Zb_tr.T @ (P_star - Y_tr)).max() / len(y_tr))
print("Tổng 'dự đoán × giờ học' = ", (P_star * Zb_tr[:, [1]]).sum(axis=0).round(2), "| tổng 'thực tế × giờ học' =", (Y_tr * Zb_tr[:, [1]]).sum(axis=0).round(2),
      "  (tài liệu: -149,28; -1,08; 150,36)")

# Số điều kiện và ngưỡng ổn định η < 2/λmax (trên các trị riêng dương)
H_goc = hessian(Xb_tr_goc, W_goc)
def thong_so(Hm):
    lam = np.linalg.eigvalsh(Hm); lam = lam[lam > 1e-9]
    return lam.max(), lam.min(), lam.max() / lam.min(), 2 / lam.max()
print(f"\n{'':<24}{'Đặc trưng gốc':>16}{'Chuẩn hóa':>14}")
for ten, g, c in zip(("λ_max", "λ_min dương", "Số điều kiện κ", "Ngưỡng 2/λ_max"), thong_so(H_goc), thong_so(H)):
    print(f"{ten:<24}{g:>16.4f}{c:>14.4f}")
print("(tài liệu: λmax 10,05 và 0,2286; λmin+ 0,0021 và 0,0058; κ ≈ 4880 và 39; ngưỡng ≈ 0,199 và 8,75)")


# **Kết luận:** việc có nhiều hơn hai lớp *không* làm mất tính lồi, nhưng đổi tính "duy nhất" từ $\mathbf W$ sang $\mathbf P$ (cái giá của tham số dư thừa). Gradient Descent không đi theo các hướng phẳng nên không bị ảnh hưởng; ngưỡng ổn định là $\eta<2/\lambda_{\max}$ (trên các trị riêng dương).
# 
# ## 4.4. Dữ liệu phân tách hoàn hảo: $\|\mathbf W\|\to\infty$ và vai trò của L2
# 
# Với 9 sinh viên (phân tách được), $J\to0$ khi $c\to\infty$ trong $c\mathbf W^\circ$ nhưng không bao giờ bằng 0 ⇒ không có cực tiểu hữu hạn. *Lề* của mẫu: $m_i=s_{i,y_i}-\max_{j\ne y_i}s_{ij}$.

# In[ ]:


Z_sv = them_bias((X_sv - X_sv.mean(axis=0)) / X_sv.std(axis=0))

def le_nho_nhat(Xb, y, W):
    S = Xb @ W; m = []
    for i in range(len(y)): m.append(S[i, y[i]] - np.delete(S[i], y[i]).max())
    return min(m)

W = np.zeros((3, 3))
print(f"{'t':>8}{'||W||_F':>10}{'J':>12}{'lề nhỏ nhất':>14}")
for t in range(1, 10**5 + 1):
    W = W - 0.5 * gradient(Z_sv, Y_sv, W)
    if t in (10, 100, 1000, 10**4, 10**5):
        print(f"{t:>8}{np.linalg.norm(W):>10.3f}{mat_mat(Z_sv, Y_sv, W):>12.2e}{le_nho_nhat(Z_sv, y_sv, W):>14.3f}")
print("Các hàng của W vẫn cộng bằng 0:", W.sum(axis=1).round(8))
print("(tài liệu: ||W|| = 1,68; 5,06; 10,98; 18,07; 25,42 và lề -0,128; 1,056; 2,911; 5,173; 7,477)")


# > ### 🔍 Giải thích mã
# > **`np.delete(S[i], y[i]).max()`** — xóa phần tử thứ $y_i$ (điểm số của lớp thật) khỏi hàng điểm số rồi lấy giá trị lớn nhất của các lớp *còn lại* ⇒ $\max_{j\neq y_i}s_{ij}$. Lề dương nghĩa là mẫu được phân loại đúng; lề tăng dần theo log của số bước nghĩa là $\|\mathbf W\|$ vẫn đang phình ra dù mọi mẫu đã đúng từ bước 100.

# In[ ]:


# Điều chuẩn hóa L2: W hữu hạn, xác suất dè dặt hơn (sinh viên 6)
x6 = them_bias(((np.array([[6, 4.]]) - X_sv.mean(axis=0)) / X_sv.std(axis=0)))
print(f"{'λ':<8}{'||W~||_F':>10}   p(sinh viên 6) = (rớt, đạt, giỏi)")
nhieu_lam = {}
for lam in (0.0, 0.001, 0.01, 0.1):
    W = huan_luyen_l2(Z_sv, Y_sv, eta=0.5, so_epoch=100000, lam=lam)
    nhieu_lam[lam] = softmax(x6 @ W)[0]
    print(f"{lam:<8}{np.linalg.norm(W[1:]):>10.3f}   {nhieu_lam[lam].round(3)}")
print("(tài liệu: 23,00 / 9,08 / 4,68 / 1,81 ; p cuối = 0,334 0,589 0,077; so với ma trận chọn tay: 0,618 cho lớp đạt)")

fig, ax = plt.subplots(figsize=(6, 3.6)); lams = list(nhieu_lam)
for k in range(3): ax.plot(range(4), [nhieu_lam[l][k] for l in lams], "o-", color=MAU_LOP[k], label=TEN_LOP[k])
ax.set_xticks(range(4), [f"λ={l}" for l in lams]); ax.set(ylabel="xác suất", title="Hình 12b: λ càng lớn, xác suất sinh viên 6 càng dè dặt"); ax.legend(); plt.show()


# In[ ]:


# Đánh đổi khi chọn λ trên 600 sinh viên (batch, eta=1, 8000 epoch)
print(f"{'λ':<8}{'||W~||_F':>10}{'Acc train':>11}{'Acc test':>10}{'Log-loss test':>15}")
for lam in (0, 0.001, 0.01, 0.1, 1):
    Wl = huan_luyen_l2(Zb_tr, Y_tr, eta=1.0, so_epoch=8000, lam=lam)
    print(f"{lam:<8}{np.linalg.norm(Wl[1:]):>10.2f}{np.mean(du_doan(Zb_tr, Wl) == y_tr):>11.1%}{np.mean(du_doan(Zb_te, Wl) == y_te):>10.1%}{mat_mat(Zb_te, Y_te, Wl):>15.3f}")
print("(tài liệu: acc test 85,0% / 85,6% / 85,0% / 80,0% / 73,3%; log-loss test 0,354 / 0,358 / 0,407 / 0,591 / 0,884)")


# ## 4.5. Giới hạn của vùng quyết định tuyến tính: ba vòng tròn đồng tâm
# 
# Vùng của mỗi lớp là giao các nửa không gian ⇒ **luôn lồi** ⇒ không phân loại được lớp lồng nhau. Dữ liệu: 600 điểm trên ba vòng tròn bán kính 1, 3, 5. Mô hình tuyến tính chỉ ~35%; thêm đặc trưng đa thức $[x_1,x_2,x_1^2,x_1x_2,x_2^2]$ đẩy lên ~99%:

# In[ ]:


from itertools import combinations_with_replacement

def sinh_vong(seed=7, n_lop=200):
    # Ba vòng tròn đồng tâm (bán kính 1; 3; 5), mỗi lớp 200 điểm.
    rng = np.random.default_rng(seed)
    r = np.r_[rng.normal(1.0, 0.35, n_lop), rng.normal(3.0, 0.35, n_lop), rng.normal(5.0, 0.35, n_lop)]
    g = rng.uniform(0, 2 * np.pi, 3 * n_lop)
    return np.c_[r * np.cos(g), r * np.sin(g)], np.repeat([0, 1, 2], n_lop)

def da_thuc(X, bac=2):
    # Đặc trưng đa thức bậc 1..bac (không gồm cột 1): x1, x2, x1^2, x1*x2, x2^2, ...
    cot = []
    for k in range(1, bac + 1):
        for idx in combinations_with_replacement(range(X.shape[1]), k):
            cot.append(np.prod(X[:, idx], axis=1))
    return np.column_stack(cot)

Xv, yv = sinh_vong()
Xc_tr, Xc_te, yc_tr, yc_te = chia_train_test(Xv, yv)
Yc_tr = one_hot(yc_tr, 3)
mo_hinh_vong = {}
print(f"{'Mô hình':<20}{'Số tham số':>11}{'J':>9}{'train':>8}{'test':>8}{'||W~||_F':>10}")
for ten, bac, lam in [("tuyen tinh", 1, 0.0), ("bac 2, lam = 0", 2, 0.0), ("bac 2, lam = 1e-3", 2, 1e-3), ("bac 3, lam = 1e-3", 3, 1e-3)]:
    Fa, Fb = da_thuc(Xc_tr, bac), da_thuc(Xc_te, bac)
    Za, Zb, mu_f, s_f = chuan_hoa(Fa, Fb)
    Za, Zb = them_bias(Za), them_bias(Zb)
    Wv = huan_luyen_l2(Za, Yc_tr, eta=1.0 if bac < 3 else 0.5, so_epoch=5000, lam=lam)
    mo_hinh_vong[ten] = (bac, Wv, mu_f, s_f)
    print(f"{ten:<20}{Wv.size:>11}{mat_mat(Za, Yc_tr, Wv):>9.4f}{np.mean(du_doan(Za, Wv) == yc_tr):>8.3f}{np.mean(du_doan(Zb, Wv) == yc_te):>8.3f}{np.linalg.norm(Wv[1:]):>10.2f}")
print("(tài liệu: 0,362/0,350 ; 1,000/0,994 (L=0); 0,998/0,994; 1,000/0,994 và ||W|| = 0,11; 22,12; 11,15; 11,13)")


# In[ ]:


# Hình 13: vùng quyết định của ba mô hình
g = np.linspace(-6.5, 6.5, 260); U, V = np.meshgrid(g, g); luoi = np.c_[U.ravel(), V.ravel()]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for a, ten, tieu_de in zip(ax, ("tuyen tinh", "bac 2, lam = 1e-3", "bac 3, lam = 1e-3"),
                           ("(a) đặc trưng gốc: vùng suy biến", "(b) thêm bậc 2: biên là hai đường tròn", "(c) thêm bậc 3: gần như không đổi")):
    bac, Wv, mu_f, s_f = mo_hinh_vong[ten]
    F = da_thuc(luoi, bac)
    nhan = (them_bias((F - mu_f) / s_f) @ Wv).argmax(axis=1).reshape(U.shape)
    a.contourf(U, V, nhan, levels=[-0.5, 0.5, 1.5, 2.5], colors=MAU_LOP, alpha=0.25)
    for k in range(3): a.scatter(*Xc_te[yc_te == k].T, s=10, color=MAU_LOP[k], edgecolor="k", lw=0.3)
    a.set(title=tieu_de, xlabel="x1", ylabel="x2", aspect="equal")
plt.suptitle("Hình 13: Softmax Regression trên ba vòng tròn đồng tâm", y=1.02); plt.tight_layout(); plt.show()

from math import comb
print("Số đặc trưng đa thức C(d+k, k) - 1 và số tham số K·C(d+k, k):")
for d_, k_, K_ in [(2, 2, 3), (2, 3, 3), (10, 2, 10), (10, 3, 10)]:
    print(f"  d={d_:<3} bậc {k_} K={K_:<3} -> đặc trưng mới = {comb(d_ + k_, k_) - 1:>4} | tham số = {K_ * comb(d_ + k_, k_):>5}")
print("(tài liệu: 5/18; 9/30; 65/660; 285/2860)")


# > ### 🔍 Giải thích mã
# > **Vì sao mô hình tuyến tính "đứng yên" ở $\mathbf W\approx0$?** Các lớp đều có tâm ở gốc tọa độ nên trung bình $x_1,x_2$ của mỗi lớp bằng 0 ⇒ gradient $\frac1n\mathbf X^T(\mathbf P-\mathbf Y)$ tại $\mathbf W=\mathbf 0$ gần như bằng 0 (hàng chệch cũng 0 vì ba lớp cân bằng). Đây là phiên bản nhiều lớp của XOR.
# >
# > **`da_thuc`** — `combinations_with_replacement(range(d), k)` sinh các tổ hợp có lặp $k$ chỉ số cột (ví dụ `(0,0), (0,1), (1,1)` cho $x_1^2,x_1x_2,x_2^2$); `np.prod(X[:, idx], axis=1)` nhân các cột đó theo hàng thành một đặc trưng mới.
# >
# > **Vẽ biên cong:** với mỗi điểm lưới, ta dựng đặc trưng đa thức rồi chuẩn hóa bằng đúng `mu_f, s_f` của tập huấn luyện (lưu trong `mo_hinh_vong`) *trước khi* đưa vào mô hình — thiếu bước này biên sẽ sai.

# In[ ]:


print("--- Kiểm tra nhanh mục 4 ---")
p = np.array([0.1, 0.7, 0.2]); ds = 0.05 * 10 * (np.array([1, 0, 0]) - p)
s_new = np.log(p) + ds; p_new = softmax(s_new[None])[0]
print("1) Δs =", ds.round(2), "| p mới =", p_new.round(3), "| mất mát", round(float(-np.log(0.1)), 3), "->", round(float(-np.log(p_new[0])), 3))
pA, pB = np.array([0.05, 0.9, 0.05]), np.array([0.5, 0.4, 0.1]); yy = np.array([0, 1, 0])
print("2) ||p-y|| mẫu A =", round(float(np.linalg.norm(pA - yy)), 3), "| mẫu B =", round(float(np.linalg.norm(pB - yy)), 3), "-> B lớn hơn", round(float(np.linalg.norm(pB - yy) / np.linalg.norm(pA - yy)), 1), "lần")
print("4) d=5, K=4: Hessian", (6 * 4, 6 * 4), "| hạng (K-1)(d+1) =", 3 * 6, "| số trị riêng bằng 0 =", 6)
print("6) lề m=2, K=3: l(c) = ln(1 + 2 e^{-2c}):", [round(float(np.log(1 + 2 * np.exp(-2 * c))), 4) for c in (1, 2, 3)])
print("7) λ=0,1, η=0,5: sau 10 bước còn", round((1 - 0.5 * 0.1) ** 10, 3), "so với ban đầu")
print("8) d=3, bậc 2, K=4: đặc trưng mới =", comb(3 + 2, 2) - 1, "| tham số = K·(9+1) =", 4 * 10)


# ---
# # 5. Đánh giá và trực quan hóa mô hình
# 
# Mô hình dùng ở mục này: $\mathbf W$ của mục 3.7 (batch GD, $\eta=0{,}5$, 2000 epoch, đặc trưng chuẩn hóa), 420 mẫu huấn luyện / 180 mẫu kiểm tra.
# 
# ## 5.1. Đường cong học, bản đồ xác suất và phân tích lỗi
# 
# Theo dõi $J$ và accuracy trên **cả hai tập** (tập kiểm tra chỉ để quan sát, không dùng để cập nhật $\mathbf W$):

# In[ ]:


W_sm = huan_luyen(Zb_tr, Y_tr, eta=0.5, so_epoch=2000)[0]      # mô hình của mục 3.7
Wt = np.zeros((3, 3)); hist = {"J_tr": [], "J_te": [], "a_tr": [], "a_te": []}
print("epoch | J train | J test | acc train | acc test")
for epoch in range(2001):
    hist["J_tr"].append(mat_mat(Zb_tr, Y_tr, Wt)); hist["J_te"].append(mat_mat(Zb_te, Y_te, Wt))
    hist["a_tr"].append(np.mean(du_doan(Zb_tr, Wt) == y_tr)); hist["a_te"].append(np.mean(du_doan(Zb_te, Wt) == y_te))
    if epoch in (0, 10, 50, 300, 2000):
        print(f"{epoch:5d} | {hist['J_tr'][-1]:.4f} | {hist['J_te'][-1]:.4f} | {hist['a_tr'][-1]:.4f} | {hist['a_te'][-1]:.4f}")
    Wt = Wt - 0.5 * gradient(Zb_tr, Y_tr, Wt)
print("Epoch 0: mọi p = 1/3 nên J = ln 3 và argmax trả về lớp có chỉ số nhỏ nhất (lớp rớt) -> accuracy = 1/3")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(hist["J_tr"], label="J huấn luyện"); ax[0].plot(hist["J_te"], label="J kiểm tra"); ax[0].set(xlabel="epoch", ylabel="J", xscale="symlog", title="(a) hàm mất mát"); ax[0].legend()
ax[1].plot(hist["a_tr"], label="train"); ax[1].plot(hist["a_te"], label="test"); ax[1].set(xlabel="epoch", ylabel="accuracy", xlim=(0, 300), title="(b) accuracy bão hòa sớm"); ax[1].legend()
plt.suptitle("Hình 14: đường cong học", y=1.02); plt.tight_layout(); plt.show()


# In[ ]:


# Hình 15: bản đồ xác suất từng lớp + độ tin cậy; vòng vàng = mẫu kiểm tra bị dự đoán sai
P_te = softmax(Zb_te @ W_sm); y_hat = P_te.argmax(axis=1); sai = y_hat != y_te
u, v = np.meshgrid(np.linspace(-0.5, 14, 300), np.linspace(-1, 7, 300))
Gz = them_bias((np.c_[u.ravel(), v.ravel()] - mu) / s); Pg = softmax(Gz @ W_sm)
fig, ax = plt.subplots(1, 4, figsize=(18, 4))
for k in range(3):
    ax[k].contourf(u, v, Pg[:, k].reshape(u.shape), levels=np.linspace(0, 1, 11), cmap="Blues")
    ax[k].set_title(f"({'abc'[k]}) xác suất p{k} ({TEN_LOP[k]})")
ax[3].contourf(u, v, Pg.max(axis=1).reshape(u.shape), levels=np.linspace(1/3, 1, 11), cmap="Greys"); ax[3].set_title("(d) độ tin cậy max_k p_k")
for a in ax:
    a.contour(u, v, Pg.argmax(axis=1).reshape(u.shape), levels=[0.5, 1.5], colors="k", linewidths=1.5)
    for k in range(3): a.scatter(*X_te[y_te == k].T, s=18, color=MAU_LOP[k], edgecolor="w", lw=0.5)
    a.scatter(*X_te[sai].T, s=110, facecolors="none", edgecolors="gold", linewidths=2); a.set(xlabel="x1", ylabel="x2")
plt.suptitle("Hình 15: bản đồ xác suất (đường đen: biên quyết định; vòng vàng: mẫu sai)", y=1.03); plt.tight_layout(); plt.show()

W_g = np.vstack([W_sm[0] - (mu / s) @ W_sm[1:], W_sm[1:] / s[:, None]])
d01, d12 = np.linalg.norm(W_g[1:, 0] - W_g[1:, 1]), np.linalg.norm(W_g[1:, 1] - W_g[1:, 2])
print(f"||w0 - w1|| = {d01:.3f}, ||w1 - w2|| = {d12:.3f} (tài liệu: 1,825 và 1,288)")
print(f"Khoảng cách từ biên đến mức tỉ số xác suất 9:1: rớt/đạt = ln9/{d01:.3f} = {np.log(9)/d01:.2f} | đạt/giỏi = ln9/{d12:.3f} = {np.log(9)/d12:.2f}  (≈ 1,20 và 1,71)")


# In[ ]:


q = P_te[np.arange(len(y_te)), y_te]                     # xác suất gán cho lớp thật
mat_mat_mau = -np.log(q); tin_cay = P_te.max(axis=1)
print("so mau sai:", sai.sum(), "| sai voi do tin cay < 0.7:", (sai & (tin_cay < 0.7)).sum(),
      "| 0.7-0.9:", (sai & (tin_cay >= 0.7) & (tin_cay < 0.9)).sum(), "| >= 0.9:", (sai & (tin_cay >= 0.9)).sum())
print("ti le mat mat do cac mau sai dong gop:", (mat_mat_mau[sai].sum() / mat_mat_mau.sum()).round(3), "(tài liệu: 0,582)")
i = mat_mat_mau.argmax()
print("mau mat mat lon nhat:", X_te[i].round(2), "lop that", y_te[i], "p =", P_te[i].round(3), "| mat mat =", mat_mat_mau[i].round(2))
print(f"Độ tin cậy ≥ 0,9: {(tin_cay >= 0.9).sum()} mẫu, sai {(sai & (tin_cay >= 0.9)).sum()} ({(sai & (tin_cay >= 0.9)).sum() / (tin_cay >= 0.9).sum():.1%}) | "
      f"độ tin cậy < 0,7: {(tin_cay < 0.7).sum()} mẫu, sai {(sai & (tin_cay < 0.7)).sum()} ({(sai & (tin_cay < 0.7)).sum() / (tin_cay < 0.7).sum():.0%})")


# ## 5.2. Tập huấn luyện/kiểm tra và hiện tượng học thuộc (overfitting)
# 
# Chỉ lấy **45 mẫu huấn luyện** (15 mẫu mỗi lớp), dựng đặc trưng đa thức bậc $k$, huấn luyện bằng **L-BFGS** (tối đa 300 vòng lặp) với/không điều chuẩn L2, đánh giá trên cả 180 mẫu kiểm tra:

# In[ ]:


def giai_lbfgs(Xb, Y, lam=0.0, max_iter=300):
    # Tối ưu J + (lam/2)||W~||_F^2 bằng L-BFGS của SciPy, dùng lại hàm mất mát và gradient đã cài đặt.
    d1, K = Xb.shape[1], Y.shape[1]
    D0 = np.eye(d1); D0[0, 0] = 0
    f = lambda v: mat_mat(Xb, Y, v.reshape(d1, K)) + 0.5 * lam * np.sum((D0 @ v.reshape(d1, K)) ** 2)
    g = lambda v: (gradient(Xb, Y, v.reshape(d1, K)) + lam * D0 @ v.reshape(d1, K)).ravel()
    r = minimize(f, np.zeros(d1 * K), jac=g, method="L-BFGS-B", options={"maxiter": max_iter})
    return r.x.reshape(d1, K)

rng = np.random.default_rng(3)
idx = np.concatenate([rng.choice(np.where(y_tr == k)[0], 15, replace=False) for k in range(3)])   # 15 mẫu mỗi lớp
X_nho, y_nho = X_tr[idx], y_tr[idx]; Y_nho = one_hot(y_nho, 3)
bang_qt = []
print(f"{'bậc':>4} | {'λ':<5} | {'tham số':>7} | {'J train':>8} | {'J test':>9} | {'acc train':>9} | {'acc test':>8}")
for bac in (1, 2, 3, 5):
    F_nho, F_kt = da_thuc(X_nho, bac), da_thuc(X_te, bac)
    Z_nho, Z_kt, _, _ = chuan_hoa(F_nho, F_kt)
    Z_nho, Z_kt = them_bias(Z_nho), them_bias(Z_kt)
    for lam in (0.0, 0.01):
        Wn = giai_lbfgs(Z_nho, Y_nho, lam=lam, max_iter=300)
        r = (bac, lam, Wn.size, mat_mat(Z_nho, Y_nho, Wn), mat_mat(Z_kt, Y_te, Wn), np.mean(du_doan(Z_nho, Wn) == y_nho), np.mean(du_doan(Z_kt, Wn) == y_te))
        bang_qt.append(r)
        print(f"{r[0]:>4} | {r[1]:<5} | {r[2]:>7} | {r[3]:>8.3f} | {r[4]:>9.3f} | {r[5]:>9.3f} | {r[6]:>8.3f}")


# > ### 🔍 Giải thích mã
# > **`minimize(f, x0, jac=g, method="L-BFGS-B")`** — bộ giải của SciPy làm việc với **vector** một chiều, nên ta *duỗi* $\mathbf W$ thành vector: `np.zeros(d1 * K)` là điểm khởi đầu, và trong `f`, `g` ta `v.reshape(d1, K)` để khôi phục dạng ma trận trước khi gọi `mat_mat`/`gradient`; gradient trả về được `.ravel()` lại thành vector. Đây là cách chuẩn để dùng bộ giải tổng quát cho tham số dạng ma trận.
# >
# > **`lambda`** — hai hàm nặc danh `f` (mục tiêu) và `g` (gradient) bắt lấy `Xb, Y, lam, D0` từ phạm vi bao quanh. `0.5 * lam * np.sum((D0 @ W) ** 2)` là số hạng phạt $\frac\lambda2\|\tilde{\mathbf W}\|_F^2$ (bỏ hàng chệch nhờ $\mathbf D_0$).
# >
# > **`rng.choice(np.where(y_tr == k)[0], 15, replace=False)`** — chọn ngẫu nhiên 15 chỉ số *không lặp* trong số các mẫu thuộc lớp $k$; `np.concatenate` ghép ba lớp lại thành tập 45 mẫu cân bằng. **Chuẩn hóa dùng thống kê của 45 mẫu này** rồi áp cho tập kiểm tra (tránh rò rỉ); với đặc trưng đa thức, chuẩn hóa là bắt buộc vì $x^5$ lệch thang đo với $x$.

# In[ ]:


bacs = [1, 2, 3, 5]
def lay(lam, c): return [r[c] for r in bang_qt if r[1] == lam]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for lam, mau_ in [(0.0, "tab:red"), (0.01, "tab:blue")]:
    ax[0].plot(bacs, np.array(lay(lam, 3)) + 1e-3, "--o", color=mau_, label=f"train, λ={lam}"); ax[0].plot(bacs, lay(lam, 4), "-o", color=mau_, label=f"test, λ={lam}")
    ax[1].plot(bacs, lay(lam, 5), "--o", color=mau_); ax[1].plot(bacs, lay(lam, 6), "-o", color=mau_)
ax[0].set(xlabel="bậc đa thức", ylabel="log-loss (thang log)", yscale="log", title="(a) log-loss"); ax[0].legend(fontsize=7)
ax[1].set(xlabel="bậc đa thức", ylabel="accuracy", title="(b) accuracy")
plt.suptitle("Hình 16: không điều chuẩn (đỏ) → log-loss kiểm tra vọt lên; L2 (xanh) → ổn định", y=1.02); plt.tight_layout(); plt.show()
print("Log-loss phát hiện overfitting nhạy hơn accuracy: ở bậc 2, accuracy kiểm tra giảm ~9 điểm trong khi log-loss tăng ~6,6 lần (0,396 -> 2,609).")
print("Giá trị như 209,8 phụ thuộc giới hạn 300 vòng lặp (chạy thêm còn tăng): hãy nhớ BẬC ĐỘ LỚN chứ không phải con số chính xác.")


# ## 5.3. Ma trận nhầm lẫn $K\times K$, precision/recall/F1 theo lớp, ma trận chi phí
# 
# $C_{ij}$ = số mẫu lớp thật $i$ bị dự đoán là $j$. Với lớp $k$: $TP_k=C_{kk}$, $FN_k$ = phần còn lại của **hàng** $k$, $FP_k$ = phần còn lại của **cột** $k$; precision chia cho tổng **cột**, recall chia cho tổng **hàng**.

# In[ ]:


def ma_tran_nham_lan(y, y_hat, K):
    # Ma trận nhầm lẫn K x K: C[i, j] = số mẫu thuộc lớp thật i bị dự đoán là lớp j.
    C = np.zeros((K, K), dtype=int)
    for t, p in zip(y, y_hat):
        C[t, p] += 1
    return C

def chi_so_theo_lop(C):
    # Precision, recall, F1 của từng lớp (coi lớp đó là 'dương', các lớp còn lại là 'âm').
    tp = np.diag(C).astype(float)
    fp = C.sum(axis=0) - tp          # tổng cột trừ đường chéo
    fn = C.sum(axis=1) - tp          # tổng hàng trừ đường chéo
    return tp / (tp + fp), tp / (tp + fn), 2 * tp / (2 * tp + fp + fn), C.sum(axis=1)

def trung_binh(C):
    prec, rec, f1, sup = chi_so_theo_lop(C)
    return {"macro": (prec.mean(), rec.mean(), f1.mean()),
            "weighted": tuple(np.average(v, weights=sup) for v in (prec, rec, f1)),
            "micro": (np.trace(C) / C.sum(),) * 3}          # micro: precision = recall = F1 = accuracy

P_tr, P_te = softmax(Zb_tr @ W_sm), softmax(Zb_te @ W_sm)
yh_tr, yh_te = P_tr.argmax(axis=1), P_te.argmax(axis=1)
C_te = ma_tran_nham_lan(y_te, yh_te, 3)
print("C kiểm tra =\n", C_te, "| accuracy = trace/n =", np.trace(C_te) / C_te.sum())
print("C huấn luyện =\n", ma_tran_nham_lan(y_tr, yh_tr, 3))
prec, rec, f1, sup = chi_so_theo_lop(C_te)
print("\nprecision =", prec.round(3), "| recall =", rec.round(3), "| F1 =", f1.round(3), "| support =", sup)
for ten, vv in trung_binh(C_te).items(): print(f"{ten:9s}", np.round(vv, 3))

C_lech = np.array([[90, 5, 5], [10, 30, 10], [2, 3, 5]])          # ví dụ tập lệch: lớp giỏi hiếm
prec, rec, f1, sup = chi_so_theo_lop(C_lech)
print("\nVí dụ tập lệch — precision =", prec.round(3), "| recall =", rec.round(3), "| F1 =", f1.round(3), "| support =", sup)
for ten, vv in trung_binh(C_lech).items(): print(f"{ten:9s}", np.round(vv, 3))
print("-> macro-F1 (0,635) thấp hơn nhiều accuracy (0,781) vì lớp hiếm có F1 chỉ 0,333; weighted/micro che mất điểm yếu đó.")


# > ### 🔍 Giải thích mã
# > **`np.diag(C)`** lấy đường chéo (TP của từng lớp); **`C.sum(axis=0)`** là tổng theo **cột** (số mẫu được *dự đoán* là lớp $k$) nên `fp = tổng cột - tp`; **`C.sum(axis=1)`** là tổng theo **hàng** (support, số mẫu *thật* của lớp $k$) nên `fn = tổng hàng - tp`. Ba vector `tp, fp, fn` cho precision/recall/F1 của **tất cả lớp cùng lúc**, không cần vòng lặp qua lớp.
# >
# > **`np.average(v, weights=sup)`** — trung bình có trọng số theo support (*weighted*), còn `v.mean()` là trung bình đơn giản (*macro*). **Micro** gộp TP/FP/FN toàn bộ rồi tính; vì mỗi mẫu sai là đúng một FP và đúng một FN, nên micro-precision = micro-recall = micro-F1 = accuracy.

# ### Quyết định theo chi phí
# 
# Với ma trận chi phí $c_{kj}$ (thật $k$, dự đoán $j$), chi phí kỳ vọng khi dự đoán $j$ là $\sum_kp_kc_{kj}$ và quy tắc tối ưu là $\hat y=\arg\min_j(\mathbf p\,\mathbf c)_j$. Khi mọi sai lầm tốn như nhau, quy tắc trở về $\arg\max_jp_j$.

# In[ ]:


chi_phi = np.array([[0, 3, 3], [1, 0, 1], [1, 1, 0]])          # chi_phi[thật, dự đoán]; bỏ sót sinh viên rớt tốn gấp 3
y_cp = (P_te @ chi_phi).argmin(axis=1)                         # chọn lớp có chi phí kỳ vọng nhỏ nhất
C_cp = ma_tran_nham_lan(y_te, y_cp, 3)
print(C_cp)
print("tong chi phi: argmax =", (C_te * chi_phi).sum(), "| theo chi phi =", (C_cp * chi_phi).sum(), "| acc =", np.mean(y_cp == y_te).round(4))
print(f"Recall lớp rớt: {C_te[0,0]/60:.1%} -> {C_cp[0,0]/60:.1%}; đổi lại accuracy {np.mean(yh_te == y_te):.1%} -> {np.mean(y_cp == y_te):.1%}")
p_vd = np.array([0.2, 0.5, 0.3]); print("Ví dụ p = [0.2, 0.5, 0.3]: p·c =", p_vd @ chi_phi, "-> chọn lớp", (p_vd @ chi_phi).argmin(), "(rớt) dù argmax chọn lớp", p_vd.argmax(), "(đạt)")


# > ### 🔍 Giải thích mã
# > **`P_te @ chi_phi`** — $\mathbf P$ ($n\times K$) nhân ma trận chi phí ($K\times K$) cho ma trận $n\times K$ mà phần tử $(i,j)$ là $\sum_kp_{ik}c_{kj}$ = **chi phí kỳ vọng** khi dự đoán lớp $j$ cho mẫu $i$. `argmin(axis=1)` chọn lớp rẻ nhất cho từng mẫu. Cả bước ra quyết định tối ưu chỉ là *một phép nhân ma trận*. Quy tắc chỉ đúng khi các xác suất được **hiệu chuẩn** tốt, và ma trận chi phí phải do bài toán quy định chứ không được chọn bằng cách thử trên tập kiểm tra.

# ## 5.4. Chất lượng xác suất: log-loss, Brier, top-$k$, hiệu chuẩn (ECE), AUC one-vs-rest

# In[ ]:


def log_loss_da_lop(P, y):
    return -np.mean(np.log(P[np.arange(len(y)), y]))

def brier_da_lop(P, y):
    return np.mean(np.sum((P - one_hot(y, P.shape[1])) ** 2, axis=1))

def top_k_accuracy(P, y, k):
    so_lop_cao_hon = (P > P[np.arange(len(y)), y][:, None]).sum(axis=1)     # số lớp có xác suất cao hơn lớp thật
    return np.mean(so_lop_cao_hon < k)

def bang_hieu_chuan(P, y):
    do_tin_cay, dung = P.max(axis=1), P.argmax(axis=1) == y
    bien = [1 / P.shape[1], 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    hang = []
    for a, b in zip(bien[:-1], bien[1:]):
        m = (do_tin_cay >= a) & (do_tin_cay < b)
        if m.any(): hang.append((a, min(b, 1.0), int(m.sum()), do_tin_cay[m].mean(), dung[m].mean()))
    return hang

def ece(P, y):
    # Expected calibration error theo độ tin cậy max_k p_k.
    return sum(n / len(y) * abs(acc - conf) for _, _, n, conf, acc in bang_hieu_chuan(P, y))

def auc_cap(diem, nhan_duong):
    # AUC bằng cách đếm cặp (dương, âm).
    pos, neg = diem[nhan_duong], diem[~nhan_duong]
    return np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])

print("log-loss =", round(log_loss_da_lop(P_te, y_te), 4), "| Brier =", round(brier_da_lop(P_te, y_te), 4))
p0 = Y_tr.mean(axis=0); print("hang (ti le lop): log-loss =", round(-np.mean(np.log(p0[y_te])), 4), "| Brier =", round(np.mean(np.sum((p0 - one_hot(y_te, 3)) ** 2, axis=1)), 4))
print(f"e^(-log-loss) = {np.exp(-log_loss_da_lop(P_te, y_te)):.3f} (trung bình nhân xác suất gán cho lớp thật; so với 0,333 nếu đoán mò)")
print("top-1 =", top_k_accuracy(P_te, y_te, 1), "| top-2 =", top_k_accuracy(P_te, y_te, 2), "| top-3 =", top_k_accuracy(P_te, y_te, 3), "-> lớp thật luôn nằm trong 2 ứng viên đầu")
print("ECE =", round(ece(P_te, y_te), 4), "(tài liệu: 0,050)")
auc = [auc_cap(P_te[:, k], y_te == k) for k in range(3)]
print("AUC OvR theo lop =", np.round(auc, 4), "| macro =", round(np.mean(auc), 4), "(tài liệu: 0,969; 0,920; 0,993; macro 0,961)")
print(f"\n{'Nhóm theo max p':<18}{'Số mẫu':>8}{'tin cậy TB':>12}{'tỉ lệ đúng':>12}")
for a, b, n, conf, acc in bang_hieu_chuan(P_te, y_te):
    print(f"[{a:.3f}; {b:.1f}){'':<5}{n:>8}{conf:>12.3f}{acc:>12.3f}")
print("Trung bình pk trên tập kiểm tra:", P_te.mean(axis=0).round(3), "so với tỉ lệ thực 0,333 (trên tập huấn luyện bằng nhau chính xác)")


# In[ ]:


# Hình 17 + 18
fig, ax = plt.subplots(1, 4, figsize=(18, 4))
im = ax[0].imshow(C_te, cmap="Blues")
for (i_, j_), val in np.ndenumerate(C_te): ax[0].text(j_, i_, val, ha="center", va="center", fontsize=14, color="white" if val > C_te.max() / 2 else "black")
ax[0].set_xticks(range(3), TEN_LOP); ax[0].set_yticks(range(3), TEN_LOP); ax[0].set(xlabel="dự đoán", ylabel="thực tế", title="(17a) ma trận nhầm lẫn"); ax[0].grid(False)
prec, rec, f1, _ = chi_so_theo_lop(C_te); x_ = np.arange(3)
ax[1].bar(x_ - 0.25, prec, 0.25, label="precision"); ax[1].bar(x_, rec, 0.25, label="recall"); ax[1].bar(x_ + 0.25, f1, 0.25, label="F1")
ax[1].set_xticks(x_, TEN_LOP); ax[1].set(ylim=(0.6, 1), title="(17b) chỉ số theo lớp"); ax[1].legend(fontsize=8)

dung_te = yh_te == y_te; tc = P_te.max(axis=1)
ax[2].hist([tc[dung_te], tc[~dung_te]], bins=15, stacked=True, color=["tab:blue", "tab:red"], label=["đúng", "sai"])
ax[2].set(xlabel="độ tin cậy max p", title="(18a) lỗi dồn về độ tin cậy thấp"); ax[2].legend()
hc = bang_hieu_chuan(P_te, y_te)
ax[3].plot([0, 1], [0, 1], "k--", lw=0.8); ax[3].plot([h[3] for h in hc], [h[4] for h in hc], "o-")
for h in hc: ax[3].annotate(f"n={h[2]}", (h[3], h[4]), fontsize=7)
ax[3].set(xlabel="độ tin cậy trung bình", ylabel="tỉ lệ đúng thực tế", title="(18c) hiệu chuẩn")
plt.tight_layout(); plt.show()

# Hình 18b: ROC one-vs-rest
def duong_roc(y_nhi, p):
    thu_tu = np.argsort(-p); y_s = y_nhi[thu_tu]
    return np.r_[0, np.cumsum(1 - y_s) / (1 - y_nhi).sum()], np.r_[0, np.cumsum(y_s) / y_nhi.sum()]
plt.figure(figsize=(5, 4.2))
for k in range(3):
    fpr, tpr = duong_roc((y_te == k).astype(int), P_te[:, k]); plt.plot(fpr, tpr, color=MAU_LOP[k], label=f"{TEN_LOP[k]} (AUC={auc[k]:.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=0.8); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("Hình 18b: ROC one-vs-rest"); plt.legend(); plt.show()


# > ### 🔍 Giải thích mã
# > **`P[np.arange(len(y)), y]`** — lập chỉ mục bằng *hai mảng cùng độ dài*: cặp $(i,y_i)$ cho mỗi $i$ ⇒ lấy đúng $p_{i,y_i}$ (xác suất lớp thật) cho mọi mẫu, tức vector $\mathbf q$. `log_loss = -mean(log(q))`.
# >
# > **Top-$k$:** `P > P[np.arange(n), y][:, None]` so sánh *mọi* xác suất của mẫu với xác suất lớp thật của chính nó (`[:, None]` tạo cột để broadcast); `.sum(axis=1)` đếm số lớp có xác suất **cao hơn** lớp thật. Nếu số đó `< k` thì lớp thật nằm trong $k$ ứng viên đầu.
# >
# > **ECE:** `bang_hieu_chuan` chia mẫu theo độ tin cậy $\max_kp_{ik}$ thành các nhóm; trong mỗi nhóm so *độ tin cậy trung bình* với *tỉ lệ đoán đúng thực tế*. $\mathrm{ECE}=\sum_b\frac{n_b}{n}|\mathrm{acc}_b-\mathrm{conf}_b|$ là độ lệch trung bình có trọng số; ECE nhỏ nghĩa là "khi mô hình nói chắc 80% thì thực tế đúng ~80%".
# >
# > **AUC của lớp $k$ (OvR):** coi $p_{ik}$ là điểm số cho bài toán "lớp $k$ hay không", rồi đếm cặp (dương, âm): `[(a > b) + 0.5*(a == b) for a in pos for b in neg]` — xác suất một mẫu thuộc lớp $k$ có $p_k$ cao hơn một mẫu không thuộc lớp $k$. **Đường ROC** (`duong_roc`) sắp mẫu theo điểm giảm dần (`np.argsort(-p)`) rồi `cumsum` số dương thật/âm giả sau mỗi bước chia cho tổng.

# ## 5.5. So sánh với scikit-learn, one-vs-rest và kiểm định chéo
# 
# `LogisticRegression` đa lớp dùng chính cross-entropy nhiều lớp (multinomial/softmax). So khớp **trên đặc trưng gốc**, $C=10^{10}$ (gần như không điều chuẩn):

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import log_loss, roc_auc_score, f1_score, confusion_matrix, classification_report, top_k_accuracy_score

clf = LogisticRegression(C=1e10, max_iter=10000).fit(X_tr, y_tr)
W_goc_sm = np.vstack([W_sm[0] - (mu / s) @ W_sm[1:], W_sm[1:] / s[:, None]])         # đổi về thang đo gốc (mục 4.2)
print("intercept_ =", clf.intercept_.round(4)); print("coef_ =\n", clf.coef_.round(4))
print("tu cai dat (thang goc), W^T =\n", W_goc_sm.T.round(4))
p_sk = clf.predict_proba(X_te); p_ours = softmax(them_bias(X_te) @ W_goc_sm)
print("max |p_sk - p_ours| =", np.abs(p_sk - p_ours).max().round(5), "(tài liệu: 0,00035)")
print("Tổng intercept_ =", float(clf.intercept_.sum().round(6)), "(thư viện khởi tạo 0 nên tổng các cột của W cũng không đổi, mục 3.3)")
W_tu_sk = np.vstack([clf.intercept_, clf.coef_.T]); print("Dựng W từ sklearn: np.vstack([intercept_, coef_.T]) có kích thước", W_tu_sk.shape)


# In[ ]:


clf_md = LogisticRegression().fit(X_tr, y_tr)                                    # C = 1 (mặc định)
ovr = OneVsRestClassifier(LogisticRegression(C=1e10, max_iter=10000)).fit(X_tr, y_tr)
print(f"{'Mô hình':<24}{'train':>8}{'test':>8}{'log-loss':>10}{'AUC OvR':>9}")
for ten, m in [("tu cai dat", None), ("sklearn C=1e10", clf), ("sklearn mac dinh C=1", clf_md), ("sklearn one-vs-rest", ovr)]:
    p = p_ours if m is None else m.predict_proba(X_te)
    a_tr = np.mean(du_doan(Zb_tr, W_sm) == y_tr) if m is None else m.score(X_tr, y_tr)
    a_te = np.mean(p.argmax(axis=1) == y_te)
    print(f"{ten:<24}{a_tr:>8.4f}{a_te:>8.4f}{log_loss(y_te, p):>10.4f}{roc_auc_score(y_te, p, multi_class='ovr'):>9.4f}")
print("(tài liệu: 0,8619/0,8500/0,3536/0,9606 cho hai dòng đầu; OvR 0,8333/0,8056/0,4772/0,9337)")


# In[ ]:


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
print(f"{'bậc':>4} | {'C':<7}{'CV accuracy':>18}{'CV log-loss':>14}")
for bac in (1, 2):
    for C in (0.001, 0.01, 0.1, 1, 10, 100):
        mo_hinh = make_pipeline(PolynomialFeatures(bac, include_bias=False), StandardScaler(), LogisticRegression(C=C, max_iter=10000))
        acc = cross_val_score(mo_hinh, X_tr, y_tr, cv=cv)
        ll = -cross_val_score(mo_hinh, X_tr, y_tr, cv=cv, scoring="neg_log_loss")
        print(f"{bac:>4} | {C:<7}{acc.mean():>11.4f} ± {acc.std():.4f}{ll.mean():>14.4f}")
print("-> C quá nhỏ (0,001) làm điều chuẩn quá mạnh (underfit); từ C ≈ 1 kết quả gần như không đổi; bậc 2 không cải thiện đáng kể (ưu tiên mô hình đơn giản).")
for bac, C in ((1, 10), (2, 1)):
    mh = make_pipeline(PolynomialFeatures(bac, include_bias=False), StandardScaler(), LogisticRegression(C=C, max_iter=10000)).fit(X_tr, y_tr)
    print(f"bac {bac}, C = {C}: test accuracy = {mh.score(X_te, y_te):.4f} | log-loss = {log_loss(y_te, mh.predict_proba(X_te)):.4f}   (đánh giá MỘT lần sau khi chọn xong)")


# > ### 🔍 Giải thích mã
# > **`make_pipeline(PolynomialFeatures, StandardScaler, LogisticRegression)`** — đóng gói các bước thành một "mô hình" duy nhất. Trong kiểm định chéo, `cross_val_score` gọi `fit` của **toàn pipeline trên 4 phần huấn luyện** của mỗi lần chia (nên `StandardScaler` chỉ tính $\mu,\sigma$ từ 4 phần đó) rồi mới `predict` trên phần kiểm định ⇒ không rò rỉ thông tin. Nếu chuẩn hóa *cả tập huấn luyện* trước khi chia fold thì phần kiểm định đã "lọt" vào thống kê, điểm CV lạc quan giả tạo.
# >
# > **`scoring="neg_log_loss"`** — scikit-learn quy ước *điểm càng lớn càng tốt* nên log-loss được trả về dưới dạng **âm**; ta thêm dấu `-` ở trước. Chọn siêu tham số theo log-loss (phân biệt rõ hơn accuracy vốn dao động ±0,03).
# >
# > **`np.vstack([intercept_, coef_.T])`** — `intercept_` (độ dài $K$) là **hàng chệch** của $\mathbf W$ và `coef_` có dạng $K\times d$ (mỗi *hàng* là một lớp) nên cần chuyển vị thành $d\times K$ trước khi xếp xuống dưới. **Liên hệ $C$–$\lambda$:** hàm mục tiêu của thư viện là $C\sum_il_i+\frac12\|\mathbf W\|_F^2$ ⇒ $\lambda=\frac1{Cn}$.

# In[ ]:


print(confusion_matrix(y_te, clf.predict(X_te)))
print(classification_report(y_te, clf.predict(X_te), target_names=["rot", "dat", "gioi"], digits=3))
print("top-2 =", top_k_accuracy_score(y_te, p_sk, k=2))
print("Dòng 'accuracy' của báo cáo chính là micro-F1.")

print("\n--- Kiểm tra nhanh mục 5 ---")
C5 = np.array([[50, 5, 5], [10, 30, 10], [0, 5, 35]])
print("2) C =", C5.tolist(), "| accuracy =", round(np.trace(C5) / C5.sum(), 3))
pr, rc, ff, su = chi_so_theo_lop(C5)
print("   precision =", pr.round(3), "| recall =", rc.round(3), "| F1 =", ff.round(3), "| macro-F1 =", round(ff.mean(), 3), "| weighted-F1 =", round(float(np.average(ff, weights=su)), 3))
print("3) luôn dự đoán lớp 0 (95% / 2,5% / 2,5%): accuracy = 0,95 | macro-recall =", round((1 + 0 + 0) / 3, 3))
print("4) p = [0.5, 0.3, 0.2], thật = lớp 1: top-2 đúng | -ln q =", round(float(-np.log(0.3)), 3), "| ||p-y||² =", round(0.5**2 + 0.7**2 + 0.2**2, 2))
print("5) c = [[0,3,3],[1,0,1],[1,1,0]], p = [0.4, 0.35, 0.25]: p·c =", (np.array([0.4, 0.35, 0.25]) @ chi_phi).round(2), "-> chọn lớp", (np.array([0.4, 0.35, 0.25]) @ chi_phi).argmin())
print("8) C=1, n=420: λ = 1/(C n) =", round(1 / 420, 4))


# ---
# # 6. Tổng kết và mở rộng
# 
# ## 6.1. Diễn giải trọng số bằng odds ratio giữa hai lớp và giả định IIA
# 
# Từ $p_k/p_l=e^{s_k-s_l}$: tăng $x_j$ thêm 1 đơn vị nhân tỉ số odds của hai lớp với $e^{w_{jk}-w_{jl}}$ — chỉ có **hiệu** các cột mới có nghĩa. Với $\mathbf W$ trên thang đo gốc:

# In[ ]:


Wg = W_goc_sm
print("W (thang đo gốc):\n", Wg.round(3))
print(f"\n{'Thay đổi':<26}{'đạt : rớt':>12}{'giỏi : đạt':>12}{'giỏi : rớt':>12}")
for ten, j in (("Thêm 1 giờ tự học (x1)", 1), ("Thêm 1 buổi vắng (x2)", 2)):
    print(f"{ten:<26}{np.exp(Wg[j, 1] - Wg[j, 0]):>12.3f}{np.exp(Wg[j, 2] - Wg[j, 1]):>12.3f}{np.exp(Wg[j, 2] - Wg[j, 0]):>12.3f}")
print("(tài liệu: 4,43 / 3,48 / 15,4 và 0,348 / 0,722 / 0,251)")

x_a = np.array([[1, 6, 2.]]); x_b = np.array([[1, 7, 2.]])
pa, pb = softmax(x_a @ Wg)[0], softmax(x_b @ Wg)[0]
print(f"\nSV 6 giờ, vắng 2: p = {pa.round(3)}, odds đạt:rớt = {pa[1]/pa[0]:.1f}")
print(f"Học thêm 1 giờ:    p = {pb.round(3)}, odds đạt:rớt = {pb[1]/pb[0]:.1f} = {pa[1]/pa[0]:.1f} × {np.exp(Wg[1, 1] - Wg[1, 0]):.2f}")


# **Hai giả định ẩn.** (1) Softmax ép $\sum_kp_k=1$ nên các lớp **loại trừ nhau** (bài toán *đa nhãn* cần $K$ sigmoid độc lập). (2) **IIA** — tỉ lệ giữa hai lớp không phụ thuộc các lớp khác. Ví dụ "ô tô và xe buýt đỏ/xanh": người dùng thực tế chia đôi phần xe buýt ($\frac12,\frac14,\frac14$) nhưng softmax cho cùng điểm số hai xe buýt sẽ dự đoán $\frac13,\frac13,\frac13$:

# In[ ]:


s_a = np.array([[1., 2., 3.]]); pa3 = softmax(s_a)[0]
pa2 = softmax(s_a[:, :2])[0]
print(f"3 lớp: p = {pa3.round(3)}, p1/p0 = {pa3[1]/pa3[0]:.3f} | bỏ lớp thứ ba: p = {pa2.round(3)}, p1/p0 = {pa2[1]/pa2[0]:.3f}  (cùng bằng e = {np.e:.3f})")
oto_bus = softmax(np.array([[0., 0.]]))[0]; sau = softmax(np.array([[0., 0., 0.]]))[0]
print("Ô tô vs xe buýt đỏ:", oto_bus.round(3), "-> thêm xe buýt xanh cùng điểm số, softmax dự đoán:", sau.round(3), "(thực tế: [0.5, 0.25, 0.25])")


# ## 6.2. Softmax $K=2$ ≡ Logistic; one-vs-rest vs Softmax; nhiệt độ
# 
# **$K=2$:** gradient theo $\mathbf w_0$ là số đối của gradient theo $\mathbf w_1$ nên $\mathbf w_1-\mathbf w_0$ đi đúng quỹ đạo của Logistic Regression có tốc độ học **gấp đôi**:

# In[ ]:


m = np.isin(y_tr, [0, 2])                                          # chỉ giữ lớp rớt và lớp giỏi
Z2, t2 = Zb_tr[m], (y_tr[m] == 2).astype(float)
W2l, _ = huan_luyen(Z2, one_hot(t2.astype(int), 2), eta=0.5, so_epoch=300)     # Softmax, K = 2
w_lg = huan_luyen_logistic(Z2, t2, eta=1.0, so_epoch=300)                      # Logistic, eta gấp đôi
print("w1 - w0 (softmax) =", (W2l[:, 1] - W2l[:, 0]).round(4), "| w (logistic) =", w_lg.round(4))
print("max |khac biet| =", np.abs(W2l[:, 1] - W2l[:, 0] - w_lg).max(), "| w1 = -w0 = w/2 ở mọi bước:", np.allclose(W2l[:, 1], -W2l[:, 0]))


# In[ ]:


print("One-vs-rest (K Logistic độc lập) vs Softmax trên cùng dữ liệu:")
for k in range(3): print(f"  bộ phân loại {k}: J = {j_nhi_phan(Zb_tr, Y_tr[:, k], W_ovr[:, k]):.3f}")
tong = P_ovr.sum(axis=1)
print(f"tổng xác suất OvR: trung bình {tong.mean():.3f}, nhỏ nhất {tong.min():.3f}, lớn nhất {tong.max():.3f}")
print("tỉ lệ mẫu có tổng ngoài [0.8, 1.2]:", np.mean((tong < 0.8) | (tong > 1.2)).round(3), "| không lớp nào >= 0.5:", int((P_ovr < 0.5).all(axis=1).sum()),
      "| hơn một lớp >= 0.5:", int(((P_ovr >= 0.5).sum(axis=1) > 1).sum()))
P_ovr_n = P_ovr / tong[:, None]                                    # chuẩn hóa lại để có thể so log-loss
P_sm = softmax(Zb_te @ W_sm)
print(f"\n{'':<13}{'test acc':>10}{'macro-F1':>10}{'log-loss':>10}{'AUC':>8}")
for ten, Pm in [("one-vs-rest", P_ovr_n), ("softmax", P_sm)]:
    yh_ = Pm.argmax(axis=1)
    print(f"{ten:<13}{np.mean(yh_ == y_te):>10.4f}{f1_score(y_te, yh_, average='macro'):>10.4f}{log_loss(y_te, Pm):>10.4f}{roc_auc_score(y_te, Pm, multi_class='ovr'):>8.4f}")
print("(tài liệu: OvR 0,8056/0,7982/0,4772/0,9337; softmax 0,8500/0,8507/0,3536/0,9606)")


# ### "Nhiệt độ" và Perceptron nhiều lớp (giới hạn cứng)
# 
# Nhân điểm số với $c$: $c\to\infty$ ⇒ softmax thành **max cứng** (one-hot) và $\frac1cl_i\to\max_js_{ij}-s_{i,y_i}$ — chính là mất mát của Perceptron nhiều lớp; $c\to0$ ⇒ phân phối đều. Mẫu $\mathbf s=[0{,}5;1{,}5;2{,}5]$, lớp thật là 1:

# In[ ]:


s_v = np.array([[0.5, 1.5, 2.5]]); lop_that = 1
print(f"{'c':>5}  {'p = softmax(c·s)':<26}{'l_i / c':>9}")
for c in (1, 2, 5, 20):
    pc = softmax(c * s_v)[0]
    print(f"{c:>5}  {str(pc.round(3)):<26}{-np.log(pc[lop_that]) / c:>9.3f}")
print(f"{'→∞':>5}  one-hot của lớp 2          {s_v.max() - s_v[0, lop_that]:>9.3f}   (= max s - s_đúng = 2,5 - 1,5)")
print("c → 0 : p →", softmax(1e-6 * s_v)[0].round(4), "(phân phối đều)")

# Hình 19a và 19b
cs = np.linspace(0.05, 6, 120); pc = np.array([softmax(c * s_v)[0] for c in cs])
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for k in range(3): ax[0].plot(cs, pc[:, k], color=MAU_LOP[k], label=f"p{k}")
ax[0].set(xlabel="hệ số c (nghịch đảo nhiệt độ)", ylabel="xác suất", title="(a) c lớn: dồn về lớp có điểm cao nhất"); ax[0].legend()
ax[1].hist(P_ovr.sum(axis=1), bins=25, alpha=0.7, label="one-vs-rest"); ax[1].axvline(1, color="k", ls="--", label="Softmax: luôn = 1")
ax[1].set(xlabel="p0 + p1 + p2", title="(b) tổng xác suất"); ax[1].legend(); plt.tight_layout(); plt.show()


# ## 6.3. Từ Softmax Regression đến mạng nơ-ron nhiều lớp
# 
# Softmax Regression là **mạng nơ-ron không có lớp ẩn**. Thêm một lớp ẩn $m$ nơ-ron:
# $$\mathbf H=\tanh(\bar{\mathbf X}\mathbf W_1),\quad\mathbf S=\bar{\mathbf H}\mathbf W_2,\quad\mathbf P=\mathrm{softmax}(\mathbf S).$$
# Lớp ra **chính là Softmax Regression** chạy trên đặc trưng $\mathbf H$ do mạng tự học. **Lan truyền ngược:** $\mathbf D_2=\frac1n(\mathbf P-\mathbf Y)$ (đúng kết quả mục 3.3), $\nabla_{\mathbf W_2}J=\bar{\mathbf H}^T\mathbf D_2$, $\mathbf D_1=\big(\mathbf D_2\mathbf V^T\big)\odot(1-\mathbf H\odot\mathbf H)$ với $\mathbf V$ là $\mathbf W_2$ bỏ hàng chệch, $\nabla_{\mathbf W_1}J=\bar{\mathbf X}^T\mathbf D_1$.

# In[ ]:


def huan_luyen_mang(X, Y, W1, W2, eta=0.5, so_epoch=5000):
    # Mạng nơ-ron một lớp ẩn (tanh), đầu ra softmax, huấn luyện bằng lan truyền ngược (batch).
    n = len(X); Xb = them_bias(X); lich_su = []
    W1, W2 = W1.copy(), W2.copy()
    for ep in range(so_epoch):
        H = np.tanh(Xb @ W1)                             # lớp ẩn
        Hb = them_bias(H)
        P = softmax(Hb @ W2)                             # xác suất đầu ra
        lich_su.append(-np.mean(np.sum(Y * np.log(np.clip(P, 1e-12, 1)), axis=1)))
        D2 = (P - Y) / n                                 # gradient theo điểm số đầu ra: (P - Y)/n
        g_W2 = Hb.T @ D2                                 # lớp ra: dạng giống hệt Softmax Regression
        dH = (D2 @ W2[1:].T) * (1 - H ** 2)              # lan truyền ngược về lớp ẩn (tanh' = 1 - tanh^2)
        g_W1 = Xb.T @ dH
        W2 = W2 - eta * g_W2
        W1 = W1 - eta * g_W1
    return W1, W2, lich_su

def mang_softmax(X, Y, so_an=8, eta=0.5, so_epoch=5000, seed=0):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 1, (X.shape[1] + 1, so_an))       # (d+1) x m: mỗi cột là một nơ-ron ẩn
    W2 = rng.normal(0, 1, (so_an + 1, Y.shape[1]))       # (m+1) x K: lớp ra chính là Softmax Regression
    return huan_luyen_mang(X, Y, W1, W2, eta, so_epoch)

def du_doan_mang(X, W1, W2):
    return softmax(them_bias(np.tanh(them_bias(X) @ W1)) @ W2)


# > ### 🔍 Giải thích mã (lan truyền ngược)
# > **`D2 = (P - Y) / n`** — gradient của $J$ theo ma trận điểm số của lớp ra. Đây *chính xác* là kết quả đã chứng minh ở mục 3.3 ($\partial l_i/\partial\mathbf s_i=\mathbf p_i-\mathbf y_i$, chia $n$ vì $J$ là trung bình). Đó là "bước đầu của lan truyền ngược" trong mọi mạng phân loại.
# >
# > **`g_W2 = Hb.T @ D2`** — gradient của lớp ra có dạng như Softmax Regression nhưng với $\bar{\mathbf H}$ (đặc trưng ẩn, đã thêm cột 1) thay cho $\mathbf X$.
# >
# > **`dH = (D2 @ W2[1:].T) * (1 - H ** 2)`** — truyền sai số ngược về lớp ẩn: `W2[1:]` là $\mathbf V$ ($\mathbf W_2$ **bỏ hàng chệch**, vì hàng chệch không nối với nơ-ron ẩn nào); `D2 @ V.T` ($n\times K$ nhân $K\times m$) chuyển sai số của $K$ đầu ra về $m$ nơ-ron ẩn; `* (1 - H**2)` là nhân **từng phần tử** với đạo hàm $\tanh'(z)=1-\tanh^2z$ (quy tắc đạo hàm hàm hợp). `g_W1 = Xb.T @ dH` cho gradient của $\mathbf W_1$ với đúng kích thước $(d{+}1)\times m$.
# >
# > **`np.clip(P, 1e-12, 1)`** chỉ để tránh `log(0)` khi tính $J$ ghi lại (không ảnh hưởng gradient).

# In[ ]:


# Dữ liệu ba vòng tròn, chuẩn hóa; kiểm tra gradient lan truyền ngược bằng đạo hàm số
Za, Zb_, _, _ = chuan_hoa(Xc_tr, Xc_te); Ya = one_hot(yc_tr, 3)

def J_mang(X, Y, W1, W2):
    P = softmax(them_bias(np.tanh(them_bias(X) @ W1)) @ W2); return -np.mean(np.sum(Y * np.log(P), axis=1))
def gradient_mang(X, Y, W1, W2):
    n = len(X); Xb = them_bias(X); H = np.tanh(Xb @ W1); Hb = them_bias(H)
    D2 = (softmax(Hb @ W2) - Y) / n
    return Xb.T @ ((D2 @ W2[1:].T) * (1 - H ** 2)), Hb.T @ D2

rng = np.random.default_rng(1); Xs, Ys = Za[:20], Ya[:20]
W1 = rng.normal(0, 1, (3, 4)); W2 = rng.normal(0, 1, (5, 3)); e = 1e-6; sai_max = 0
g1, g2 = gradient_mang(Xs, Ys, W1, W2)
for A, g in ((W1, g1), (W2, g2)):
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i, j] += e; jp = J_mang(Xs, Ys, W1, W2); A[i, j] -= 2 * e; jm = J_mang(Xs, Ys, W1, W2); A[i, j] += e
            sai_max = max(sai_max, abs((jp - jm) / (2 * e) - g[i, j]))
print("gradient check max err", sai_max, " (tài liệu: ≈ 2,5e-10)")


# In[ ]:


ket_qua_mang = {}
print(f"{'m':>3} {'tham số':>8}  test acc (5 khởi tạo)                     J train (5 khởi tạo)")
for m_an in (2, 4, 8):
    accs, Js, Ws = [], [], []
    for seed in range(5):
        W1, W2, ls = mang_softmax(Za, Ya, so_an=m_an, eta=0.5, so_epoch=5000, seed=seed)
        accs.append(np.mean(du_doan_mang(Zb_, W1, W2).argmax(axis=1) == yc_te)); Js.append(ls[-1]); Ws.append((W1, W2, ls))
    ket_qua_mang[m_an] = Ws
    print(f"{m_an:>3} {3 * m_an + (m_an + 1) * 3:>8}  {str(np.round(accs, 3)):<40}  {np.round(Js, 4)}")
print("(tài liệu: m=2: 62,8–70,0%, J≈0,65 | m=4: 96,1–98,9%, J≈0,06 | m=8: 96,1–99,4%, J≈0,02)")
print("-> Hàm mất mát của mạng KHÔNG còn lồi: các khởi tạo khác nhau cho J cuối và độ chính xác khác nhau (khác Softmax Regression).")


# In[ ]:


# Hình 20: vùng quyết định của mạng 2-8-3 và đường J theo epoch
W1b, W2b, _ = ket_qua_mang[8][0]
g = np.linspace(-6.5, 6.5, 260); U, V = np.meshgrid(g, g)
Pm = du_doan_mang((np.c_[U.ravel(), V.ravel()] - Xc_tr.mean(0)) / Xc_tr.std(0), W1b, W2b)
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.5))
ax[0].contourf(U, V, Pm.argmax(axis=1).reshape(U.shape), levels=[-0.5, 0.5, 1.5, 2.5], colors=MAU_LOP, alpha=0.25)
for k in range(3): ax[0].scatter(*Xc_te[yc_te == k].T, s=10, color=MAU_LOP[k], edgecolor="k", lw=0.3)
ax[0].set(title="(b) vùng quyết định của mạng 2-8-3 (biên cong do mạng tự học)", xlabel="x1", ylabel="x2", aspect="equal")
for m_an, mau_ in zip((2, 4, 8), ("tab:red", "tab:orange", "tab:green")):
    for sd, (_, _, ls) in enumerate(ket_qua_mang[m_an]): ax[1].plot(ls, color=mau_, alpha=0.7, label=f"m = {m_an}" if sd == 0 else None)
ax[1].axhline(np.log(3), color="k", ls=":", label="ln 3"); ax[1].set(xlabel="epoch", ylabel="J", yscale="log", title="(c) J theo epoch, 5 khởi tạo mỗi cỡ"); ax[1].legend()
plt.suptitle("Hình 20", y=1.02); plt.tight_layout(); plt.show()


# ### Vì sao mạng cần khởi tạo **ngẫu nhiên** (khác Softmax Regression)?

# In[ ]:


W1_0, W2_0 = np.zeros((3, 4)), np.zeros((5, 3))
_, _, ls0 = huan_luyen_mang(Za, Ya, W1_0, W2_0, so_epoch=5000)
print(f"Khởi tạo W1 = W2 = 0: sau 5000 epoch J = {ls0[-1]:.4f} (= ln 3 = {np.log(3):.4f}); nơ-ron ẩn không bao giờ học (V = 0 nên ∇W1 = 0)")
Wa, Wb, _ = huan_luyen_mang(Za, Ya, W1_0, W2_0, so_epoch=5000)
print("   W1 sau huấn luyện toàn số 0:", np.allclose(Wa, 0))

# Khởi tạo hằng số khác 0: các nơ-ron ẩn giống hệt nhau mãi (đối xứng không bị phá)
W1_h, W2_h = np.full((3, 4), 0.3), np.full((5, 3), 0.2) + np.array([[0, 0.1, -0.1]])
Wa, Wb, lsh = huan_luyen_mang(Za, Ya, W1_h, W2_h, so_epoch=3000)
print(f"Khởi tạo hằng số: các cột của W1 giống hệt nhau? {np.allclose(Wa, Wa[:, [0]])} | J cuối = {lsh[-1]:.4f} (như chỉ có MỘT nơ-ron ẩn)")
print("-> Ở Softmax Regression, W = 0 vẫn ổn vì J lồi và các cột của Y khác nhau nên các cột w_k phân hóa ngay từ bước đầu.")


# **Học sâu = cùng công thức, nhiều lớp hơn:** softmax là lớp đầu ra chuẩn (điểm số cuối gọi là *logits*), gradient của lớp ra vẫn là $\mathbf P-\mathbf Y$; L2 gọi là *weight decay*; mini-batch SGD/Adam là cách huấn luyện chuẩn; ReLU thay tanh; nhiệt độ dùng để *hiệu chuẩn lại* các mạng quá tự tin.
# 
# | Mô hình | Điểm số | Đầu ra | Mất mát | Gradient theo điểm số |
# | --- | --- | --- | --- | --- |
# | Hồi quy tuyến tính | $\mathbf{Xw}$ | chính điểm số | bình phương sai số | $\propto\hat y-y$ |
# | Logistic Regression | $\mathbf{Xw}$ | $\sigma(s)$ | cross-entropy nhị phân | $\sigma(s)-y$ |
# | Softmax Regression | $\mathbf{XW}$ | softmax | cross-entropy nhiều lớp | $\mathbf P-\mathbf Y$ |
# | Mạng nơ-ron (đầu ra softmax) | $\bar{\mathbf H}\mathbf W_L$ ($\mathbf H$ do mạng học) | softmax | cross-entropy nhiều lớp | $\mathbf P-\mathbf Y$, rồi lan truyền ngược |
# 
# ## 6.4. Bài tập thực hành — kiểm chứng đáp án bằng mã
# 
# ### Phần A: tính tay và lý thuyết

# In[ ]:


print("A1) s = [2, 0, -1]: p =", softmax(np.array([[2., 0, -1]]))[0].round(3), "| lớp thật 2: mất mát =", round(float(-np.log(softmax(np.array([[2., 0, -1]]))[0, 2])), 3),
      "| p - y =", (softmax(np.array([[2., 0, -1]]))[0] - np.array([0, 0, 1])).round(3))

# A2: hai mẫu một đặc trưng, y = [0, 2], K = 3, W0 = 0, eta = 1
XA = np.array([[1, 0], [1, 2.]]); YA = one_hot(np.array([0, 2]), 3)
print("\nA2) J(W0) =", round(float(mat_mat(XA, YA, np.zeros((2, 3)))), 4), "\n   ∇J =\n", gradient(XA, YA, np.zeros((2, 3))).round(4))
W1A = -gradient(XA, YA, np.zeros((2, 3)))
print("   W1 = W0 - η∇J =\n", W1A.round(4), "\n   P1 =\n", softmax(XA @ W1A).round(3), "| J(W1) =", round(float(mat_mat(XA, YA, W1A)), 3), "(tài liệu: 0,577)")

# A4: W* + v 1^T
Wst = np.array([[1, -1, 0], [2, 0, -2.]]); vv = np.array([[1.], [3.]]); xA4 = np.array([[1., 1.]])
print("\nA4) s* =", (xA4 @ Wst)[0], "| s' =", (xA4 @ (Wst + vv))[0], "| p* =", softmax(xA4 @ Wst)[0].round(3), "| p' =", softmax(xA4 @ (Wst + vv))[0].round(3), "-> như nhau")

# A5, A7, A8
print("\nA5) odds giỏi:đạt khi thêm 1 buổi vắng ×", round(float(np.exp(Wg[2, 2] - Wg[2, 1])), 3), "| odds đạt:rớt khi thêm 2 giờ ×", round(float(np.exp(2 * (Wg[1, 1] - Wg[1, 0]))), 1), "(tài liệu: 0,722 và 19,6)")
print("A7) c=1:", softmax(np.array([[1., 2, 3]]))[0].round(3), "| c=2:", softmax(2 * np.array([[1., 2, 3]]))[0].round(3))
print("A8) d=4, m=10, K=3: tham số =", (4 + 1) * 10 + (10 + 1) * 3, "(so với Softmax:", (4 + 1) * 3, ") | ∇W1: 5×10 | ∇W2: 11×3")


# ### Phần B: lập trình (Iris và chữ số viết tay)

# In[ ]:


from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split

# B1: Softmax Regression từ đầu trên Iris (4 đặc trưng, 3 lớp)
Xi, yi = load_iris(return_X_y=True)
Xi_tr, Xi_te, yi_tr, yi_te = train_test_split(Xi, yi, test_size=0.3, stratify=yi, random_state=0)
Zi_tr, Zi_te, _, _ = chuan_hoa(Xi_tr, Xi_te); Zbi_tr, Zbi_te = them_bias(Zi_tr), them_bias(Zi_te); Yi_tr = one_hot(yi_tr, 3)
Wi, lsi = huan_luyen(Zbi_tr, Yi_tr, eta=0.5, so_epoch=2000)
print(f"B1) Iris: J ban dau = {lsi[0]:.4f} (= ln 3) | J cuoi = {lsi[-1]:.4f} | acc train = {np.mean(du_doan(Zbi_tr, Wi) == yi_tr):.3f} | acc test = {np.mean(du_doan(Zbi_te, Wi) == yi_te):.3f}")
print("    (tài liệu: J cuối ≈ 0,0634; train ≈ 97,1%; test = 100%)")
Hi = hessian(Zbi_tr, Wi); li = np.linalg.eigvalsh(Hi)
print(f"B2) Hessian {Hi.shape}: {(np.abs(li) < 1e-9).sum()} trị riêng ≈ 0 (kỳ vọng d+1 = 5) | hạng = {np.linalg.matrix_rank(Hi, tol=1e-9)} (kỳ vọng (K-1)(d+1) = 10)")


# In[ ]:


# B5/B6: chữ số viết tay (64 đặc trưng, 10 lớp): Softmax vs one-vs-rest
Xd, yd = load_digits(return_X_y=True)
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, yd, test_size=0.3, stratify=yd, random_state=0)
sm = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=5000)).fit(Xd_tr, yd_tr)
ov = make_pipeline(StandardScaler(), OneVsRestClassifier(LogisticRegression(C=1, max_iter=5000))).fit(Xd_tr, yd_tr)
for ten, mo in (("Softmax", sm), ("One-vs-rest", ov)):
    pr = mo.predict_proba(Xd_te); yh_ = pr.argmax(axis=1)
    print(f"{ten:<12} acc = {np.mean(yh_ == yd_te):.4f} | macro-F1 = {f1_score(yd_te, yh_, average='macro'):.4f} | log-loss = {log_loss(yd_te, pr):.4f}"
          + (f" | top-3 = {top_k_accuracy_score(yd_te, pr, k=3):.3f} | AUC OvR = {roc_auc_score(yd_te, pr, multi_class='ovr'):.4f}" if ten == "Softmax" else ""))
print("(tài liệu: Softmax ≈ 97,2% / 0,972 / 0,102; one-vs-rest ≈ 96,9% / 0,969 / 0,138)")
C10 = confusion_matrix(yd_te, sm.predict(Xd_te)); np.fill_diagonal(C10, 0)
i_, j_ = np.unravel_index(C10.argmax(), C10.shape)
print(f"Cặp bị nhầm nhiều nhất: chữ số {i_} bị đoán thành {j_} ({C10[i_, j_]} mẫu)")


# > ### 🔍 Giải thích mã
# > **`np.fill_diagonal(C10, 0)`** — xóa đường chéo (các dự đoán đúng) của ma trận nhầm lẫn $10\times10$ để chỉ còn các ô *nhầm lẫn*. **`np.unravel_index(C10.argmax(), C10.shape)`** — `argmax()` trả về vị trí trong mảng *duỗi phẳng*; `unravel_index` đổi nó thành cặp (hàng, cột) = (lớp thật, lớp dự đoán) có số nhầm lẫn lớn nhất.

# ---
# # Tổng kết
# 
# | Mục | Câu hỏi chính | Kết quả cần nhớ |
# | --- | --- | --- |
# | 1 | Vì sao Logistic chưa đủ khi $K>2$? | Nhãn số làm hồi quy phụ thuộc cách đánh số; one-vs-rest cho xác suất không cộng bằng 1 và che khuất lớp giữa; cần nhãn **one-hot** và mô hình xác suất chung |
# | 2 | Mô hình là gì? | $\mathbf S=\mathbf{XW}$, $\mathbf P=\mathrm{softmax}(\mathbf S)$; $K=2$ ⇒ sigmoid; vùng quyết định lồi; $\mathbf W$ dư thừa tham số |
# | 3 | Huấn luyện thế nào? | $J=-\frac1n\sum\ln q_i$; $\nabla_{\mathbf W}J=\frac1n\mathbf X^T(\mathbf P-\mathbf Y)$; $J(\mathbf 0)=\ln K$; tổng các cột của $\mathbf W$ không đổi |
# | 4 | Mô hình học ra sao? | Kéo lớp đúng, đẩy lớp sai theo $p_{ik}$; $J$ lồi nhưng Hessian **suy biến** ($d{+}1$ hướng phẳng); phân tách hoàn hảo ⇒ $\mathbf W\to\infty$ (cần L2); vùng quyết định luôn lồi |
# | 5 | Mô hình tốt đến đâu? | Ma trận nhầm lẫn $K\times K$, F1 theo lớp, ma trận chi phí, log-loss, ECE, AUC OvR; 600 sinh viên: 85,0% kiểm tra, log-loss 0,354 |
# | 6 | Mở rộng đi đâu? | odds ratio giữa hai lớp, giả định IIA; Softmax > OvR khi lớp loại trừ nhau; Softmax là lớp đầu ra của **mạng nơ-ron** |
# 
# **Bài học chính**
# 
# 1. Với $K>2$, dùng **một** phân phối xác suất trên $K$ lớp (softmax) thay vì ghép $K$ bộ nhị phân độc lập — các lớp *cạnh tranh* và xác suất cộng bằng 1.
# 2. Mọi thứ "gọn" như Logistic: gradient theo điểm số là $\mathbf P-\mathbf Y$, cập nhật $\mathbf W\leftarrow\mathbf W-\frac\eta n\mathbf X^T(\mathbf P-\mathbf Y)$; chỉ đổi vector thành ma trận và sigmoid thành softmax.
# 3. Softmax **bất biến khi dịch** ⇒ tính ổn định số học (trừ max), tham số dư thừa, Hessian suy biến, $\mathbf P$ duy nhất còn $\mathbf W$ thì không.
# 4. Chuẩn hóa đặc trưng (thống kê chỉ từ tập huấn luyện) giảm số điều kiện từ ~4880 xuống ~39 và cho phép dùng $\eta$ lớn hơn nhiều; L2 giữ $\mathbf W$ hữu hạn khi dữ liệu phân tách hoàn hảo.
# 5. Giới hạn tuyến tính (vùng lồi) → thêm đặc trưng đa thức hoặc để **mạng nơ-ron tự học đặc trưng**; đổi lại mất tính lồi và cần khởi tạo ngẫu nhiên.
# 6. Đánh giá bài bản: accuracy chưa đủ — xem ma trận nhầm lẫn, chỉ số theo lớp (macro khi lớp lệch), log-loss, hiệu chuẩn, AUC; chọn siêu tham số bằng kiểm định chéo trong pipeline và chỉ chạm tập kiểm tra một lần.
