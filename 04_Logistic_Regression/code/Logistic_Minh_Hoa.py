#!/usr/bin/env python
# coding: utf-8

# # MÔ HÌNH LOGISTIC REGRESSION — Notebook minh họa
# 
# Notebook minh họa bằng mã Python toàn bộ nội dung tài liệu *Mô hình Logistic Regression*, theo mạch **từ Perceptron → xác suất → mô hình → huấn luyện → cơ chế học → đánh giá**:
# 
# | Mục | Nội dung | Phần minh họa |
# | --- | --- | --- |
# | 1 | Từ Perceptron đến phân loại bằng xác suất | Hạn chế của Perceptron, odds/log-odds, hàm sigmoid, hồi quy tuyến tính trên nhãn 0/1 |
# | 2 | Mô hình Logistic Regression | Tính chất sigmoid, ngưỡng tùy chọn, đường đồng mức xác suất, odds ratio |
# | 3 | Huấn luyện | MSE vs cross-entropy, hợp lý cực đại, gradient, ví dụ tính tay, tốc độ học, Batch/SGD/Mini-batch |
# | 4 | Cơ chế học | Đóng góp của mỗi mẫu, chuẩn hóa, Hessian và tính lồi, dữ liệu phân tách hoàn hảo, L2, XOR, đa thức |
# | 5 | Đánh giá | Đường cong học, overfitting, ngưỡng, ROC/AUC, hiệu chuẩn, scikit-learn, kiểm định chéo |
# 
# **Quy ước ký hiệu:** $\mathbf x_i$ là vector hàng (phần tử đầu bằng 1), $\mathbf w$ là vector cột, nhãn $y_i\in\{0,1\}$ (*khác* bài Perceptron dùng $\{-1,+1\}$), điểm số $s_i=\mathbf x_i\mathbf w$ (chính là **log-odds**), xác suất $p_i=\sigma(s_i)$.
# 
# > **Logistic Regression = hồi quy tuyến tính (tính điểm số) + hàm sigmoid (đưa ra xác suất).**
# 
# Những đoạn mã khó được kèm khung **🔍 Giải thích mã** ngay bên dưới.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
plt.rcParams.update({
    "figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})

def sigmoid(s):
    return 0.5 * (1.0 + np.tanh(0.5 * s))      # xem giải thích ở mục 3.7


# ---
# # 1. Từ Perceptron đến bài toán phân loại bằng xác suất
# 
# ## 1.1. Hạn chế của Perceptron: chỉ cho nhãn cứng
# 
# Với $\mathbf w=[-2{,}5;\,1;\,-1]^T$, hai sinh viên A (9 giờ, vắng 0) và B (5 giờ, vắng 2) đều có $s\ge0$ nên Perceptron trả lời giống hệt nhau: "đậu". Nhưng A nằm sâu trong vùng đậu còn B sát đường biên:

# In[ ]:


w_A = np.array([-2.5, 1.0, -1.0])
xA, xB = np.array([1, 9, 0]), np.array([1, 5, 2])
for ten, x in [("A (9 giờ, vắng 0)", xA), ("B (5 giờ, vắng 2)", xB)]:
    s = x @ w_A
    print(f"Sinh viên {ten}: s = {s:+.1f} -> Perceptron: {'đậu' if s >= 0 else 'rớt'} | Logistic: p = σ(s) = {sigmoid(s):.3f}")
print("-> Perceptron không phân biệt được A và B; xác suất thì có (0.998 so với 0.622).")


# ## 1.2. Bài toán theo xác suất, nhãn $\{0,1\}$ và dữ liệu xuyên suốt
# 
# Ví dụ 6 sinh viên (ma trận $\mathbf X$ không đổi so với bài Perceptron, chỉ nhãn đổi sang 0/1) và bộ dữ liệu mô phỏng 400 sinh viên (mỗi lớp 200, hai lớp **chồng lấn**):

# In[ ]:


X_sv = np.array([[2, 4], [3, 3], [4, 5], [6, 1], [8, 2], [9, 0]], dtype=float)
y_sv = np.array([0, 0, 0, 1, 1, 1])
Xb_sv = np.hstack([np.ones((6, 1)), X_sv])
y_pm = 2 * y_sv - 1                                # y± = 2y - 1
print("y± = ", y_pm, "-> y = (y± + 1)/2 =", (y_pm + 1) // 2)

# ---- 400 sinh viên mô phỏng (dùng xuyên suốt notebook) ----
rng = np.random.default_rng(42)
n_lop = 200
dau = np.c_[rng.normal(8, 2.2, n_lop), rng.normal(1.5, 1.2, n_lop)]   # lớp 1 (đậu)
rot = np.c_[rng.normal(4, 2.2, n_lop), rng.normal(3.5, 1.5, n_lop)]   # lớp 0 (rớt)
X = np.clip(np.vstack([dau, rot]), 0, None)
y = np.r_[np.ones(n_lop), np.zeros(n_lop)].astype(int)

def chia_train_test(X, y, ti_le_test=0.3, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = int(len(y) * ti_le_test)
    return X[idx[n_test:]], X[idx[:n_test]], y[idx[n_test:]], y[idx[:n_test]]

def them_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

X_tr, X_te, y_tr, y_te = chia_train_test(X, y, ti_le_test=0.3, seed=0)
Xb_tr, Xb_te = them_bias(X_tr), them_bias(X_te)
print(f"\nHuấn luyện: {len(y_tr)} mẫu ({y_tr.sum()} đậu) | Kiểm tra: {len(y_te)} mẫu ({y_te.sum()} đậu)")

# ---- Tỉ lệ đậu theo nhóm số giờ học (chỉ dùng x1, cả 400 sinh viên) ----
bien = [0, 2, 4, 6, 8, 10, np.inf]
print(f"\n{'Nhóm giờ học':<14}{'Số SV':>7}{'Số đậu':>8}{'Tỉ lệ đậu':>11}")
nhom_x, nhom_ti_le = [], []
for lo, hi in zip(bien[:-1], bien[1:]):
    m = (X[:, 0] >= lo) & (X[:, 0] < hi)
    ten = f"[{lo},{hi})" if np.isfinite(hi) else f">={lo}"
    print(f"{ten:<14}{m.sum():>7}{y[m].sum():>8}{y[m].mean():>10.1%}")
    nhom_x.append(X[m, 0].mean()); nhom_ti_le.append(y[m].mean())


# ## 1.3. Ý tưởng: biến điểm số thành xác suất
# 
# ### Thử hồi quy tuyến tính trên nhãn 0/1 (Hình 1)
# Hồi quy tuyến tính cho "xác suất" ngoài $[0,1]$ và không bám dáng chữ S của dữ liệu:

# In[ ]:


from scipy.optimize import minimize

x1 = X[:, 0]
A = np.column_stack([np.ones(len(x1)), x1])
w_lin = np.linalg.solve(A.T @ A, A.T @ y)                  # hồi quy tuyến tính (phương trình chuẩn)
print(f"Hồi quy tuyến tính: p = {w_lin[0]:.3f} + {w_lin[1]:.3f} x1   (tài liệu: -0,222 + 0,122 x1)")
print(f"  x1 = 0 giờ : p = {w_lin[0]:.3f}  (< 0)")
print(f"  x1 = 12 giờ: p = {w_lin[0] + w_lin[1]*12:.3f}  (> 1)")

# Logistic Regression chỉ với đặc trưng x1: tối thiểu cross-entropy bằng L-BFGS
def ce_1d(w):
    p = np.clip(sigmoid(A @ w), 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
w_log1 = minimize(ce_1d, np.zeros(2), method="L-BFGS-B").x

xs = np.linspace(0, 14, 200)
fig, ax = plt.subplots(figsize=(8, 4.2))
rng_j = np.random.default_rng(0)
ax.scatter(x1, y + rng_j.uniform(-0.04, 0.04, len(y)), s=8, alpha=0.35, color="gray", label="sinh viên (rung nhẹ)")
ax.plot(nhom_x, nhom_ti_le, "kD", ms=7, label="tỉ lệ đậu theo nhóm")
ax.plot(xs, w_lin[0] + w_lin[1] * xs, "r--", label="hồi quy tuyến tính")
ax.plot(xs, sigmoid(w_log1[0] + w_log1[1] * xs), "b", label="sigmoid (logistic)")
ax.axhline(0, color="k", lw=0.6); ax.axhline(1, color="k", lw=0.6)
ax.set(xlabel="giờ tự học x1", ylabel="xác suất đậu", ylim=(-0.4, 1.4), title="Hình 1: đường thẳng vượt ra ngoài [0,1], đường cong chữ S bám sát dữ liệu")
ax.legend(fontsize=8); plt.show()


# ### Odds và log-odds
# $\text{odds}=\dfrac p{1-p}\in(0,\infty)$ và $\text{logit}(p)=\ln\dfrac p{1-p}\in(-\infty,+\infty)$ — đúng bằng miền giá trị của điểm số. **Giả thiết của Logistic Regression:** log-odds là hàm tuyến tính của đặc trưng, $\ln\frac p{1-p}=\mathbf{xw}$, và giải ra $p=\sigma(s)=\dfrac1{1+e^{-s}}$.

# In[ ]:


print(f"{'p':>5}{'odds':>9}{'log-odds':>10}")
for p in (0.1, 0.2, 0.5, 0.8, 0.9):
    print(f"{p:>5}{p/(1-p):>9.3f}{np.log(p/(1-p)):>10.3f}")

print("\nσ(s) tại một số điểm nhớ:")
ss = np.array([-4, -2, -1, 0, 0.5, 1, 2, 4])
print("s     :", ss); print("σ(s)  :", np.round(sigmoid(ss), 3))
print("Đối xứng σ(-2) = 1 - σ(2):", np.isclose(sigmoid(-2), 1 - sigmoid(2)))

# Kiểm tra: sigmoid là hàm ngược của logit
p = np.array([0.1, 0.5, 0.9]); print("σ(logit(p)) = p:", np.allclose(sigmoid(np.log(p / (1 - p))), p))


# ### Ba cách xử lý cùng một điểm số (Hình 2) và vai trò của độ lớn $\mathbf w$ (Hình 3)

# In[ ]:


s = np.linspace(-6, 6, 600)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(s, s, label="hồi quy tuyến tính (giữ nguyên)")
ax[0].step(s, (s >= 0).astype(float), where="post", label="Perceptron (nhãn 0/1)")
ax[0].plot(s, sigmoid(s), label="Logistic: σ(s)")
ax[0].set(ylim=(-0.5, 1.5), xlabel="điểm số s = xw", title="Hình 2: ba hàm kích hoạt"); ax[0].legend(fontsize=8)

for c in (0.25, 0.5, 1, 2, 4):
    ax[1].plot(s, sigmoid(c * s), label=f"c = {c}")
ax[1].set(xlabel="s", title="Hình 3: σ(c·s) — c lớn thì dốc hơn, quả quyết hơn"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()

print("Sinh viên B (s = 0.5):")
for c, ten in [(0.5, "0,5 w"), (1, "w"), (2, "2 w")]:
    print(f"  {ten:<6} s = {c*0.5:.2f}  ->  σ(s) = {sigmoid(c*0.5):.3f}")


# ### Ví dụ số với 6 sinh viên: điểm số, xác suất và nhãn
# Đường biên vẫn là $\mathbf{xw}=0$ vì $p\ge0{,}5\Leftrightarrow s\ge0$.

# In[ ]:


s6 = Xb_sv @ w_A
p6 = sigmoid(s6)
print("s =", s6)
print("p =", np.round(p6, 3))
print("\nSV |    s    |   p   | y_hat | y")
for i in range(6):
    print(f"{i+1:>2} | {s6[i]:>6.1f}  | {p6[i]:.3f} |   {int(p6[i] >= 0.5)}   | {y_sv[i]}")

print("\n--- Kiểm tra nhanh mục 1 ---")
x_q = np.array([1, 6, 3]); s_q = x_q @ w_A
print(f"1) x=[1,6,3]: s = {s_q}, p = {sigmoid(s_q):.3f}, y_hat = {int(sigmoid(s_q) >= 0.5)}")
for p in (0.75, 0.25): print(f"2) p = {p}: odds = {p/(1-p):.3f}, log-odds = {np.log(p/(1-p)):+.3f}")
print(f"3) log-odds = -2 -> p = {sigmoid(-2):.3f}")


# ---
# # 2. Mô hình Logistic Regression
# 
# ## 2.2. Điểm số là log-odds
# 
# Hệ số chệch $w_0$: $\sigma(w_0)$ là **xác suất nền**. Với $w_0=-2{,}5$: $\sigma(-2{,}5)\approx7{,}6\%$. Các đặc trưng **cộng** vào log-odds nhưng **nhân** vào odds: $\text{odds}=e^{w_0}\cdot e^{w_1x_1}\cdots$

# In[ ]:


print(f"Xác suất nền σ(w0 = -2.5) = {sigmoid(-2.5):.3f}")
x = np.array([1, 5, 2.0])
odds_tich = np.exp(w_A[0]) * np.exp(w_A[1] * x[1]) * np.exp(w_A[2] * x[2])
print(f"odds = e^s = {np.exp(x @ w_A):.4f} | tích các thừa số = {odds_tich:.4f}")


# ## 2.3. Tính chất của sigmoid
# 
# Đối xứng $\sigma(-s)=1-\sigma(s)$; đạo hàm $\sigma'(s)=\sigma(s)(1-\sigma(s))\le0{,}25$ (lớn nhất tại $s=0$). Kiểm chứng đạo hàm bằng sai phân hữu hạn và vẽ Hình 5:

# In[ ]:


s = np.linspace(-8, 8, 400)
d_giai_tich = sigmoid(s) * (1 - sigmoid(s))
h = 1e-5
d_so = (sigmoid(s + h) - sigmoid(s - h)) / (2 * h)
print("Sai khác tối đa giữa đạo hàm giải tích và đạo hàm số:", np.max(np.abs(d_giai_tich - d_so)))
print("σ'(0) =", sigmoid(0) * (1 - sigmoid(0)), "| σ'(2) =", round(float(sigmoid(2) * (1 - sigmoid(2))), 3))

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(s, sigmoid(s)); ax[0].axhline(0.5, color="gray", ls=":"); ax[0].set(title="(a) σ(s)", xlabel="s")
ax[1].plot(s, d_giai_tich, color="tab:red"); ax[1].set(title="(b) σ'(s) = σ(s)(1−σ(s)), lớn nhất = 0,25 tại s = 0", xlabel="s")
plt.tight_layout(); plt.show()


# **Xác suất của nhãn thật.** Hai trường hợp gộp làm một công thức Bernoulli: $P(y\mid\mathbf x)=p^{y}(1-p)^{1-y}$. Đặt $q_i$ là xác suất mô hình gán cho nhãn *thật* của mẫu $i$:

# In[ ]:


q = p6 ** y_sv * (1 - p6) ** (1 - y_sv)
print("q =", np.round(q, 3))
print("Tích các q_i (xác suất mô hình gán cho toàn bộ nhãn quan sát) =", round(float(np.prod(q)), 3), "(tài liệu: ≈ 0,795)")


# ### Ngưỡng quyết định tùy chọn
# 
# Với ngưỡng $\tau$: $\hat y=1\Leftrightarrow p\ge\tau\Leftrightarrow s\ge\ln\frac{\tau}{1-\tau}$. Ứng dụng: cảnh báo học vụ khi $P(\text{đậu})<0{,}7$ (tức $s<0{,}847$).

# In[ ]:


print(f"{'τ':>5}{'điểm số tối thiểu ln(τ/(1-τ))':>34}")
for tau in (0.2, 0.3, 0.5, 0.7, 0.8):
    print(f"{tau:>5}{np.log(tau/(1-tau)):>34.3f}")

s_B = xB @ w_A
print(f"\nSinh viên B: s = {s_B}, p = {sigmoid(s_B):.3f} -> ngưỡng 0,5: 'đậu'; "
      f"nhưng p < 0,7 nên {'CÓ' if sigmoid(s_B) < 0.7 else 'KHÔNG'} cảnh báo học vụ.")
print("Cảnh báo 6 SV (p < 0.7):", ["Có" if p < 0.7 else "Không" for p in p6])


# ## 2.5. Đường biên, đường đồng mức xác suất và odds ratio
# 
# Tập các điểm có cùng xác suất $p=c$ thỏa $\mathbf{xw}=\ln\frac c{1-c}$: siêu phẳng **song song** với đường biên. Khoảng cách từ đường biên đến đường $p=0{,}9$ là $\ln 9/\|\tilde{\mathbf w}\|\approx1{,}55$ (Hình 6):

# In[ ]:


u, v = np.meshgrid(np.linspace(0, 12, 300), np.linspace(-1, 7, 300))
S = w_A[0] + w_A[1] * u + w_A[2] * v
fig, ax = plt.subplots(figsize=(6.8, 5))
im = ax.contourf(u, v, sigmoid(S), levels=np.linspace(0, 1, 21), cmap="RdBu", alpha=0.85)
ax.contour(u, v, S, levels=[0], colors="k", linewidths=2)
ax.contour(u, v, S, levels=[-np.log(9), np.log(9)], colors="k", linestyles="--")
ax.scatter(*X_sv[y_sv == 0].T, c="tab:red", edgecolor="k", label="rớt"); ax.scatter(*X_sv[y_sv == 1].T, c="tab:blue", marker="s", edgecolor="k", label="đậu")
plt.colorbar(im, label="xác suất đậu p"); ax.set(xlabel="x1 (giờ)", ylabel="x2 (vắng)", title="Hình 6: bản đồ xác suất (liền: p=0,5; đứt: p=0,1 và 0,9)")
ax.legend(); plt.show()
print("Khoảng cách biên -> đường p=0,9:", round(np.log(9) / np.linalg.norm(w_A[1:]), 3), "(≈ 1,55)")


# **Odds ratio:** khi $x_j$ tăng 1 đơn vị, odds nhân với $e^{w_j}$ — hằng số. Nhưng *xác suất* thay đổi theo $\partial p/\partial x_j=p(1-p)w_j$, lớn nhất quanh $p=0{,}5$:

# In[ ]:


print(f"Mỗi giờ học thêm nhân odds với e^1 = {np.exp(1):.3f} | mỗi buổi vắng nhân odds với e^-1 = {np.exp(-1):.3f}\n")
x1s = np.arange(3, 10)
s_ = x1s - 4.5                                # x2 = 2 cố định
odds, p_ = np.exp(s_), sigmoid(s_)
print("x1     :", x1s); print("s      :", s_); print("odds   :", np.round(odds, 3))
print("p      :", np.round(p_, 3)); print("Δp     :", np.round(np.diff(p_), 3))
print("odds sau / odds trước:", np.round(odds[1:] / odds[:-1], 3), "(luôn = e = 2,718)")

fig, ax = plt.subplots(figsize=(6.5, 3.8))
xx = np.linspace(0, 12, 200)
for x2_, st in [(0, ":"), (2, "-"), (4, "--")]:
    ax.plot(xx, sigmoid(-2.5 + xx - x2_), st, label=f"x2 = {x2_}")
ax.plot(x1s, p_, "ko", ms=5); ax.set(xlabel="x1 (giờ)", ylabel="p", title="Hình 7: cùng một sigmoid, tịnh tiến theo x1"); ax.legend(); plt.show()

print("\n--- Kiểm tra nhanh mục 2 ---")
print(f"2) σ'(0) = {0.25}, σ'(2) = {sigmoid(2)*(1-sigmoid(2)):.3f}")
print(f"4) SV B vắng thêm 1 buổi: s = {0.5-1}, odds = {np.exp(-0.5):.3f} = 1,649·e^-1 = {np.exp(0.5)*np.exp(-1):.3f}, p = {sigmoid(-0.5):.3f}")
print(f"5) w1 = 0,7: học 1 giờ -> {np.exp(0.7):.3f}; học 2 giờ -> {np.exp(1.4):.3f} = {np.exp(0.7)**2:.3f}")
print(f"7) Sinh viên 2 (y=0, p={p6[1]:.3f}): xác suất gán cho nhãn thật = {1-p6[1]:.3f}")


# ---
# # 3. Huấn luyện mô hình Logistic Regression
# 
# ## 3.1. Vì sao không dùng MSE?
# 
# Với sigmoid, MSE cho $\partial l/\partial s=-2(1-p)\cdot p(1-p)$ — thừa số $\sigma'=p(1-p)$ làm gradient **tắt** khi mô hình *sai chắc chắn* ($p\to0$ trong khi $y=1$). Cross-entropy thì $\partial l/\partial s=p-y$ (lớn nhất đúng ở vùng đó). So sánh (Hình 8), với mẫu $y=1$:

# In[ ]:


s = np.linspace(-8, 8, 400)
p = sigmoid(s)
ce = np.log1p(np.exp(-s))                       # -ln σ(s) = ln(1+e^{-s})
mse = (1 - p) ** 2
g_ce = p - 1                                     # ∂CE/∂s
g_mse = -2 * (1 - p) * p * (1 - p)               # ∂MSE/∂s

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(s, ce, label="cross-entropy"); ax[0].plot(s, mse, label="MSE"); ax[0].set(title="(a) mất mát theo s (y = 1)", xlabel="s", ylim=(0, 6)); ax[0].legend()
ax[1].plot(s, g_ce, label="gradient cross-entropy = p − y"); ax[1].plot(s, g_mse, label="gradient MSE"); ax[1].set(title="(b) gradient theo s", xlabel="s"); ax[1].legend()
plt.tight_layout(); plt.show()

s0 = -6.0; p0 = sigmoid(s0)
print(f"y = 1, s = -6 -> p = {p0:.4f}")
print(f"  gradient cross-entropy = p - y = {p0 - 1:.4f}")
print(f"  gradient MSE           = -2(1-p)·p(1-p) = {-2*(1-p0)*p0*(1-p0):.4f}  (gần 0: gần như không học!)")


# ## 3.2. Hợp lý cực đại và cross-entropy
# 
# Hàm hợp lý $L(\mathbf w)=\prod_ip_i^{y_i}(1-p_i)^{1-y_i}$; lấy $-\frac1n\ln$:
# $$J(\mathbf w)=-\frac1n\sum_i\big[y_i\ln p_i+(1-y_i)\ln(1-p_i)\big],\qquad l_i=-\ln q_i.$$
# So sánh bốn bộ tham số trên 6 sinh viên:

# In[ ]:


def likelihood_va_J(Xb, y, w):
    p = sigmoid(Xb @ w)
    q = p ** y * (1 - p) ** (1 - y)
    L = np.prod(q)
    return L, np.log(L), -np.mean(np.log(q))

print(f"{'w':<18}{'L(w)':>11}{'ln L':>9}{'J(w)':>8}")
for w_ in ([0, 0, 0], [-2.5, 1, 0], [-2.5, 0, -1], [-2.5, 1, -1]):
    L, lnL, J = likelihood_va_J(Xb_sv, y_sv, np.array(w_, dtype=float))
    print(f"{str(w_):<18}{L:>11.5f}{lnL:>9.3f}{J:>8.3f}")
print("(tài liệu: L = 0,0156; 0,0414; 0,00002; 0,795 | J = 0,693; 0,531; 1,771; 0,038)")

print("\nMất mát -ln q:")
for qv in (0.99, 0.9, 0.5, 0.1, 0.01): print(f"  q = {qv:<5} -> {-np.log(qv):.3f}")
print("J(w = 0) = ln 2 =", round(np.log(2), 4))


# **Liên hệ với dạng margin của Perceptron.** Với $y^\pm=2y-1$ và $m=y^\pm s$: $l=\ln(1+e^{-m})$ cho cả hai nhãn.

# In[ ]:


s = 1.0
l_y0 = -np.log(1 - sigmoid(s))                 # y = 0
m = (2 * 0 - 1) * s                            # y± = -1  ->  m = -s
print(f"y=0, s=1: -ln(1-σ(s)) = {l_y0:.4f} | ln(1+e^-m) = {np.log1p(np.exp(-m)):.4f}")
s = 1.0; print(f"y=1, s=1: -ln σ(s)    = {-np.log(sigmoid(s)):.4f} | ln(1+e^-m) = {np.log1p(np.exp(-s)):.4f}")


# ## 3.3. Gradient và quy tắc cập nhật
# 
# Chuỗi $\mathbf w\to s_i\to p_i\to l_i$ cho: $\ \nabla l_i=(p_i-y_i)\mathbf x_i^T$, và
# $$\nabla J(\mathbf w)=\frac1n\mathbf X^T(\mathbf p-\mathbf y),\qquad\mathbf w\leftarrow\mathbf w-\frac\eta n\mathbf X^T(\mathbf p-\mathbf y).$$
# Dạng này **giống hệt** hồi quy tuyến tính, chỉ khác "dự đoán" là $\sigma(\mathbf{Xw})$ thay vì $\mathbf{Xw}$.
# 
# ## 3.4–3.7. Cài đặt từ đầu bằng NumPy

# In[ ]:


def mat_mat(Xb, y, w):
    # Hàm mất mát cross-entropy trung bình J(w).
    p = np.clip(sigmoid(Xb @ w), 1e-12, 1 - 1e-12)      # tránh log(0)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def gradient(Xb, y, w, lam=0.0):
    # Gradient của J (cộng thêm lam * D0 w nếu có điều chuẩn hóa L2).
    p = sigmoid(Xb @ w)
    g = Xb.T @ (p - y) / len(y)
    if lam > 0:
        reg = lam * w
        reg[0] = 0.0                                    # không phạt hệ số chệch w0
        g = g + reg
    return g

def du_doan_xac_suat(Xb, w):
    return sigmoid(Xb @ w)

def du_doan(Xb, w, nguong=0.5):
    return (du_doan_xac_suat(Xb, w) >= nguong).astype(int)

def huan_luyen(Xb, y, eta=0.1, so_epoch=1000, kich_thuoc_batch=None, seed=0, lam=0.0, w_init=None, log_moi=1):
    # Gradient Descent: toàn bộ (batch), từng mẫu (SGD) hoặc mini-batch.
    n, d1 = Xb.shape
    b = n if kich_thuoc_batch is None else kich_thuoc_batch
    rng = np.random.default_rng(seed)
    w = np.zeros(d1) if w_init is None else np.array(w_init, dtype=float)
    lich_su = [mat_mat(Xb, y, w)]                      # J sau mỗi epoch
    for epoch in range(so_epoch):
        idx = rng.permutation(n) if b < n else np.arange(n)
        for k in range(0, n, b):
            lo = idx[k:k + b]                          # chỉ số của mini-batch hiện tại
            w = w - eta * gradient(Xb[lo], y[lo], w, lam)
        if (epoch + 1) % log_moi == 0:
            lich_su.append(mat_mat(Xb, y, w))
    return w, lich_su


# > ### 🔍 Giải thích mã
# > **`sigmoid(s) = 0.5 * (1 + np.tanh(0.5 * s))`** — Công thức gốc $1/(1+e^{-s})$ *tràn số* khi $s$ rất âm (ví dụ $s=-1000$ thì `np.exp(1000)` = `inf`). Đẳng thức $\sigma(s)=\tfrac12\big(1+\tanh\tfrac s2\big)$ cho *cùng giá trị* nhưng `tanh` luôn nằm trong $[-1,1]$ nên không bao giờ tràn. Ta kiểm chứng ngay bên dưới.
# >
# > **`np.clip(p, 1e-12, 1-1e-12)`** — Nếu $p$ bằng đúng 0 hoặc 1 thì `np.log(p)` hoặc `np.log(1-p)` là $-\infty$ và $J$ thành `nan`. `clip` ép $p$ nằm cách 0 và 1 một khoảng nhỏ vô hại.
# >
# > **`Xb.T @ (p - y) / len(y)`** — Đây chính là $\frac1n\mathbf X^T(\mathbf p-\mathbf y)$: kích thước $(d{+}1)\times n$ nhân $n\times1$ cho vector $(d{+}1)\times1$. *Một phép nhân ma trận thay cho vòng lặp qua từng mẫu.*
# >
# > **`reg = lam * w; reg[0] = 0.0`** — Gradient của số hạng phạt $\frac\lambda2\sum_{j\ge1}w_j^2$ là $\lambda\mathbf D_0\mathbf w$ với $\mathbf D_0=\mathrm{diag}(0,1,\dots,1)$. Gán `reg[0] = 0` là cách viết ngắn của việc nhân với $\mathbf D_0$ (không phạt $w_0$).
# >
# > **Vòng lặp mini-batch** — `idx = rng.permutation(n)` xáo trộn thứ tự mẫu mỗi epoch; `range(0, n, b)` nhảy từng bước $b$; `idx[k:k+b]` cắt ra $b$ chỉ số của mini-batch hiện tại (mini-batch cuối có thể ngắn hơn $b$). Với `b = n` ta chỉ có **một** lát cắt = toàn bộ dữ liệu (batch); với `b = 1` ta có $n$ lát cắt, mỗi lát một mẫu (SGD). Nhờ vậy *một hàm* xử lý cả ba biến thể.

# In[ ]:


# Kiểm chứng hai chi tiết ổn định số học
s_cuc_doan = np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0])
with np.errstate(over="ignore"):
    print("Công thức thẳng 1/(1+e^-s):", 1 / (1 + np.exp(-s_cuc_doan)), "  (tràn số ở s = -1000, có cảnh báo)")
print("Công thức tanh            :", sigmoid(s_cuc_doan), "  (an toàn)")
print("Hai công thức bằng nhau ở vùng bình thường:", np.allclose(sigmoid(np.linspace(-20, 20, 101)), 1 / (1 + np.exp(-np.linspace(-20, 20, 101)))))


# ## 3.6. Ví dụ tính tay một vòng huấn luyện (batch GD, $\eta=0{,}1$, $\mathbf w^{(0)}=\mathbf 0$)
# 
# Bước 1: mọi $p_i=0{,}5$ nên $\mathbf p-\mathbf y=[0{,}5,0{,}5,0{,}5,-0{,}5,-0{,}5,-0{,}5]^T$ và $J=\ln2$. Ta in từng đại lượng để đối chiếu:

# In[ ]:


w = np.zeros(3); eta = 0.1
print(f"t=0: w = {w}, J = {mat_mat(Xb_sv, y_sv, w):.4f}")
for t in (1, 2):
    p = sigmoid(Xb_sv @ w)
    XTe = Xb_sv.T @ (p - y_sv)
    g = XTe / 6
    w = w - eta * g
    print(f"\nBước {t}: p - y = {np.round(p - y_sv, 4)}")
    print(f"        X^T(p-y) = {np.round(XTe, 4)} | gradient = {np.round(g, 4)}")
    print(f"        w mới = {np.round(w, 4)} | J = {mat_mat(Xb_sv, y_sv, w):.4f}")

print("\nBảng đầy đủ (đối chiếu tài liệu):")
for k in (0, 1, 2, 3, 4, 5):
    w_k, ls = huan_luyen(Xb_sv, y_sv, eta=0.1, so_epoch=k)
    print(f"  t = {k}: w = {np.round(w_k, 4)} | J = {ls[-1]:.4f}")


# Hàm mất mát **giảm đều** ở mọi bước; cả 6 mẫu đều đóng góp (với trọng số $p_i-y_i$), khác Perceptron chỉ dùng mẫu sai.
# 
# ### Kiểm tra gradient bằng đạo hàm số
# $\dfrac{\partial J}{\partial w_j}\approx\dfrac{J(\mathbf w+\epsilon\mathbf e_j)-J(\mathbf w-\epsilon\mathbf e_j)}{2\epsilon}$ — kỹ thuật tìm lỗi công thức gradient rất hữu ích:

# In[ ]:


w = np.array([-1.0, 0.3, -0.2]); eps = 1e-6
g_so = np.array([(mat_mat(Xb_tr, y_tr, w + eps * e) - mat_mat(Xb_tr, y_tr, w - eps * e)) / (2 * eps)
                 for e in np.eye(3)])
print("gradient giai tich:", gradient(Xb_tr, y_tr, w))
print("gradient so       :", g_so)
print("Kích thước:", Xb_tr.shape, Xb_te.shape)


# > ### 🔍 Giải thích mã
# > **`for e in np.eye(3)`** — `np.eye(3)` là ma trận đơn vị $3\times3$; mỗi *hàng* của nó là một vector đơn vị $\mathbf e_j$ (chỉ phần tử thứ $j$ bằng 1). Vòng lặp lần lượt lấy $\mathbf e_0,\mathbf e_1,\mathbf e_2$.
# >
# > **`w + eps * e` và `w - eps * e`** — dịch **chỉ thành phần thứ $j$** của $\mathbf w$ lên/xuống một đoạn $\epsilon=10^{-6}$ rồi lấy hiệu hai giá trị $J$ chia $2\epsilon$ — đó là sai phân trung tâm (sai số $O(\epsilon^2)$, chính xác hơn sai phân một phía).
# >
# > Nếu hai dòng in ra *trùng nhau* đến 4 chữ số thập phân thì công thức $\frac1n\mathbf X^T(\mathbf p-\mathbf y)$ chắc chắn đúng. Kỹ thuật này gọi là **gradient check**.

# ## 3.4. Vai trò của tốc độ học
# 
# Logistic Regression **rất nhạy** với $\eta$ (khác Perceptron). Chạy batch GD trên 280 mẫu huấn luyện, đặc trưng giữ nguyên thang đo, từ $\mathbf w=\mathbf 0$. Trước hết tìm mức tối ưu $J^*$ bằng bộ giải L-BFGS:

# In[ ]:


from scipy.optimize import minimize

def toi_uu_chinh_xac(Xb, y, w0=None):
    res = minimize(lambda w: mat_mat(Xb, y, w), np.zeros(Xb.shape[1]) if w0 is None else w0,
                   jac=lambda w: gradient(Xb, y, w), method="L-BFGS-B", options=dict(maxiter=10000, ftol=1e-15, gtol=1e-12))
    return res.x, res.fun

w_opt, J_opt = toi_uu_chinh_xac(Xb_tr, y_tr)
print("Nghiệm tối ưu (đặc trưng gốc): w* =", np.round(w_opt, 4), "| J* =", round(J_opt, 4), "  (tài liệu: [-3,520; 1,151; -1,291], 0,2466)")

etas = (0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1)
moc = (10, 100, 500, 2000)
lich_su_eta = {}
print(f"\n{'η':<7}" + "".join(f"{'sau '+str(m)+' epoch':>16}" for m in moc))
for eta_ in etas:
    _, ls = huan_luyen(Xb_tr, y_tr, eta=eta_, so_epoch=2000)
    lich_su_eta[eta_] = ls
    print(f"{eta_:<7}" + "".join(f"{ls[m]:>16.3f}" for m in moc))
print("J lớn nhất khi η = 0,5:", round(max(lich_su_eta[0.5]), 2), "| khi η = 1:", round(max(lich_su_eta[1]), 2), " (tài liệu: ≈2,7 và ≈5,4)")


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for eta_, ls in lich_su_eta.items():
    ax[0].plot(ls[:200], label=f"η = {eta_}")
    ax[1].plot(ls, label=f"η = {eta_}")
ax[0].set(xlabel="epoch", ylabel="J", title="Hình 9a: 200 epoch đầu", ylim=(0, 3)); ax[0].legend(fontsize=8)
ax[1].axhline(J_opt, color="k", ls=":", label="J* ≈ 0,2466"); ax[1].set(xlabel="epoch", title="Hình 9b: 2000 epoch", ylim=(0.2, 1.0), xscale="log"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()


# **Nhận xét:** $\eta$ quá nhỏ (0,001) giảm rất chậm; $\eta$ vừa (0,1–0,5) hội tụ về $J^*\approx0{,}2466$; $\eta=1$ **vọt qua** điểm tối ưu và dao động, không hội tụ. Cách nhận biết: vẽ $J$ theo epoch — $J$ tăng hoặc dao động ⇒ giảm $\eta$.
# 
# ## 3.5. Batch, SGD và Mini-batch (Hình 10)

# In[ ]:


cau_hinh = {"batch (b = n = 280)":  dict(eta=0.2,  kich_thuoc_batch=None),
            "mini-batch (b = 32)":  dict(eta=0.1,  kich_thuoc_batch=32),
            "SGD (b = 1)":          dict(eta=0.01, kich_thuoc_batch=1)}
ket_qua = {}
for ten, cfg in cau_hinh.items():
    w_k, ls = huan_luyen(Xb_tr, y_tr, so_epoch=200, **cfg)
    ket_qua[ten] = (w_k, ls)
    acc_tr = np.mean(du_doan(Xb_tr, w_k) == y_tr)
    acc_te = np.mean(du_doan(Xb_te, w_k) == y_te)
    print(f"{ten:22s} J = {ls[-1]:.4f} | w = {np.round(w_k, 3)} | train {acc_tr:.4f} | test {acc_te:.4f}")
print("J ban dau =", round(ket_qua["batch (b = n = 280)"][1][0], 4), "= ln 2 =", round(np.log(2), 4))
print("Số lần cập nhật mỗi epoch: batch = 1 | mini-batch =", int(np.ceil(280 / 32)), "| SGD = 280")

plt.figure(figsize=(7.5, 4))
for ten, (_, ls) in ket_qua.items(): plt.plot(ls, label=ten)
plt.axhline(J_opt, color="k", ls=":", label="J* ≈ 0,2466")
plt.ylim(0.2, 0.7); plt.xlabel("epoch"); plt.ylabel("J"); plt.title("Hình 10: ba biến thể Gradient Descent"); plt.legend(); plt.show()


# Mini-batch và SGD thực hiện **nhiều bước hơn mỗi epoch** nên giảm nhanh hơn (nhưng dao động); batch mượt nhưng chưa hội tụ sau 200 epoch (đó là lý do accuracy kiểm tra của batch thấp hơn, không phải vì batch "kém hơn"). Dữ liệu chồng lấn nhưng Logistic Regression hội tụ **êm** — không cần thuật toán túi như Perceptron.
# 
# **Kiểm tra nhanh mục 3.**

# In[ ]:


# Câu 1: cross-entropy một mẫu
for y_, p_ in [(1, 0.9), (1, 0.1), (0, 0.9)]:
    l = -np.log(p_) if y_ == 1 else -np.log(1 - p_)
    print(f"1) y={y_}, p={p_}: mất mát = {l:.3f}")

# Câu 2: mẫu x=[1,2,3], y=1, w=0, eta=0.1
x_ = np.array([1, 2, 3.0]); w_ = np.zeros(3)
p_ = sigmoid(x_ @ w_); g_ = (p_ - 1) * x_; w2 = w_ - 0.1 * g_
print(f"2) p = {p_}, gradient = {g_}, w mới = {w2}, p mới = σ({x_ @ w2:.2f}) = {sigmoid(x_ @ w2):.3f}")

# Câu 4: hai mẫu độc lập q1 = 0.8, q2 = 0.5
L = 0.8 * 0.5; print(f"4) L = {L}, J = -ln(L)/2 = {-np.log(L)/2:.3f}")
print(f"6) n=1000, b=50: batch 1 lần | SGD 1000 lần | mini-batch {int(np.ceil(1000/50))} lần mỗi epoch")


# ---
# # 4. Phân tích cơ chế học của Logistic Regression
# 
# ## 4.1. Điều gì xảy ra khi một mẫu bị phân loại sai?
# 
# Đóng góp của mẫu $i$ vào gradient: $\|\nabla l_i\|=|p_i-y_i|\cdot\|\mathbf x_i\|$ — **liên tục** theo mức sai. Với nghiệm tối ưu $\mathbf w^*$:

# In[ ]:


w_star = w_opt
mau = [(2, 5, 0), (9, 0.5, 1), (6, 3, 0), (6, 3, 1), (4, 4, 1), (12, 0, 0)]
tinh_trang = ["Đúng, rất chắc chắn", "Đúng, rất chắc chắn", "Đúng nhưng sát biên", "Sai nhẹ, sát biên", "Sai nặng", "Sai nặng và ở xa"]
print(f"{'x1;x2':<10}{'y':>2}{'s':>9}{'p':>8}{'|p-y|':>8}{'||x||':>7}{'||∇l||':>8}  Tình trạng")
for (a, b, y_), tt in zip(mau, tinh_trang):
    x = np.array([1, a, b]); s = x @ w_star; p = sigmoid(s)
    print(f"{a};{b:<7}{y_:>2}{s:>9.3f}{p:>8.4f}{abs(p-y_):>8.4f}{np.hypot(a, b):>7.2f}{abs(p-y_)*np.hypot(a,b):>8.3f}  {tt}")


# **Một bước cập nhật theo một mẫu** dịch điểm số của chính mẫu đó về phía đúng: $s_i'=s_i+\eta(y_i-p_i)\|\mathbf x_i\|^2$ (phiên bản "mềm" của $m'=m+\eta\|\mathbf x\|^2$ ở Perceptron).

# In[ ]:


x = np.array([1, 4, 4.0]); y_ = 1; eta = 0.1
s = x @ w_star; p = sigmoid(s)
w_moi = w_star - eta * (p - y_) * x
print(f"Mẫu (4;4), y=1: s = {s:.3f}, p = {p:.4f}, ||x||² = {x @ x}")
print(f"Công thức  : s' = s + η(y-p)||x||² = {s + eta*(y_-p)*(x@x):.3f}")
print(f"Tính trực tiếp: s' = {x @ w_moi:.3f} -> p' = {sigmoid(x @ w_moi):.3f}")

# Hình 11: kích thước điểm tỉ lệ |p - y|
p_tr = sigmoid(Xb_tr @ w_star)
fig, ax = plt.subplots(figsize=(6.8, 4.8))
for lbl, mk, mau_ in [(0, "o", "tab:red"), (1, "s", "tab:blue")]:
    m = y_tr == lbl
    ax.scatter(X_tr[m, 0], X_tr[m, 1], s=8 + 220 * np.abs(p_tr[m] - lbl), marker=mk, c=mau_, alpha=0.5, label="rớt" if lbl == 0 else "đậu")
xx = np.linspace(0, 14, 50); ax.plot(xx, -(w_star[0] + w_star[1] * xx) / w_star[2], "k")
ax.set(xlim=(0, 14), ylim=(-1, 8), xlabel="x1", ylabel="x2", title="Hình 11: kích thước điểm ∝ |p − y| (đóng góp vào gradient)"); ax.legend(); plt.show()


# ## 4.2. Chuẩn hóa đặc trưng, khởi tạo
# 
# $z_{ij}=(x_{ij}-\mu_j)/s_j$ với $\mu_j,s_j$ tính **chỉ từ tập huấn luyện**. Chuyển ngược về thang gốc: $w_j=w_j'/s_j$, $\ w_0=w_0'-\sum_j w_j'\mu_j/s_j$.

# In[ ]:


def chuan_hoa(X_tr, X_te):
    mu, s = X_tr.mean(axis=0), X_tr.std(axis=0)         # chỉ dùng tập huấn luyện
    return (X_tr - mu) / s, (X_te - mu) / s, mu, s

Z_tr, Z_te, mu, s = chuan_hoa(X_tr, X_te)
print("mu =", mu, "| s =", s, "  (tài liệu: [6,0934 2,5169], [2,8113 1,6205])")
Zb_tr, Zb_te = them_bias(Z_tr), them_bias(Z_te)
w_z, ls = huan_luyen(Zb_tr, y_tr, eta=5.0, so_epoch=300)
print("w' (chuan hoa) =", np.round(w_z, 4), "| J =", round(ls[-1], 6))
w_goc = np.r_[w_z[0] - np.sum(w_z[1:] * mu / s), w_z[1:] / s]
print("w (thang goc)  =", np.round(w_goc, 4), "  (tài liệu: [-3,520, 1,151, -1,291])")


# > ### 🔍 Giải thích mã
# > **`X_tr.std(axis=0)`** — `axis=0` nghĩa là tính theo **từng cột** (từng đặc trưng) chứ không phải toàn bộ ma trận; kết quả là vector độ lệch chuẩn dài $d$. `NumPy` mặc định dùng độ lệch chuẩn tổng thể (chia $n$), khớp với công thức $s_j=\sqrt{\frac1n\sum(x_{ij}-\mu_j)^2}$ trong tài liệu.
# >
# > **`(X_tr - mu) / s`** — *broadcasting*: `mu` và `s` có độ dài $d$ được "trải" lên từng hàng của ma trận $n\times d$, nên mỗi cột được chuẩn hóa bằng trung bình/độ lệch chuẩn của chính nó.
# >
# > **Vì sao trả về `mu, s`?** Để áp **đúng các giá trị đó** cho tập kiểm tra (`X_te`) — nếu tính lại trên tập kiểm tra là *rò rỉ thông tin*.
# >
# > **`np.r_[a, b]`** nối vector `a` (một số) và `b` (một vector) thành một vector. Dòng `w_goc = ...` chính là công thức chuyển ngược: điểm số không đổi, $\mathbf z\mathbf w'=\mathbf x\mathbf w$, nên $w_j=w_j'/s_j$ và $w_0=w_0'-\sum_jw_j'\mu_j/s_j$ (`np.sum(w_z[1:] * mu / s)` là tổng đó).

# ### Chuẩn hóa làm Gradient Descent nhanh hơn rất nhiều (Hình 12)
# 
# Đếm số epoch cần để $J$ cách $J^*$ không quá $0{,}001$:

# In[ ]:


def so_epoch_hoi_tu(Xb, y, eta, J_star, dung_sai=1e-3, toi_da=6000):
    w = np.zeros(Xb.shape[1])
    for t in range(1, toi_da + 1):
        w = w - eta * gradient(Xb, y, w)
        J = mat_mat(Xb, y, w)
        if J - J_star < dung_sai:
            return t
        if not np.isfinite(J) or J > 50:
            break
    # kiểm tra có dao động không: nếu chưa đạt sau tối đa epoch
    return None

J_star = mat_mat(Xb_tr, y_tr, w_opt)
print(f"J* = {J_star:.5f}")
print(f"{'η':<6}{'Đặc trưng gốc':>18}{'Đặc trưng chuẩn hóa':>22}")
for eta_ in (0.05, 0.1, 0.2, 0.5, 0.8, 1, 2, 5):
    a = so_epoch_hoi_tu(Xb_tr, y_tr, eta_, J_star)
    b = so_epoch_hoi_tu(Zb_tr, y_tr, eta_, J_star)
    fa = f"{a}" if a else ("không hội tụ" if eta_ >= 0.8 else ">6000")
    fb = f"{b}" if b else "không hội tụ"
    print(f"{eta_:<6}{fa:>18}{fb:>22}")


# In[ ]:


# Hình 12: đường đồng mức J theo (w1, w2), giữ w0 ở giá trị tối ưu, và đường đi của GD từ w = 0
def ve_duong_di(ax, Xb, y, w_opt_, eta, so_buoc, ten, w1_rng, w2_rng):
    g1, g2 = np.linspace(*w1_rng, 120), np.linspace(*w2_rng, 120)
    G1, G2 = np.meshgrid(g1, g2)
    W = np.stack([np.full(G1.size, w_opt_[0]), G1.ravel(), G2.ravel()])     # 3 x (120*120)
    P = np.clip(sigmoid(Xb @ W), 1e-12, 1 - 1e-12)                          # n x (120*120)
    J = -np.mean(y[:, None] * np.log(P) + (1 - y[:, None]) * np.log(1 - P), axis=0).reshape(G1.shape)
    ax.contour(G1, G2, J, levels=np.linspace(J.min(), 1.5, 20), cmap="viridis")
    w = np.zeros(3); qd = [w.copy()]
    for _ in range(so_buoc):
        w = w - eta * gradient(Xb, y, w); qd.append(w.copy())
    qd = np.array(qd)
    ax.plot(qd[:, 1], qd[:, 2], "k.-", ms=3, lw=0.8); ax.plot(0, 0, "ko", ms=7)
    ax.plot(w_opt_[1], w_opt_[2], "r*", ms=15)
    ax.set(title=ten, xlabel="w1", ylabel="w2")

w_opt_z, _ = toi_uu_chinh_xac(Zb_tr, y_tr)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
ve_duong_di(ax[0], Xb_tr, y_tr, w_opt, 0.5, 120, "(a) đặc trưng gốc (η=0,5): thung lũng dài hẹp, đi dích dắc", (-0.2, 1.6), (-1.8, 0.2))
ve_duong_di(ax[1], Zb_tr, y_tr, w_opt_z, 1.0, 30, "(b) đặc trưng chuẩn hóa (η=1): đi thẳng về đích", (-0.5, 4), (-3, 0.5))
plt.tight_layout(); plt.show()


# > ### 🔍 Giải thích mã (Hình 12)
# > **Mục tiêu:** vẽ đường đồng mức của $J(w_0^*,w_1,w_2)$ trên một luoi giá trị $(w_1,w_2)$, rồi vẽ đường đi của Gradient Descent lên trên.
# >
# > - `np.meshgrid(g1, g2)` tạo hai ma trận $120\times120$ chứa mọi cặp $(w_1,w_2)$ trên luoi.
# > - `np.stack([np.full(G1.size, w0*), G1.ravel(), G2.ravel()])` xếp 3 hàng thành ma trận `W` cỡ $3\times14400$; **mỗi cột là một bộ tham số $\mathbf w$ đầy đủ** $(w_0^*,w_1,w_2)$ (ở đây $w_0$ cố định ở giá trị tối ưu — cố ý "cắt lát" cho vẽ được 2 chiều).
# > - `Xb @ W` ($n\times3$ nhân $3\times14400$) cho ma trận điểm số $n\times14400$: **điểm số của mọi mẫu tại mọi bộ tham số cùng lúc**, không cần vòng lặp qua 14400 điểm luoi.
# > - `y[:, None]` biến vector $y$ ($n$) thành cột ($n\times1$) để *broadcast* với ma trận xác suất $n\times14400$; `np.mean(..., axis=0)` trung bình theo mẫu cho ra $J$ tại từng điểm luoi, `reshape` đưa về dạng $120\times120$ để `contour` vẽ.
# > - Phần sau chạy Gradient Descent từ $\mathbf w=\mathbf 0$ và ghi lại quỹ đạo `qd` để vẽ `qd[:,1], qd[:,2]` (chỉ hai thành phần $w_1,w_2$).
# >
# > Kết quả: với đặc trưng gốc các đường mức là elip dài hẹp nên đường đi dích dắc; sau chuẩn hóa các đường mức gần tròn nên đi thẳng.

# **Chuẩn hóa còn làm trọng số dễ so sánh** (cùng đơn vị "một độ lệch chuẩn"):

# In[ ]:


print(f"Tăng 1 độ lệch chuẩn số giờ học (≈ {s[0]:.2f} giờ): odds đậu ×{np.exp(w_z[1]):.1f}   (tài liệu ≈ 25,4)")
print(f"Tăng 1 độ lệch chuẩn số buổi vắng            : odds đậu ×{np.exp(w_z[2]):.3f}  (tài liệu ≈ 0,123)")

print("\nKhởi tạo khác nhau, cùng một nghiệm (η = 5, 300 epoch, đặc trưng chuẩn hóa):")
for w_i in ([0, 0, 0], [5, -3, 2], [-8, 4, 4], [2, -6, -6]):
    w_, ls = huan_luyen(Zb_tr, y_tr, eta=5.0, so_epoch=300, w_init=w_i)
    print(f"  w0 = {str(w_i):<12} J đầu = {ls[0]:.3f} -> w = {np.round(w_, 4)} | J = {ls[-1]:.6f}")


# ## 4.3. Hàm mất mát lồi và hội tụ về nghiệm duy nhất
# 
# Hessian: $\mathbf H(\mathbf w)=\frac1n\mathbf X^T\mathbf S\mathbf X$ với $\mathbf S=\mathrm{diag}(p_i(1-p_i))$. Vì $\mathbf z^T\mathbf H\mathbf z=\frac1n\sum p_i(1-p_i)(\mathbf x_i\mathbf z)^2\ge0$ nên $J$ **lồi**: mọi cực tiểu địa phương là toàn cục, và nếu $\mathbf H\succ0$ thì nghiệm **duy nhất**.

# In[ ]:


def hessian(Xb, w):
    p = sigmoid(Xb @ w)
    S = p * (1 - p)                              # đường chéo của ma trận S
    return (Xb.T * S) @ Xb / len(p)              # (1/n) X^T S X

H_goc = hessian(Xb_tr, w_opt)
H_z = hessian(Zb_tr, w_opt_z)
lam_goc, lam_z = np.linalg.eigvalsh(H_goc), np.linalg.eigvalsh(H_z)
print("Trị riêng của H (đặc trưng gốc)     :", np.round(lam_goc, 4), " (tài liệu: 0,0046; 0,0688; 3,502)")
print("Trị riêng của H (đặc trưng chuẩn hóa):", np.round(lam_z, 4))
print(f"\n{'':<24}{'Đặc trưng gốc':>15}{'Chuẩn hóa':>12}")
print(f"{'λ_max':<24}{lam_goc.max():>15.4f}{lam_z.max():>12.4f}")
print(f"{'λ_min':<24}{lam_goc.min():>15.4f}{lam_z.min():>12.4f}")
print(f"{'Số điều kiện κ':<24}{lam_goc.max()/lam_goc.min():>15.1f}{lam_z.max()/lam_z.min():>12.2f}")
print(f"{'Ngưỡng ổn định 2/λ_max':<24}{2/lam_goc.max():>15.2f}{2/lam_z.max():>12.1f}")


# > ### 🔍 Giải thích mã (Hessian)
# > **`(Xb.T * S) @ Xb / len(p)`** là cách viết gọn của $\frac1n\mathbf X^T\mathbf S\mathbf X$ *mà không tạo ma trận chéo $\mathbf S$ cỡ $n\times n$* (tốn bộ nhớ):
# > - `Xb.T` có dạng $(d{+}1)\times n$ và `S` là vector độ dài $n$; phép `Xb.T * S` *nhân từng cột $i$ của $\mathbf X^T$ với $S_i$* (broadcasting) — đúng bằng $\mathbf X^T\mathbf S$.
# > - Nhân tiếp với `Xb` ($n\times(d{+}1)$) cho ma trận $(d{+}1)\times(d{+}1)$.
# >
# > **`np.linalg.eigvalsh`** tính trị riêng của ma trận *đối xứng* (chữ `h` = Hermitian), nhanh và ổn định hơn `eigvals`. Trị riêng lớn nhất $\lambda_{\max}$ quyết định $\eta$ tối đa ($\eta<2/\lambda_{\max}$), trị riêng nhỏ nhất $\lambda_{\min}$ quyết định tốc độ dọc hướng thoải nhất; số điều kiện $\kappa=\lambda_{\max}/\lambda_{\min}$ đo độ "dài hẹp" của thung lũng.

# In[ ]:


# Hình 13: (a) lát cắt c·w* của cross-entropy (lồi) và MSE+sigmoid (chữ S); (b) bốn khởi tạo hội tụ về cùng J
cs = np.linspace(-1, 3, 300)
J_ce = [mat_mat(Zb_tr, y_tr, c * w_opt_z) for c in cs]
J_mse = [np.mean((y_tr - sigmoid(Zb_tr @ (c * w_opt_z))) ** 2) for c in cs]
fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.9))
ax[0].plot(cs, J_ce, label="cross-entropy (lồi)"); ax[0].plot(cs, J_mse, label="MSE + sigmoid (không lồi)")
ax[0].set(xlabel="c (w = c·w*)", ylabel="mất mát", title="(a) lát cắt theo hướng w*"); ax[0].legend()
for w_i in ([0, 0, 0], [5, -3, 2], [-8, 4, 4], [2, -6, -6]):
    _, ls = huan_luyen(Zb_tr, y_tr, eta=5.0, so_epoch=100, w_init=w_i); ax[1].plot(ls, label=f"w0 = {w_i}")
ax[1].set(xlabel="epoch", ylabel="J", yscale="log", title="(b) mọi khởi tạo đều hội tụ về cùng J"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()

# Điều kiện tối ưu: trung bình xác suất dự đoán = tỉ lệ lớp dương
p_tr_opt = sigmoid(Xb_tr @ w_opt)
print(f"Trung bình p dự đoán = {p_tr_opt.mean():.4f} | tỉ lệ đậu thực tế = {y_tr.mean():.4f}  (= 149/280)")
print("X^T (p - y) tại nghiệm tối ưu:", np.round(Xb_tr.T @ (p_tr_opt - y_tr), 6), "(≈ 0 vì gradient = 0)")


# ## 4.4. Dữ liệu phân tách hoàn hảo: $\|\mathbf w\|$ tăng không giới hạn
# 
# Với dữ liệu 6 sinh viên (phân tách được), nhân $\mathbf w^*$ với $c\to\infty$ làm $J\to0$ nhưng **không bao giờ bằng 0** — không tồn tại cực tiểu hữu hạn:

# In[ ]:


import time
t0 = time.time()
w = np.zeros(3)
moc = {10, 100, 1000, 10**4, 10**5, 10**6}
huong_chuan = np.array([-1.0, 2.0, -4.0]) / np.linalg.norm([-1, 2, -4])
print(f"{'Số bước t':>10}{'||w||':>8}{'J':>12}{'cos(w, [-1,2,-4])':>20}")
for t in range(1, 10**6 + 1):
    p = 1.0 / (1.0 + np.exp(-(Xb_sv @ w)))
    w -= 0.5 * (Xb_sv.T @ (p - y_sv)) / 6                # batch GD, eta = 0.5
    if t in moc:
        J = mat_mat(Xb_sv, y_sv, w)
        print(f"{t:>10}{np.linalg.norm(w):>8.2f}{J:>12.2e}{(w @ huong_chuan) / np.linalg.norm(w):>20.4f}")
print(f"(chạy {time.time() - t0:.0f} giây) -> ||w|| tăng ~1,5 mỗi khi t tăng 10 lần; hướng của w tiến về hướng margin lớn nhất (SVM).")


# > ### 🔍 Giải thích mã
# > - Vòng lặp chạy **một triệu** bước batch GD nên viết trực tiếp, bỏ qua `huan_luyen` (hàm đó tính $J$ mỗi epoch — quá tốn). Công thức `p = 1/(1+exp(-s))` dùng thẳng được vì $|s|$ ở đây còn nhỏ (<30), không tràn.
# > - **`w -= ...`** cập nhật tại chỗ (in-place); **`if t in moc`** chỉ in tại các mốc $10,100,\dots,10^6$ (tập `moc`).
# > - **Độ tương đồng cosin** $\cos(\mathbf w,\mathbf v)=\dfrac{\mathbf w\cdot\mathbf v}{\|\mathbf w\|\|\mathbf v\|}$: `huong_chuan` đã được chuẩn hóa độ dài 1 nên chỉ cần chia thêm cho $\|\mathbf w\|$. Giá trị gần 1 nghĩa là $\mathbf w$ gần như cùng *hướng* với $[-1,2,-4]$ — hướng có margin lớn nhất — dù độ dài của nó tăng mãi.

# ### Điều chuẩn hóa L2
# $$J_\lambda(\mathbf w)=J(\mathbf w)+\frac\lambda2\mathbf w^T\mathbf D_0\mathbf w,\qquad\nabla J_\lambda=\frac1n\mathbf X^T(\mathbf p-\mathbf y)+\lambda\mathbf D_0\mathbf w.$$
# Số hạng phạt luôn kéo $\mathbf w$ về 0 nên nghiệm **hữu hạn duy nhất** ngay cả khi dữ liệu phân tách hoàn hảo:

# In[ ]:


print(f"{'λ':<8}{'w':<36}{'||w~||':>8}{'p(SV B)':>9}")
for lam in (0.0, 0.001, 0.01, 0.1):
    w_l, _ = huan_luyen(Xb_sv, y_sv, eta=0.5, so_epoch=200000, lam=lam, log_moi=10**6)
    p_B = sigmoid(np.array([1, 5, 2]) @ w_l)
    print(f"{lam:<8}{str(np.round(w_l, 4)):<36}{np.linalg.norm(w_l[1:]):>8.3f}{p_B:>9.3f}")
print("(tài liệu: ||w~|| = 7,422; 3,042; 2,052; 1,184 | p(B) = 0,907; 0,751; 0,665; 0,577)")


# $\lambda$ càng lớn thì $\mathbf w$ càng nhỏ và xác suất càng "dè dặt" (gần 0,5) — đánh đổi giữa khớp dữ liệu và độ lớn của $\mathbf w$.
# 
# ## 4.5. Giới hạn của đường biên tuyến tính: XOR và đặc trưng đa thức
# 
# **XOR:** gradient tại $\mathbf w=\mathbf 0$ bằng 0 (do đối xứng) và $J$ lồi nên $\mathbf w=\mathbf 0$ là cực tiểu toàn cục: mô hình hội tụ êm về "mọi $p=0{,}5$" (khác Perceptron quay vòng mãi).

# In[ ]:


X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([0, 1, 1, 0])
Xb_xor = them_bias(X_xor)

g0 = Xb_xor.T @ (0.5 - y_xor) / 4
print("Gradient tại w = 0:", g0, "-> bằng 0, w = 0 là cực tiểu toàn cục")
w_x, ls = huan_luyen(Xb_xor, y_xor, eta=0.5, so_epoch=5000, w_init=[2, -3, 1])
print("GD từ w0 = [2,-3,1]:  w =", np.round(w_x, 4), "| p =", np.round(sigmoid(Xb_xor @ w_x), 3), "| J =", round(ls[-1], 4), "(= ln 2 =", round(np.log(2), 4), ")")

# Thêm đặc trưng x1*x2 và điều chuẩn L2
Xb_xor2 = np.column_stack([Xb_xor, X_xor[:, 0] * X_xor[:, 1]])
print(f"\n{'λ':<8}{'w':<36}{'p cho (0,0),(0,1),(1,0),(1,1)'}")
for lam in (0.001, 0.01):
    w_, _ = huan_luyen(Xb_xor2, y_xor, eta=0.5, so_epoch=300000, lam=lam, log_moi=10**6)
    p_ = sigmoid(Xb_xor2 @ w_)
    print(f"{lam:<8}{str(np.round(w_, 3)):<36}{np.round(p_, 3)}  -> nhãn {(p_ >= 0.5).astype(int)}")
print("(tài liệu: λ=0,001: w = [-2,389; 5,076; 5,076; -10,856]; λ=0,01: [-0,735; 1,827; 1,827; -4,450])")


# ### Ví dụ: dữ liệu hình vòng tròn (Hình 15)
# 
# Hai lớp đồng tâm — không đường thẳng nào tách được. Ta so Logistic Regression **tuyến tính** ($\mathbf x=[1,x_1,x_2]$) với **đặc trưng bậc hai** ($[1,x_1,x_2,x_1^2,x_1x_2,x_2^2]$). *(Tài liệu không công bố dữ liệu nên notebook mô phỏng 300 điểm — 210 huấn luyện, 90 kiểm tra; con số có thể lệch nhẹ.)*

# In[ ]:


from itertools import combinations_with_replacement

def dac_trung_da_thuc(X, bac):
    # Trả về ma trận [x_j, x_j*x_k, ...] cho mọi tổ hợp có lặp của các cột, bậc 1..bac (không có cột 1).
    cot = []
    for k in range(1, bac + 1):
        for tohop in combinations_with_replacement(range(X.shape[1]), k):
            cot.append(np.prod(X[:, tohop], axis=1))
    return np.column_stack(cot)

# Dữ liệu vòng tròn
rng_c = np.random.default_rng(1)
n_c = 150
goc1, goc0 = rng_c.uniform(0, 2 * np.pi, n_c), rng_c.uniform(0, 2 * np.pi, n_c)
r1, r0 = rng_c.normal(1.0, 0.25, n_c), rng_c.normal(2.3, 0.35, n_c)
Xc = np.vstack([np.c_[r1 * np.cos(goc1), r1 * np.sin(goc1)], np.c_[r0 * np.cos(goc0), r0 * np.sin(goc0)]])
yc = np.r_[np.ones(n_c), np.zeros(n_c)].astype(int)
Xc_tr, Xc_te, yc_tr, yc_te = chia_train_test(Xc, yc, ti_le_test=0.3, seed=0)

def huan_luyen_chuan_hoa(Xtr, ytr, Xte, bac, eta=1.0, so_epoch=3000, lam=0.0):
    F_tr, F_te = (dac_trung_da_thuc(Xtr, bac), dac_trung_da_thuc(Xte, bac)) if bac > 1 else (Xtr, Xte)
    F_tr_z, F_te_z, _, _ = chuan_hoa(F_tr, F_te)
    A_tr, A_te = them_bias(F_tr_z), them_bias(F_te_z)
    w_, _ = huan_luyen(A_tr, ytr, eta=eta, so_epoch=so_epoch, lam=lam, log_moi=so_epoch)
    return w_, A_tr, A_te

print(f"{'Mô hình':<22}{'Số tham số':>11}{'Acc huấn luyện':>16}{'Acc kiểm tra':>14}")
mo_hinh_c = {}
for bac, ten in [(1, "Tuyến tính"), (2, "Đặc trưng bậc hai")]:
    w_, A_tr, A_te = huan_luyen_chuan_hoa(Xc_tr, yc_tr, Xc_te, bac)
    mo_hinh_c[bac] = w_
    print(f"{ten:<22}{len(w_):>11}{np.mean(du_doan(A_tr, w_) == yc_tr):>16.1%}{np.mean(du_doan(A_te, w_) == yc_te):>14.1%}")
print("(tài liệu: tuyến tính 51,0% / 58,9%; bậc hai 99,0% / 98,9%)")

# Hình 15
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
g = np.linspace(-3.5, 3.5, 250); U, V = np.meshgrid(g, g); luoi = np.c_[U.ravel(), V.ravel()]
for a, bac, ten in zip(ax, (1, 2), ("(a) tuyến tính: chỉ một đường thẳng", "(b) đặc trưng bậc hai: đường biên là đường tròn")):
    F_tr = dac_trung_da_thuc(Xc_tr, bac) if bac > 1 else Xc_tr
    mu_f, s_f = F_tr.mean(0), F_tr.std(0)
    F_g = dac_trung_da_thuc(luoi, bac) if bac > 1 else luoi
    P = sigmoid(them_bias((F_g - mu_f) / s_f) @ mo_hinh_c[bac]).reshape(U.shape)
    a.contourf(U, V, P, levels=np.linspace(0, 1, 21), cmap="RdBu", alpha=0.8); a.contour(U, V, P, levels=[0.5], colors="k")
    a.scatter(*Xc_tr[yc_tr == 0].T, c="tab:red", s=12, edgecolor="k", lw=0.3); a.scatter(*Xc_tr[yc_tr == 1].T, c="tab:blue", s=12, marker="s", edgecolor="k", lw=0.3)
    a.set(title=ten, xlabel="x1", ylabel="x2")
plt.tight_layout(); plt.show()

from math import comb
print("\nSố đặc trưng đa thức C(d+k, k) - 1 cho (d,k) = (2,2), (10,3), (4,3):", [comb(d_ + k_, k_) - 1 for d_, k_ in [(2, 2), (10, 3), (4, 3)]], "(tài liệu: 5, 285, 34)")


# > ### 🔍 Giải thích mã (đặc trưng đa thức)
# > **`combinations_with_replacement(range(d), k)`** sinh mọi *tổ hợp có lặp* $k$ chỉ số cột, ví dụ với 2 cột ($0,1$) và $k=2$: $(0,0),(0,1),(1,1)$ — tương ứng $x_1^2,\ x_1x_2,\ x_2^2$.
# >
# > **`np.prod(X[:, tohop], axis=1)`** — `X[:, tohop]` lấy các cột trong `tohop` (có thể lặp cột); `np.prod(..., axis=1)` nhân chúng *theo hàng* → một cột đặc trưng mới. Với `tohop=(0,0)` ta được $x_1\cdot x_1=x_1^2$.
# >
# > Vòng `for k in range(1, bac+1)` gom bậc 1, 2, …, `bac`. Với 2 đặc trưng gốc và bậc 2 ta được $2+3=5$ cột — khớp công thức $\binom{d+k}{k}-1=5$.
# >
# > **Vẽ biên cong:** để tô nền xác suất ta phải tạo đặc trưng đa thức *cho từng điểm luoi* **bằng đúng $\mu,s$ của tập huấn luyện** (`mu_f, s_f`) rồi mới đưa qua mô hình — nếu quên bước chuẩn hóa này thì xác suất vẽ ra sẽ sai.

# ---
# # 5. Đánh giá và trực quan hóa
# 
# Dùng bộ dữ liệu 400 sinh viên (280 huấn luyện, 120 kiểm tra) và mô hình đã huấn luyện ở mục 4.2: $\mathbf w=[-3{,}520;\,1{,}151;\,-1{,}291]^T$ (đặc trưng gốc) hay $\mathbf w'=[0{,}243;\,3{,}235;\,-2{,}092]^T$ (chuẩn hóa).
# 
# ## 5.1. Đường cong học và bản đồ xác suất
# 
# Huấn luyện batch GD trên đặc trưng chuẩn hóa ($\eta=0{,}5$, 300 epoch), ghi $J$ và accuracy trên **cả hai tập** sau mỗi epoch (Hình 16):

# In[ ]:


def huan_luyen_theo_doi(Xb_tr, y_tr, Xb_te, y_te, eta, so_epoch):
    w = np.zeros(Xb_tr.shape[1])
    hist = {"J_tr": [], "J_te": [], "a_tr": [], "a_te": []}
    for t in range(so_epoch + 1):
        hist["J_tr"].append(mat_mat(Xb_tr, y_tr, w)); hist["J_te"].append(mat_mat(Xb_te, y_te, w))
        hist["a_tr"].append(np.mean(du_doan(Xb_tr, w) == y_tr)); hist["a_te"].append(np.mean(du_doan(Xb_te, w) == y_te))
        w = w - eta * gradient(Xb_tr, y_tr, w)
    return w, hist

w_300, hist = huan_luyen_theo_doi(Zb_tr, y_tr, Zb_te, y_te, 0.5, 300)
print(f"{'Epoch':>6}{'J train':>10}{'J test':>9}{'Acc train':>11}{'Acc test':>10}")
for t in (0, 10, 50, 300):
    print(f"{t:>6}{hist['J_tr'][t]:>10.4f}{hist['J_te'][t]:>9.4f}{hist['a_tr'][t]:>11.1%}{hist['a_te'][t]:>10.1%}")
print("Lúc w = 0 mọi p = 0,5 nên dự đoán y = 1 cho mọi mẫu: accuracy = tỉ lệ lớp dương =", f"{y_tr.mean():.3f} (train), {y_te.mean():.3f} (test)")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(hist["J_tr"], label="J huấn luyện"); ax[0].plot(hist["J_te"], label="J kiểm tra"); ax[0].set(xlabel="epoch", ylabel="J", title="(a) hàm mất mát"); ax[0].legend()
ax[1].plot(hist["a_tr"], label="train"); ax[1].plot(hist["a_te"], label="test"); ax[1].set(xlabel="epoch", ylabel="accuracy", title="(b) accuracy bão hòa sớm", xlim=(0, 100)); ax[1].legend()
plt.tight_layout(); plt.show()


# In[ ]:


# Hình 17: bản đồ xác suất, điểm kiểm tra to có viền, vòng vàng quanh mẫu kiểm tra bị dự đoán sai
p_te = sigmoid(Xb_te @ w_goc)
sai = (p_te >= 0.5).astype(int) != y_te
fig, ax = plt.subplots(figsize=(7.5, 5.2))
u, v = np.meshgrid(np.linspace(0, 15, 300), np.linspace(-1, 8, 300))
S = w_goc[0] + w_goc[1] * u + w_goc[2] * v
im = ax.contourf(u, v, sigmoid(S), levels=np.linspace(0, 1, 21), cmap="RdBu", alpha=0.8)
ax.contour(u, v, S, levels=[0], colors="k", linewidths=2); ax.contour(u, v, S, levels=[-np.log(9), np.log(9)], colors="k", linestyles="--")
ax.scatter(*X_tr[y_tr == 0].T, c="tab:red", s=10, alpha=0.7); ax.scatter(*X_tr[y_tr == 1].T, c="tab:blue", s=10, marker="s", alpha=0.7)
ax.scatter(*X_te[y_te == 0].T, c="tab:red", s=45, edgecolor="w"); ax.scatter(*X_te[y_te == 1].T, c="tab:blue", s=45, marker="s", edgecolor="w")
ax.scatter(*X_te[sai].T, s=170, facecolors="none", edgecolors="gold", linewidths=2, label="mẫu kiểm tra dự đoán sai")
plt.colorbar(im, label="p (đậu)"); ax.set(xlabel="x1 (giờ)", ylabel="x2 (vắng)", title="Hình 17: bản đồ xác suất"); ax.legend(loc="upper right"); plt.show()

nw = np.linalg.norm(w_goc[1:])
print(f"||w~|| = {nw:.3f}; khoảng cách biên -> p=0,1 / 0,9 = {np.log(9)/nw:.2f}")
trong_dai = (p_te > 0.1) & (p_te < 0.9)
print(f"{sai.sum()} mẫu kiểm tra sai; trong dải 0,1<p<0,9: {(sai & trong_dai).sum()} | sai chắc chắn: {(sai & ~trong_dai).sum()}")
loss_te = -(y_te * np.log(p_te) + (1 - y_te) * np.log(1 - p_te))
idx_sc = np.where(sai & ~trong_dai)[0]
print("p của các mẫu 'sai chắc chắn':", np.round(p_te[idx_sc], 3), "| mất mát:", np.round(loss_te[idx_sc], 2))
print(f"Chúng chiếm {len(idx_sc)/len(y_te):.1%} số mẫu nhưng đóng góp {loss_te[idx_sc].sum()/loss_te.sum():.0%} tổng mất mát kiểm tra  (tài liệu: ≈ 31% và 2,5%)")


# ## 5.2. Tập huấn luyện/kiểm tra và hiện tượng học thuộc (overfitting)
# 
# Chỉ lấy **40 mẫu huấn luyện** (20 đậu, 20 rớt), dựng đặc trưng đa thức bậc $k=1,\dots,5$, huấn luyện bằng **L-BFGS** với/không điều chuẩn L2, rồi đánh giá trên 120 mẫu kiểm tra:

# In[ ]:


def huan_luyen_lbfgs(Xb, y, lam=0.0, maxiter=300):
    def f_va_g(w):
        p = np.clip(sigmoid(Xb @ w), 1e-12, 1 - 1e-12)
        J = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)) + 0.5 * lam * np.sum(w[1:] ** 2)
        return J, gradient(Xb, y, w, lam)
    res = minimize(f_va_g, np.zeros(Xb.shape[1]), jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter))
    return res.x

# 40 mẫu cân bằng: 20 đậu + 20 rớt đầu tiên của tập huấn luyện
idx40 = np.r_[np.where(y_tr == 1)[0][:20], np.where(y_tr == 0)[0][:20]]
X40, y40 = X_tr[idx40], y_tr[idx40]

ket = []
print(f"{'k':>2}{'#đặc trưng':>11}{'λ':>7}{'J train':>9}{'J test':>9}{'Acc train':>11}{'Acc test':>10}")
for k in range(1, 6):
    F40 = dac_trung_da_thuc(X40, k) if k > 1 else X40
    F_te = dac_trung_da_thuc(X_te, k) if k > 1 else X_te
    Z40, Zte, _, _ = chuan_hoa(F40, F_te)
    B40, Bte = them_bias(Z40), them_bias(Zte)
    for lam in (0.0, 0.01):
        w_ = huan_luyen_lbfgs(B40, y40, lam=lam)
        r = (k, lam, mat_mat(B40, y40, w_), mat_mat(Bte, y_te, w_), np.mean(du_doan(B40, w_) == y40), np.mean(du_doan(Bte, w_) == y_te))
        ket.append(r)
        print(f"{k:>2}{F40.shape[1]:>11}{lam:>7}{r[2]:>9.3f}{r[3]:>9.3f}{r[4]:>11.1%}{r[5]:>10.1%}")


# > ### 🔍 Giải thích mã
# > **`minimize(f_va_g, w0, jac=True, method="L-BFGS-B")`** — `scipy.optimize.minimize` là bộ giải tối ưu có sẵn. Với `jac=True`, hàm mục tiêu **trả về cùng lúc hai thứ `(J, gradient)`** (nên ta đặt tên `f_va_g`); L-BFGS dùng gradient này (và xấp xỉ độ cong từ các bước trước — "có nhớ") nên hội tụ nhanh hơn Gradient Descent thuần rất nhiều. Đây là loại bộ giải mà scikit-learn dùng mặc định (`lbfgs`).
# >
# > **`maxiter=300`** — với $\lambda=0$ và dữ liệu phân tách được, $\|\mathbf w\|\to\infty$ nên phải *giới hạn số vòng lặp* (một dạng dừng sớm, như đã nói ở mục 4.4).
# >
# > **`chuan_hoa(F40, F_te)`** — mean/std chỉ lấy từ **40 mẫu huấn luyện**, rồi áp cho tập kiểm tra (tránh rò rỉ). Với đặc trưng đa thức, chuẩn hóa là *bắt buộc* vì $x^5$ có thang đo rất khác $x$.
# >

# In[ ]:


ks = list(range(1, 6))
def lay(lam, c): return [r[c] for r in ket if r[1] == lam]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for lam, mau_ in [(0.0, "tab:red"), (0.01, "tab:blue")]:
    ax[0].plot(ks, lay(lam, 2), "--o", color=mau_, label=f"train, λ={lam}"); ax[0].plot(ks, lay(lam, 3), "-o", color=mau_, label=f"test, λ={lam}")
    ax[1].plot(ks, lay(lam, 4), "--o", color=mau_); ax[1].plot(ks, lay(lam, 5), "-o", color=mau_)
ax[0].set(xlabel="bậc đa thức k", ylabel="log-loss", title="(a) log-loss", yscale="log"); ax[0].legend(fontsize=7)
ax[1].set(xlabel="bậc đa thức k", ylabel="accuracy", title="(b) accuracy")
plt.suptitle("Hình 18: không điều chuẩn (đỏ) → log-loss kiểm tra tăng vọt; L2 (xanh) → ổn định", y=1.02)
plt.tight_layout(); plt.show()

r5 = [r for r in ket if r[0] == 5]
print(f"k=5: accuracy test = {r5[0][5]:.1%} (không điều chuẩn) so với {r5[1][5]:.1%} (L2) — chênh khoảng {100*(r5[1][5]-r5[0][5]):.0f} điểm,")
print(f"     nhưng log-loss test = {r5[0][3]:.2f} so với {r5[1][3]:.2f} (gấp ~{r5[0][3]/r5[1][3]:.0f} lần): log-loss phản ánh mức 'quá tự tin' rõ rệt hơn nhiều.")
print("(Số liệu cụ thể phụ thuộc 40 mẫu được chọn nên có thể lệch tài liệu, xu hướng thì giống nhau.)")


# ## 5.3. Ma trận nhầm lẫn, precision, recall, F1 và ngưỡng quyết định

# In[ ]:


def ma_tran_nham_lan(y, y_hat):
    # Ma trận nhầm lẫn C = [[TN, FP], [FN, TP]] (hàng: nhãn thực, cột: nhãn dự đoán).
    tp = np.sum((y == 1) & (y_hat == 1)); tn = np.sum((y == 0) & (y_hat == 0))
    fp = np.sum((y == 0) & (y_hat == 1)); fn = np.sum((y == 1) & (y_hat == 0))
    return np.array([[tn, fp], [fn, tp]])

C = ma_tran_nham_lan(y_te, (p_te >= 0.5).astype(int))
print(C)
tn, fp, fn, tp = C.ravel()
print("precision =", round(tp / (tp + fp), 3), "| recall =", round(tp / (tp + fn), 3),
      "| do dac hieu =", round(tn / (tn + fp), 3), "| F1 =", round(2 * tp / (2 * tp + fp + fn), 3))
p_tr = sigmoid(Xb_tr @ w_goc)
print("Ma trận nhầm lẫn tập huấn luyện:\n", ma_tran_nham_lan(y_tr, (p_tr >= 0.5).astype(int)))


# ### Đổi ngưỡng $\tau$ và chọn ngưỡng theo chi phí
# 
# Khi $\tau$ tăng: precision và độ đặc hiệu tăng, recall giảm. Nếu chi phí báo nhầm (FP) là $c_{FP}$ và bỏ sót (FN) là $c_{FN}$ thì ngưỡng tối ưu là $\tau^*=\dfrac{c_{FP}}{c_{FP}+c_{FN}}$ (khi xác suất hiệu chuẩn tốt).

# In[ ]:


def chi_so_theo_nguong(y, p, tau):
    C_ = ma_tran_nham_lan(y, (p >= tau).astype(int)); tn, fp, fn, tp = C_.ravel()
    return tp, fp, fn, tn, (tp + tn) / len(y), tp / max(tp + fp, 1), tp / max(tp + fn, 1), tn / max(tn + fp, 1), 2 * tp / max(2 * tp + fp + fn, 1)

print(f"{'τ':>5}{'TP':>4}{'FP':>4}{'FN':>4}{'TN':>4}{'Acc':>7}{'Prec':>7}{'Recall':>8}{'Specif':>8}{'F1':>7}{'3FP+FN':>8}")
for tau in (0.1, 0.3, 0.5, 0.7, 0.75, 0.8, 0.9):
    r = chi_so_theo_nguong(y_te, p_te, tau)
    print(f"{tau:>5}{r[0]:>4}{r[1]:>4}{r[2]:>4}{r[3]:>4}{r[4]:>7.3f}{r[5]:>7.3f}{r[6]:>8.3f}{r[7]:>8.3f}{r[8]:>7.3f}{3*r[1]+r[2]:>8}")

taus = np.linspace(0.02, 0.98, 97)
tab = np.array([chi_so_theo_nguong(y_te, p_te, t) for t in taus])
cost_te = 3 * tab[:, 1] + tab[:, 2]
cost_tr = np.array([3 * chi_so_theo_nguong(y_tr, p_tr, t)[1] + chi_so_theo_nguong(y_tr, p_tr, t)[2] for t in taus])
fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.9))
ax[0].plot(taus, tab[:, 5], label="precision"); ax[0].plot(taus, tab[:, 6], label="recall"); ax[0].plot(taus, tab[:, 7], label="độ đặc hiệu"); ax[0].plot(taus, tab[:, 4], label="accuracy")
ax[0].axvline(0.5, color="gray", ls=":"); ax[0].set(xlabel="ngưỡng τ", title="(a) đánh đổi theo ngưỡng"); ax[0].legend(fontsize=8)
ax[1].plot(taus, cost_tr, label="train"); ax[1].plot(taus, cost_te, label="test"); ax[1].axvline(3 / (3 + 1), color="k", ls="--", label="τ* = 0,75")
ax[1].set(xlabel="ngưỡng τ", ylabel="3·FP + FN", title="(b) tổng chi phí"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
print(f"Ngưỡng tối thiểu hóa chi phí trên tập huấn luyện: τ ≈ {taus[np.argmin(cost_tr)]:.2f} (lý thuyết 0,75)")
print("Chọn τ trên tập kiểm tra là vi phạm 'quy tắc vàng' — chỉ để minh họa; thực tế phải dùng tập kiểm định.")


# ## 5.4. Chất lượng xác suất: log-loss, Brier, ROC/AUC, hiệu chuẩn

# In[ ]:


def log_loss_(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

p_hang = np.full(len(y_te), y_tr.mean())
print(f"Log-loss mô hình        : {log_loss_(y_te, p_te):.4f}")
print(f"Log-loss 'không biết gì' (p=0,5): {np.log(2):.4f} | mô hình hằng (p={y_tr.mean():.3f}): {log_loss_(y_te, p_hang):.4f}  (tài liệu: 0,693 và 0,705)")
print(f"e^(-log-loss) = {np.exp(-log_loss_(y_te, p_te)):.3f}  (trung bình nhân của xác suất gán cho nhãn thật)")
print(f"Điểm Brier: mô hình {np.mean((p_te - y_te)**2):.4f} | mô hình hằng {np.mean((p_hang - y_te)**2):.4f}  (tài liệu: 0,0868 và 0,2559)")


# In[ ]:


# ---- ROC và AUC ----
def duong_roc(y, p):
    thu_tu = np.argsort(-p)                       # xếp mẫu theo xác suất giảm dần
    y_s = y[thu_tu]
    tpr = np.r_[0, np.cumsum(y_s) / y.sum()]
    fpr = np.r_[0, np.cumsum(1 - y_s) / (1 - y).sum()]
    return fpr, tpr

fpr, tpr = duong_roc(y_te, p_te)
trapz = getattr(np, "trapezoid", None) or np.trapz
auc_roc = trapz(tpr, fpr)

# AUC theo cách đếm cặp (dương, âm)
pos, neg = p_te[y_te == 1], p_te[y_te == 0]
so_sanh = pos[:, None] > neg[None, :]
hoa = pos[:, None] == neg[None, :]
auc_cap = np.mean(so_sanh + 0.5 * hoa)
print(f"Số cặp = {len(pos)} × {len(neg)} = {len(pos) * len(neg)} | AUC (đếm cặp) = {auc_cap:.4f} | AUC (diện tích dưới ROC) = {auc_roc:.4f}  (tài liệu: 0,9534)")

# ---- Biểu đồ hiệu chuẩn ----
bin_edges = [0, 0.2, 0.4, 0.6, 0.8, 1.0 + 1e-9]
nhom = np.digitize(p_te, bin_edges) - 1
print(f"\n{'Nhóm theo p':<14}{'Số mẫu':>8}{'p dự đoán TB':>14}{'Tỉ lệ đậu thực':>16}")
xs_cal, ys_cal = [], []
for b in range(5):
    m = nhom == b
    if m.sum() == 0: continue
    print(f"[{bin_edges[b]:.1f}; {min(bin_edges[b+1], 1):.1f}){'':<3}{m.sum():>8}{p_te[m].mean():>14.3f}{y_te[m].mean():>16.3f}")
    xs_cal.append(p_te[m].mean()); ys_cal.append(y_te[m].mean())

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].hist(p_te[y_te == 0], bins=20, alpha=0.7, color="tab:red", label="thực tế rớt"); ax[0].hist(p_te[y_te == 1], bins=20, alpha=0.7, color="tab:blue", label="thực tế đậu")
ax[0].set(xlabel="xác suất dự đoán p", title="(a) phân bố xác suất theo lớp thật"); ax[0].legend()
ax[1].plot(fpr, tpr, label=f"AUC = {auc_roc:.3f}"); ax[1].plot([0, 1], [0, 1], "k--", lw=0.8)
for tau in (0.3, 0.5, 0.7):
    r = chi_so_theo_nguong(y_te, p_te, tau); ax[1].plot(r[1] / 69, r[6], "o", label=f"τ = {tau}")
ax[1].set(xlabel="FPR", ylabel="TPR", title="(b) đường ROC"); ax[1].legend(fontsize=8)
ax[2].plot([0, 1], [0, 1], "k--", lw=0.8, label="hiệu chuẩn hoàn hảo"); ax[2].plot(xs_cal, ys_cal, "o-", label="mô hình")
ax[2].set(xlabel="xác suất dự đoán trung bình", ylabel="tỉ lệ đậu thực tế", title="(c) biểu đồ hiệu chuẩn"); ax[2].legend(fontsize=8)
plt.suptitle("Hình 20", y=1.02); plt.tight_layout(); plt.show()


# > ### 🔍 Giải thích mã (ROC, AUC, hiệu chuẩn)
# > **Đường ROC bằng một lần sắp xếp (`duong_roc`)**
# > - `np.argsort(-p)` trả về thứ tự chỉ số khi xếp $p$ **giảm dần**. Hãy tưởng tượng hạ ngưỡng $\tau$ từ 1 xuống 0: mẫu có $p$ cao nhất được dự đoán "dương" *trước*, rồi lần lượt các mẫu tiếp theo.
# > - `np.cumsum(y_s)` đếm *tích lũy* số mẫu **dương thật** đã được gọi dương → chia cho tổng số dương `y.sum()` được **TPR** sau mỗi bước. `np.cumsum(1 - y_s)` đếm tích lũy mẫu **âm** bị gọi dương (FP) → chia cho tổng số âm được **FPR**.
# > - `np.r_[0, ...]` thêm điểm gốc $(0,0)$ (ngưỡng = 1: không dự đoán dương nào). Diện tích dưới đường này bằng quy tắc hình thang `trapz` là **AUC**.
# >
# > **Đếm cặp bằng broadcasting**
# > - `pos[:, None]` (cột $51\times1$) so với `neg[None, :]` (hàng $1\times69$) cho ma trận $51\times69$ — *mọi cặp (dương, âm) trong một phép so sánh*, không vòng lặp lồng nhau.
# > - `so_sanh + 0.5 * hoa`: cặp xếp đúng (p dương > p âm) tính 1, cặp hòa tính 0,5; trung bình toàn ma trận chính là công thức $\mathrm{AUC}=\frac1{n_+n_-}\sum\big[\mathbb 1(p_i>p_j)+\tfrac12\mathbb 1(p_i=p_j)\big]$. Hai cách tính cho cùng kết quả — AUC là *xác suất một mẫu dương được xếp cao hơn một mẫu âm*.
# >
# > **Hiệu chuẩn bằng `np.digitize`**
# > - `np.digitize(p_te, bin_edges)` cho biết mỗi xác suất rơi vào *khoảng nào* của `bin_edges` (đánh số từ 1), trừ 1 để được chỉ số nhóm $0..4$. Với từng nhóm ta so *xác suất dự đoán trung bình* với *tỉ lệ đậu thực tế*; nếu mô hình hiệu chuẩn tốt, hai số này xấp xỉ nhau (các điểm nằm sát đường chéo).
# > - Thêm `1.0 + 1e-9` ở biên cuối để mẫu có $p=1$ vẫn rơi vào nhóm cuối thay vì ra ngoài.

# ## 5.5. So sánh với scikit-learn và chọn $C$ bằng kiểm định chéo

# In[ ]:


from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, confusion_matrix, classification_report

clf = LogisticRegression(C=1e10, max_iter=10000).fit(X_tr, y_tr)
print("intercept_ =", clf.intercept_, "| coef_ =", clf.coef_, "| classes_ =", clf.classes_)
print("tu cai dat (thang goc): w =", np.round(w_goc, 4))
p_sk = clf.predict_proba(X_te)[:, 1]
p_ours = sigmoid(them_bias(X_te) @ w_goc)
print("max |p_sk - p_ours| =", np.abs(p_sk - p_ours).max().round(4))

clf_mac_dinh = LogisticRegression().fit(X_tr, y_tr)             # C = 1 (mặc định)
print(f"\n{'Mô hình':<26}{'train':>8}{'test':>8}{'log-loss':>10}{'AUC':>8}")
print(f"{'tu cai dat':<26}{np.mean(du_doan(Xb_tr, w_goc) == y_tr):>8.4f}{np.mean(du_doan(Xb_te, w_goc) == y_te):>8.4f}{log_loss(y_te, p_ours):>10.4f}{roc_auc_score(y_te, p_ours):>8.4f}")
for ten, m in [("sklearn C=1e10", clf), ("sklearn mac dinh (C=1)", clf_mac_dinh)]:
    p = m.predict_proba(X_te)[:, 1]
    print(f"{ten:<26}{m.score(X_tr, y_tr):>8.4f}{m.score(X_te, y_te):>8.4f}{log_loss(y_te, p):>10.4f}{roc_auc_score(y_te, p):>8.4f}")
print("C=1: intercept_ =", clf_mac_dinh.intercept_, "coef_ =", clf_mac_dinh.coef_, "(nhỏ hơn: điều chuẩn L2 kéo w về 0)")

perc = Perceptron(random_state=0).fit(X_tr, y_tr)
print(f"Perceptron sklearn: accuracy test = {perc.score(X_te, y_te):.4f}, AUC (điểm số) = {roc_auc_score(y_te, perc.decision_function(X_te)):.4f} — không có xác suất nên không có log-loss")


# In[ ]:


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
print(f"{'C':<8}{'CV accuracy':>20}{'CV log-loss':>14}")
ket_cv = {}
for C in (0.001, 0.01, 0.1, 1, 10, 100, 1000):
    mo_hinh = make_pipeline(StandardScaler(), LogisticRegression(C=C))      # chuẩn hóa nằm TRONG pipeline
    acc = cross_val_score(mo_hinh, X_tr, y_tr, cv=cv)
    ll = -cross_val_score(mo_hinh, X_tr, y_tr, cv=cv, scoring="neg_log_loss")
    ket_cv[C] = ll.mean()
    print(f"{C:<8}{acc.mean():>12.4f} ± {acc.std():.4f}{ll.mean():>14.4f}")
C_tot = min(ket_cv, key=ket_cv.get)
print(f"\nC cho log-loss CV nhỏ nhất: C = {C_tot}")
final = make_pipeline(StandardScaler(), LogisticRegression(C=C_tot)).fit(X_tr, y_tr)
print(f"C = {C_tot} trên tập kiểm tra (đánh giá MỘT lần): accuracy = {final.score(X_te, y_te):.4f} | log-loss = {log_loss(y_te, final.predict_proba(X_te)[:, 1]):.3f}")


# > ### 🔍 Giải thích mã (kiểm định chéo)
# > **Vì sao `StandardScaler` nằm trong `make_pipeline`?** `cross_val_score` chia tập huấn luyện làm 5 phần; mỗi lần dùng 4 phần để *học* và 1 phần để *kiểm định*. Pipeline gọi `fit` của `StandardScaler` **chỉ trên 4 phần học** của lần đó rồi mới áp lên phần kiểm định. Nếu ta chuẩn hóa *toàn bộ* dữ liệu trước rồi mới chia, mean/std đã "nhìn thấy" phần kiểm định ⇒ *rò rỉ thông tin*, điểm CV lạc quan giả tạo.
# >
# > **`StratifiedKFold`** chia sao cho tỉ lệ đậu/rớt ở mỗi phần gần bằng nhau. **`scoring="neg_log_loss"`** — scikit-learn quy ước *điểm càng lớn càng tốt* nên log-loss được trả về dưới dạng âm; ta thêm dấu `-` ở phía trước để có lại log-loss dương.
# >
# > **Chọn $C$ theo log-loss** (`min(ket_cv, key=ket_cv.get)` lấy khóa có giá trị nhỏ nhất) vì log-loss phân biệt các giá trị $C$ rõ hơn accuracy (accuracy dao động ±1,8 điểm nên không giúp chọn được). Chỉ **sau** khi chọn xong mới huấn luyện lại trên toàn bộ tập huấn luyện và đánh giá trên tập kiểm tra *đúng một lần*.
# >
# > **Liên hệ $C$ và $\lambda$:** hàm mục tiêu của thư viện là $C\sum_il_i+\frac12\|\mathbf w\|^2$, tương đương $\lambda=\dfrac1{Cn}$ trong cách viết của tài liệu. $C$ càng nhỏ, điều chuẩn càng mạnh.

# In[ ]:


print(confusion_matrix(y_te, clf.predict(X_te), labels=[0, 1]))
print(classification_report(y_te, (p_ours >= 0.5).astype(int), labels=[0, 1], target_names=["rot", "dau"], digits=3))

print("--- Kiểm tra nhanh mục 5 ---")
tn, fp, fn, tp = np.array([[70, 10], [5, 15]]).ravel()
print(f"2) Accuracy = {(tp+tn)/100:.2f} | Precision = {tp/(tp+fp):.2f} | Recall = {tp/(tp+fn):.2f} | Độ đặc hiệu = {tn/(tn+fp):.3f} | F1 = {2*tp/(2*tp+fp+fn):.3f}")
print(f"4) c_FN = 9, c_FP = 1: τ* = {1/(1+9):.2f} (thấp hơn nhiều so với 0,5 vì bỏ sót nguy hiểm hơn)")
print(f"5) AUC = {190/200:.2f} (190 trong 200 cặp được xếp đúng)")

# Câu 6: AUC không đổi khi thay p bằng p^2 (giữ thứ tự) nhưng log-loss thì đổi
print(f"6) AUC(p) = {roc_auc_score(y_te, p_te):.4f} | AUC(p²) = {roc_auc_score(y_te, p_te**2):.4f} | log-loss(p) = {log_loss(y_te, p_te):.3f} | log-loss(p²) = {log_loss(y_te, np.clip(p_te**2, 1e-12, 1)):.3f}")


# ---
# # Tổng kết
# 
# | Thành phần | Hồi quy tuyến tính | Perceptron | Logistic Regression |
# | --- | --- | --- | --- |
# | **Mô hình** | $\hat y=\mathbf{xw}$ | $\hat y=\mathrm{sign}(\mathbf{xw})$ | $p=\sigma(\mathbf{xw})$ |
# | **Hàm mất mát** | MSE | $\max(0,-m)$ | cross-entropy $-\frac1n\sum[y\ln p+(1-y)\ln(1-p)]$ |
# | **Gradient** | $\frac2n\mathbf X^T(\mathbf{Xw}-\mathbf y)$ | $-y_i\mathbf x_i^T$ (mẫu sai) | $\frac1n\mathbf X^T(\mathbf p-\mathbf y)$ |
# | **Tối ưu** | GD / phương trình chuẩn | GD từng mẫu | GD (batch/SGD/mini-batch), L-BFGS |
# 
# **Bài học chính**
# 
# 1. Perceptron chỉ cho nhãn cứng; Logistic Regression cho **xác suất** $p=\sigma(\mathbf{xw})$ — điểm số $s$ chính là **log-odds**. Đường biên vẫn là siêu phẳng $\mathbf{xw}=0$, nhưng giờ có thêm các đường đồng mức xác suất song song.
# 2. Trọng số có ý nghĩa định lượng: $e^{w_j}$ là **odds ratio**; độ lớn của $\mathbf w$ điều khiển độ tự tin của xác suất.
# 3. MSE + sigmoid gây "gradient tắt" và không lồi; **cross-entropy** (từ hợp lý cực đại) cho gradient gọn $\frac1n\mathbf X^T(\mathbf p-\mathbf y)$, cùng dạng với hồi quy tuyến tính.
# 4. $J$ **lồi** (Hessian $\frac1n\mathbf X^T\mathbf{SX}\succeq0$): mọi khởi tạo đều đến cùng nghiệm; chuẩn hóa đặc trưng giảm số điều kiện (765 → 5,75) và cho phép dùng $\eta$ lớn hơn nhiều.
# 5. Dữ liệu phân tách hoàn hảo ⇒ $\|\mathbf w\|\to\infty$; **điều chuẩn L2** đảm bảo nghiệm hữu hạn. XOR/vòng tròn cần đặc trưng đa thức (kèm L2).
# 6. Đánh giá bài bản: **log-loss** nhạy với overfitting hơn accuracy; **ROC/AUC** đo khả năng xếp hạng (không phụ thuộc ngưỡng); **biểu đồ hiệu chuẩn** kiểm tra xác suất có đúng nghĩa không; chọn ngưỡng theo $\tau^*=\frac{c_{FP}}{c_{FP}+c_{FN}}$; chọn siêu tham số bằng **kiểm định chéo** trong pipeline.
# 
# **Hướng tiếp theo:** ghép nhiều lớp Logistic/Perceptron thành **mạng nơ-ron** để mô hình tự học đặc trưng thay vì ta phải tự đoán.
