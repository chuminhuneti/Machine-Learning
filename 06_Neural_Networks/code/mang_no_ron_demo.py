"""Minh họa mạng nơ-ron nhân tạo từ đầu bằng NumPy (chạy: python mang_no_ron_demo.py [--plot])."""
import sys
VE_HINH = "--plot" in sys.argv   # thêm --plot để vẽ hình (cần matplotlib)


# ======================================================================
# # Mạng nơ-ron nhân tạo từ đầu bằng NumPy — bản minh họa
#
# Tài liệu này chạy lại, **trong một file duy nhất**, các nội dung chính của chương *Mạng nơ-ron nhân tạo*:
#
# | Bước | Nội dung | Mục trong chương |
# |---|---|---|
# | 1 | Softmax ổn định số học, thêm cột bias | 1–2 |
# | 2 | Hàm kích hoạt và khởi tạo trọng số (Xavier, He) | 2–3 |
# | 3 | Lan truyền thuận dạng ma trận | 2 |
# | 4 | Lan truyền ngược | 3 |
# | 5 | Kiểm tra gradient bằng sai phân hữu hạn | 3 |
# | 6 | Huấn luyện mini-batch với Adam, điều chuẩn hóa L2 | 3–4 |
# | 7 | XOR: mô hình tuyến tính thất bại, mạng có lớp ẩn thành công | 1–2 |
# | 8 | Xoắn ốc 3 nhánh: Softmax Regression so với MLP | 4–6 |
# | 9 | Ma trận nhầm lẫn, precision, recall, F1, log-loss | 5 |
# | 10 | Gradient biến mất: sigmoid so với ReLU | 4 |
#
# **Quy ước:** ma trận là chữ hoa (`X`, `W`, `D`), vector là chữ thường (`x`, `w`). Mỗi mẫu là một **hàng** của `X`, nên kích thước `X` là $n \times d$. Ma trận trọng số lớp $l$ là `W[l]` kích thước $(m_{l-1}+1)\times m_l$, **hàng đầu tiên là bias**.
# ======================================================================

# ======================================================================
# ## 1. Hàm phụ trợ: thêm bias và softmax ổn định
#
# **Thêm bias.** Thay vì lưu riêng vector bias $\mathbf{b}$, ta thêm một cột toàn số $1$ vào đầu `A`, rồi hàng đầu của `W` đóng vai bias:
# $\tilde{\mathbf{A}}\mathbf{W} = \mathbf{A}\mathbf{W}_{[1:]} + \mathbf{1}\,\mathbf{w}_{[0]}$.
#
# **Softmax ổn định.** Công thức $p_k = e^{z_k}/\sum_j e^{z_j}$ tràn số khi $z_k$ lớn (ví dụ $e^{1000}$). Vì softmax không đổi khi trừ cùng một hằng số khỏi mọi $z_k$, ta trừ giá trị lớn nhất của mỗi hàng: khi đó số mũ lớn nhất bằng $e^0 = 1$ và không bao giờ tràn.
# ======================================================================
import numpy as np
np.set_printoptions(precision=4, suppress=True)

def them_bias(A):
    # Ghép một cột số 1 vào bên trái: (n, m) -> (n, m+1)
    return np.hstack([np.ones((len(A), 1)), A])

def softmax(Z):
    # Trừ max theo từng hàng để tránh tràn số; kết quả không đổi vì tử và mẫu cùng nhân e^{-max}
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)

# Kiểm tra nhanh: điểm số cực lớn vẫn cho xác suất hợp lệ
print(softmax(np.array([[1000., 1001., 999.]])))


# ======================================================================
# ## 2. Hàm kích hoạt và khởi tạo trọng số
#
# Mỗi hàm kích hoạt $\varphi$ đi kèm đạo hàm. Điểm đáng chú ý: ta viết đạo hàm **theo đầu ra $a=\varphi(z)$** (không theo $z$) khi có thể, vì $a$ đã được tính ở lượt thuận nên khỏi tính lại:
#
# - $\tanh'(z) = 1 - a^2$
# - $\sigma'(z) = a(1-a)$
# - $\mathrm{ReLU}'(z) = \mathbb{1}[z>0]$ (dùng $z$)
#
# **Khởi tạo.** Nếu mọi trọng số bằng nhau thì mọi nơ-ron trong một lớp học y hệt nhau (đối xứng không bị phá). Ngẫu nhiên còn phải có *độ lớn phù hợp* để tín hiệu không co lại hay phình ra qua các lớp:
#
# - **Xavier** (cho tanh/sigmoid): $U(-\ell,\ell)$ với $\ell=\sqrt{6/(m_{in}+m_{out})}$.
# - **He** (cho ReLU): $N(0,\,2/m_{in})$; hệ số $2$ bù cho việc ReLU triệt tiêu một nửa số đầu ra.
#
# Hàng bias khởi tạo bằng $0$.
# ======================================================================
# Mỗi hàm kích hoạt: (phi, dphi_theo_A_va_Z)
KICH_HOAT = {
    "tanh":    (np.tanh,                       lambda Z, A: 1 - A**2),
    "sigmoid": (lambda Z: 1/(1 + np.exp(-Z)),  lambda Z, A: A * (1 - A)),
    "relu":    (lambda Z: np.maximum(0, Z),    lambda Z, A: (Z > 0).astype(float)),
}

def khoi_tao(kich_thuoc, kich_hoat="tanh", seed=0):
    # kich_thuoc = [d, m1, ..., K]; trả về danh sách W, W[l] có hình (m_{l-1}+1, m_l)
    rng = np.random.default_rng(seed)
    Ws = []
    for m_vao, m_ra in zip(kich_thuoc[:-1], kich_thuoc[1:]):
        if kich_hoat == "relu":                       # He
            W = rng.normal(0, np.sqrt(2 / m_vao), (m_vao + 1, m_ra))
        else:                                         # Xavier
            gh = np.sqrt(6 / (m_vao + m_ra))
            W = rng.uniform(-gh, gh, (m_vao + 1, m_ra))
        W[0] = 0                                      # hàng bias = 0
        Ws.append(W)
    return Ws

Ws = khoi_tao([2, 4, 3])
print([W.shape for W in Ws], "-> số tham số =", sum(W.size for W in Ws))


# ======================================================================
# ## 3. Lan truyền thuận
#
# Với lớp $l=1,\dots,L$: $\mathbf{Z}^{(l)}=\tilde{\mathbf{A}}^{(l-1)}\mathbf{W}^{(l)}$, rồi $\mathbf{A}^{(l)}=\varphi(\mathbf{Z}^{(l)})$ ở các lớp ẩn, còn lớp cuối dùng softmax để ra $\mathbf{P}$.
#
# Ta **lưu lại** mọi `A` và `Z` vì lan truyền ngược sẽ cần chúng (đây là lý do huấn luyện tốn bộ nhớ theo độ sâu).
# ======================================================================
def lan_truyen_thuan(X, Ws, kich_hoat="tanh"):
    phi = KICH_HOAT[kich_hoat][0]
    As, Zs = [X], [None]               # As[0] = X; Zs[0] không dùng (để chỉ số lớp khớp nhau)
    for l, W in enumerate(Ws, start=1):
        Z = them_bias(As[-1]) @ W
        Zs.append(Z)
        As.append(softmax(Z) if l == len(Ws) else phi(Z))   # lớp cuối: softmax
    return As, Zs                      # As[-1] chính là P

def cross_entropy(P, Y):
    # J = -(1/n) sum_i sum_k y_ik ln p_ik ; cộng 1e-12 để tránh ln(0)
    return -np.mean(np.sum(Y * np.log(P + 1e-12), axis=1))

X_thu = np.random.default_rng(1).normal(size=(5, 2))
As, Zs = lan_truyen_thuan(X_thu, Ws)
print("P có tổng mỗi hàng =", As[-1].sum(axis=1))


# ======================================================================
# ## 4. Lan truyền ngược
#
# Gọi $\mathbf{D}^{(l)}=\partial J/\partial\mathbf{Z}^{(l)}$ là **tín hiệu lỗi** của lớp $l$ ($n\times m_l$). Hai kết quả cốt lõi của chương:
#
# 1. **Lớp ra** (softmax + cross-entropy) đơn giản đến bất ngờ: $\mathbf{D}^{(L)}=(\mathbf{P}-\mathbf{Y})/n$.
# 2. **Đi lùi một lớp:** $\nabla\mathbf{W}^{(l)}=\tilde{\mathbf{A}}^{(l-1)\,T}\mathbf{D}^{(l)}$ và $\mathbf{D}^{(l-1)}=\big(\mathbf{D}^{(l)}\mathbf{V}^{(l)T}\big)\odot\varphi'(\mathbf{Z}^{(l-1)})$.
#
# **Chỗ dễ sai nhất — ma trận $\mathbf{V}$.** $\mathbf{V}^{(l)}$ là $\mathbf{W}^{(l)}$ **bỏ hàng bias** (`W[1:]`). Lý do: cột số $1$ ta thêm vào không phụ thuộc đầu ra lớp trước, nên lỗi không thể truyền ngược qua nó. Nếu quên bỏ hàng này, kích thước ma trận sẽ không khớp (hoặc, tệ hơn, khớp nhầm).
#
# $\odot$ là nhân từng phần tử; nó đến từ việc $\varphi$ áp dụng độc lập lên từng phần tử của $\mathbf{Z}$.
# ======================================================================
def lan_truyen_nguoc(As, Zs, Y, Ws, kich_hoat="tanh", lam=0.0):
    # Trả về danh sách gradient grads[l-1] ứng với Ws[l-1]
    n, L = len(Y), len(Ws)
    dphi = KICH_HOAT[kich_hoat][1]
    D = (As[-1] - Y) / n                         # D^(L) = (P - Y)/n
    grads = [None] * L
    for l in range(L, 0, -1):                    # l = L, L-1, ..., 1
        grads[l-1] = them_bias(As[l-1]).T @ D    # (m_{l-1}+1, n) @ (n, m_l)
        if lam:                                  # L2 chỉ phạt trọng số, KHÔNG phạt bias (hàng 0)
            grads[l-1][1:] += lam * Ws[l-1][1:]
        if l > 1:
            V = Ws[l-1][1:]                      # bỏ hàng bias
            D = (D @ V.T) * dphi(Zs[l-1], As[l-1])   # đẩy lỗi về lớp trước rồi nhân đạo hàm kích hoạt
    return grads


# ======================================================================
# ## 5. Kiểm tra gradient bằng sai phân hữu hạn
#
# Cài đặt lan truyền ngược rất dễ sai mà vẫn chạy được (mạng vẫn "học", chỉ học tệ). Cách kiểm tra đáng tin: so với đạo hàm số
# $$\frac{\partial J}{\partial w}\approx\frac{J(w+\varepsilon)-J(w-\varepsilon)}{2\varepsilon}.$$
# Sai phân **trung tâm** có sai số $O(\varepsilon^2)$, tốt hơn sai phân một phía $O(\varepsilon)$. Ta đo sai số tương đối $\dfrac{\|g_{ta}-g_{so}\|}{\|g_{ta}\|+\|g_{so}\|}$; với `float64` và $\varepsilon=10^{-5}$ kỳ vọng cỡ $10^{-8}$ hoặc nhỏ hơn.
#
# Lưu ý: phải kiểm tra bằng **dữ liệu nhỏ, `float64`**, và tắt dropout/ngẫu nhiên.
# ======================================================================
def kiem_tra_gradient(X, Y, Ws, kich_hoat="tanh", eps=1e-5):
    As, Zs = lan_truyen_thuan(X, Ws, kich_hoat)
    g_ta = lan_truyen_nguoc(As, Zs, Y, Ws, kich_hoat)
    sai_so = []
    for W, g in zip(Ws, g_ta):
        g_so = np.zeros_like(W)
        for idx in np.ndindex(*W.shape):         # duyệt từng phần tử của W
            goc = W[idx]
            W[idx] = goc + eps; Jc = cross_entropy(lan_truyen_thuan(X, Ws, kich_hoat)[0][-1], Y)
            W[idx] = goc - eps; Jt = cross_entropy(lan_truyen_thuan(X, Ws, kich_hoat)[0][-1], Y)
            W[idx] = goc                         # trả lại giá trị gốc!
            g_so[idx] = (Jc - Jt) / (2 * eps)
        sai_so.append(np.linalg.norm(g - g_so) / (np.linalg.norm(g) + np.linalg.norm(g_so)))
    return sai_so

rng = np.random.default_rng(0)
X5 = rng.normal(size=(6, 3)); Y5 = np.eye(3)[rng.integers(0, 3, 6)]
for kh in ("tanh", "sigmoid", "relu"):
    W5 = khoi_tao([3, 5, 4, 3], kh, seed=1)
    for W in W5: W[0] = rng.normal(0, 0.5, W.shape[1])   # bias khác 0 để kiểm tra cả hàng bias
    print(f"{kh:8s} sai số tương đối từng lớp:", ["%.1e" % e for e in kiem_tra_gradient(X5, Y5, W5, kh)])


# ======================================================================
# ## 6. Huấn luyện: mini-batch và Adam
#
# **Mini-batch:** mỗi epoch xáo trộn dữ liệu rồi cập nhật theo từng khối `b` mẫu. Gradient ở mỗi khối là ước lượng nhiễu của gradient toàn bộ, nhưng rẻ hơn và nhiễu này giúp thoát điểm yên ngựa.
#
# **Adam** giữ hai trung bình trượt cho từng tham số: động lượng $\mathbf{m}$ (trung bình của gradient) và $\mathbf{v}$ (trung bình của gradient bình phương):
# $$\mathbf{m}\leftarrow\beta_1\mathbf{m}+(1-\beta_1)\mathbf{g},\quad \mathbf{v}\leftarrow\beta_2\mathbf{v}+(1-\beta_2)\mathbf{g}^{2},\quad \mathbf{W}\leftarrow\mathbf{W}-\eta\,\frac{\hat{\mathbf{m}}}{\sqrt{\hat{\mathbf{v}}}+\epsilon}.$$
# **Hiệu chỉnh độ chệch:** vì $\mathbf{m},\mathbf{v}$ bắt đầu từ $0$, vài bước đầu chúng nhỏ hơn giá trị thật; chia cho $1-\beta^t$ ($\hat{\mathbf{m}}=\mathbf{m}/(1-\beta_1^t)$) sửa điều đó. Chia cho $\sqrt{\hat{\mathbf{v}}}$ làm mỗi tham số có bước cập nhật tự điều chỉnh theo độ lớn gradient của nó.
# ======================================================================
def huan_luyen(X, Y, kich_thuoc, kich_hoat="tanh", eta=0.01, so_epoch=100, b=32,
               lam=0.0, seed=0, Ws=None, ghi_lich_su=True):
    n = len(X)
    rng = np.random.default_rng(seed)
    Ws = khoi_tao(kich_thuoc, kich_hoat, seed) if Ws is None else Ws
    m = [np.zeros_like(W) for W in Ws]           # động lượng bậc 1
    v = [np.zeros_like(W) for W in Ws]           # động lượng bậc 2
    b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
    lich_su = []
    for epoch in range(so_epoch):
        idx = rng.permutation(n)                 # xáo trộn mỗi epoch
        for k in range(0, n, b):
            lo = idx[k:k+b]
            As, Zs = lan_truyen_thuan(X[lo], Ws, kich_hoat)
            g = lan_truyen_nguoc(As, Zs, Y[lo], Ws, kich_hoat, lam)
            t += 1
            for i in range(len(Ws)):
                m[i] = b1 * m[i] + (1 - b1) * g[i]
                v[i] = b2 * v[i] + (1 - b2) * g[i]**2
                m_hat = m[i] / (1 - b1**t)       # hiệu chỉnh độ chệch
                v_hat = v[i] / (1 - b2**t)
                Ws[i] -= eta * m_hat / (np.sqrt(v_hat) + eps)
        if ghi_lich_su:
            lich_su.append(cross_entropy(lan_truyen_thuan(X, Ws, kich_hoat)[0][-1], Y))
    return Ws, lich_su

def du_doan(X, Ws, kich_hoat="tanh"):
    return lan_truyen_thuan(X, Ws, kich_hoat)[0][-1].argmax(axis=1)   # lớp có xác suất lớn nhất


# ======================================================================
# ## 7. XOR: vì sao cần lớp ẩn
#
# Quan sát then chốt của chương: **Softmax Regression chính là mạng không có lớp ẩn** (`kich_thuoc=[d, K]`). Vậy ta dùng chung hàm `huan_luyen` cho cả hai và chỉ đổi kích thước. XOR có bốn góc, hai lớp; không đường thẳng nào tách được.
# ======================================================================
def tao_xor(n, nhieu=0.25, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, (n, 2)).astype(float)       # bốn góc (0,0),(0,1),(1,0),(1,1)
    y = (X[:, 0] != X[:, 1]).astype(int)               # XOR
    return X + rng.normal(0, nhieu, X.shape), y

Xa, ya = tao_xor(400, seed=0); Xb, yb = tao_xor(400, seed=1)
Ya = np.eye(2)[ya]
for ten, ks in (("Softmax (không lớp ẩn) [2,2]", [2, 2]), ("MLP [2,8,2]", [2, 8, 2])):
    Ws_, ls = huan_luyen(Xa, Ya, ks, so_epoch=150)
    print(f"{ten:30s} J cuối = {ls[-1]:.3f}   acc kiểm tra = {np.mean(du_doan(Xb, Ws_) == yb):.3f}")

if VE_HINH:
    import matplotlib.pyplot as plt

    def ve_vung_quyet_dinh(ax, Ws, kich_hoat, X, y, tieu_de):
        # Tính lớp dự đoán trên lưới điểm rồi tô màu: đây là "vùng quyết định"
        x0, x1 = X[:, 0].min() - .5, X[:, 0].max() + .5
        y0, y1 = X[:, 1].min() - .5, X[:, 1].max() + .5
        gx, gy = np.meshgrid(np.linspace(x0, x1, 200), np.linspace(y0, y1, 200))
        luoi = np.c_[gx.ravel(), gy.ravel()]
        ax.contourf(gx, gy, du_doan(luoi, Ws, kich_hoat).reshape(gx.shape), alpha=.25,
                    levels=np.arange(-.5, y.max() + 1.5), cmap="coolwarm" if y.max() == 1 else "viridis")
        ax.scatter(X[:, 0], X[:, 1], c=y, s=8, cmap="coolwarm" if y.max() == 1 else "viridis")
        ax.set_title(tieu_de); ax.set_xticks([]); ax.set_yticks([])

    fig, axs = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, (ten, ks) in zip(axs, (("Softmax [2,2]", [2, 2]), ("MLP [2,8,2]", [2, 8, 2]))):
        W_, _ = huan_luyen(Xa, Ya, ks, so_epoch=150)
        ve_vung_quyet_dinh(ax, W_, "tanh", Xb, yb, ten)
    plt.tight_layout(); plt.show()


# ======================================================================
# ## 8. Ba nhánh xoắn ốc: Softmax Regression so với MLP
#
# Bộ dữ liệu khó hơn: ba nhánh xoắn ốc đan vào nhau (bộ sinh dữ liệu ở đây là bản đơn giản hóa, dễ hơn bộ trong chương nên độ chính xác tuyệt đối cao hơn; điều cần xem là *khoảng cách* giữa hai mô hình). Ta so sánh Softmax Regression và MLP `[2,16,16,3]` trên cùng dữ liệu, cùng thuật toán tối ưu. Dữ liệu được **chuẩn hóa bằng thống kê của tập huấn luyện** rồi dùng chính các số đó cho tập kiểm tra (nếu tính trên cả hai tập thì thông tin từ tập kiểm tra rò vào quá trình huấn luyện).
# ======================================================================
def tao_xoan_oc(n_moi_lop, nhieu=0.25, seed=0, K=3):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for k in range(K):
        r = np.linspace(0.1, 1, n_moi_lop)                                   # bán kính tăng dần
        goc = np.linspace(k * 2*np.pi/K, k * 2*np.pi/K + 3.0, n_moi_lop) + rng.normal(0, nhieu, n_moi_lop)
        X.append(np.c_[r * np.sin(goc), r * np.cos(goc)]); y += [k] * n_moi_lop
    return np.vstack(X), np.array(y)

Xtr, ytr = tao_xoan_oc(200, seed=0); Xte, yte = tao_xoan_oc(400, seed=1)
mu, sd = Xtr.mean(0), Xtr.std(0)                    # CHỈ từ tập huấn luyện
Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
Ytr = np.eye(3)[ytr]

ket_qua = {}
for ten, ks in (("Softmax", [2, 3]), ("MLP [2,16,16,3]", [2, 16, 16, 3])):
    accs = []
    for seed in range(5):                           # 5 khởi tạo: hàm mất mát không lồi nên kết quả dao động
        W_, _ = huan_luyen(Xtr, Ytr, ks, so_epoch=200, seed=seed)
        accs.append(np.mean(du_doan(Xte, W_) == yte))
    ket_qua[ten] = accs
    print(f"{ten:18s} acc kiểm tra: trung bình {np.mean(accs):.3f} ± {np.std(accs):.3f}")

if VE_HINH:
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, (ten, ks) in zip(axs, (("Softmax", [2, 3]), ("MLP [2,16,16,3]", [2, 16, 16, 3]))):
        W_, _ = huan_luyen(Xtr, Ytr, ks, so_epoch=200, seed=0)
        ve_vung_quyet_dinh(ax, W_, "tanh", Xte, yte, ten)
    plt.tight_layout(); plt.show()


# ======================================================================
# ## 9. Chỉ số đánh giá
#
# Độ chính xác (accuracy) che giấu lỗi theo từng lớp, nhất là khi lớp mất cân bằng. Từ **ma trận nhầm lẫn** $\mathbf{C}=\mathbf{Y}^{T}\hat{\mathbf{Y}}$ ($c_{kj}$ = số mẫu thật thuộc lớp $k$ bị đoán là lớp $j$):
#
# - $\mathrm{Prec}_k=c_{kk}/\sum_i c_{ik}$ (chia tổng **cột**: trong số mẫu bị đoán là $k$, bao nhiêu đúng),
# - $\mathrm{Rec}_k=c_{kk}/\sum_j c_{kj}$ (chia tổng **hàng**: trong số mẫu thật là $k$, bao nhiêu được tìm ra),
# - $\mathrm{F1}_k=2\,\mathrm{Prec}_k\mathrm{Rec}_k/(\mathrm{Prec}_k+\mathrm{Rec}_k)$; F1 macro là trung bình các $\mathrm{F1}_k$ (mọi lớp nặng như nhau).
#
# Phép nhân `Y.T @ Yhat` xây ma trận nhầm lẫn mà không cần vòng lặp: tích của hai ma trận one-hot đếm cặp (thật, đoán).
# ======================================================================
def ma_tran_nham_lan(y_that, y_doan, K):
    return np.eye(K)[y_that].T @ np.eye(K)[y_doan]           # Y^T @ Yhat

def bao_cao(C):
    tp = np.diag(C)
    prec = tp / np.maximum(C.sum(axis=0), 1)                  # max(.,1): tránh chia 0 khi lớp không được đoán lần nào
    rec = tp / np.maximum(C.sum(axis=1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return prec, rec, f1

W_mlp, _ = huan_luyen(Xtr, Ytr, [2, 16, 16, 3], so_epoch=200, seed=0)
C = ma_tran_nham_lan(yte, du_doan(Xte, W_mlp), 3)
prec, rec, f1 = bao_cao(C)
print("Ma trận nhầm lẫn:\n", C.astype(int))
print("precision:", prec, "\nrecall   :", rec, "\nF1       :", f1)
print("accuracy =", np.trace(C) / C.sum(), "  F1 macro =", f1.mean())
print("log-loss =", cross_entropy(lan_truyen_thuan(Xte, W_mlp)[0][-1], np.eye(3)[yte]))


# ======================================================================
# ## 10. Gradient biến mất: sigmoid so với ReLU
#
# Từ công thức đi lùi, chuẩn của tín hiệu lỗi bị chặn bởi tích các thừa số $\gamma\,\|\mathbf{V}\|_2$ qua từng lớp, với $\gamma=\max|\varphi'|$. Sigmoid có $\gamma=1/4$, nên mỗi lớp thêm vào làm tín hiệu co lại ít nhất khoảng 4 lần nếu $\|\mathbf{V}\|_2$ không đủ lớn. Ta đo **chuẩn gradient của từng lớp** ngay sau khởi tạo trên mạng 8 lớp ẩn: lớp gần đầu vào (lớp 1) nhận được bao nhiêu so với lớp gần đầu ra?
# ======================================================================
def chuan_gradient_tung_lop(kich_hoat, so_lop_an=8, rong=16, seed=0):
    ks = [2] + [rong] * so_lop_an + [3]
    Ws_ = khoi_tao(ks, kich_hoat, seed)
    As_, Zs_ = lan_truyen_thuan(Xtr, Ws_, kich_hoat)
    g = lan_truyen_nguoc(As_, Zs_, Ytr, Ws_, kich_hoat)
    return np.array([np.linalg.norm(x) for x in g])

for kh in ("sigmoid", "tanh", "relu"):
    ch = chuan_gradient_tung_lop(kh)
    print(f"{kh:8s} chuẩn gradient lớp 1..9:", ["%.1e" % c for c in ch], f"| lớp1/lớp9 = {ch[0]/ch[-1]:.2e}")

if VE_HINH:
    plt.figure(figsize=(5.5, 3.4))
    for kh in ("sigmoid", "tanh", "relu"):
        plt.semilogy(range(1, 10), chuan_gradient_tung_lop(kh), "o-", label=kh)
    plt.xlabel("lớp (1 = gần đầu vào)"); plt.ylabel("‖∇W‖ (log)"); plt.legend(); plt.tight_layout(); plt.show()


# ======================================================================
# ## Tóm tắt
#
# - Softmax Regression = mạng không lớp ẩn; mạng nơ-ron = Softmax Regression áp lên đặc trưng do các lớp ẩn **tự học**.
# - Lan truyền ngược chỉ là quy tắc chuỗi: $\mathbf{D}^{(L)}=(\mathbf{P}-\mathbf{Y})/n$, rồi lặp $\nabla\mathbf{W}=\tilde{\mathbf{A}}^{T}\mathbf{D}$ và $\mathbf{D}\leftarrow(\mathbf{D}\mathbf{V}^{T})\odot\varphi'$.
# - Luôn **kiểm tra gradient** bằng sai phân hữu hạn trước khi tin vào cài đặt.
# - Hàm mất mát của mạng không lồi: chạy nhiều khởi tạo và báo cáo trung bình ± độ lệch chuẩn.
# - Chuẩn hóa chỉ bằng thống kê tập huấn luyện; đánh giá bằng nhiều chỉ số, không chỉ accuracy.
# - Mạng sâu cần hàm kích hoạt và khởi tạo phù hợp (ReLU + He) để gradient không biến mất.
# ======================================================================