import numpy as np

def them_bias(X):
    """Thêm cột 1 vào đầu ma trận đặc trưng: (n, d) -> (n, d+1)."""
    return np.hstack([np.ones((X.shape[0], 1)), X])

def sigmoid(s):
    """sigma(s) = 1 / (1 + e^{-s}), viết qua tanh để không bị tràn số: sigma(s) = 0.5 * (1 + tanh(s / 2))."""
    return 0.5 * (1.0 + np.tanh(0.5 * s))

def mat_mat(Xb, y, w, lam=0.0):
    """Cross-entropy trung bình J(w), cộng thêm điều chuẩn hóa L2: (lam/2) * ||w_1..w_d||^2."""
    p = np.clip(sigmoid(Xb @ w), 1e-12, 1 - 1e-12)      # tránh log(0)
    ce = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return ce + 0.5 * lam * np.sum(w[1:] ** 2)           # không phạt hệ số chệch w_0

def gradient(Xb, y, w, lam=0.0):
    """Gradient của J: (1/n) * X^T (p - y) + lam * [0, w_1, ..., w_d]."""
    p = sigmoid(Xb @ w)
    g = Xb.T @ (p - y) / len(y)
    g[1:] += lam * w[1:]
    return g

def du_doan_xac_suat(Xb, w):
    """Xác suất p = sigma(Xw) của lớp dương."""
    return sigmoid(Xb @ w)

def du_doan(Xb, w, nguong=0.5):
    """Nhãn dự đoán: 1 nếu p >= nguong, ngược lại 0."""
    return (du_doan_xac_suat(Xb, w) >= nguong).astype(int)

def huan_luyen(Xb, y, eta=0.1, so_epoch=1000, kich_thuoc_batch=None, seed=0, lam=0.0, w0=None):
    """Gradient Descent: toàn bộ (batch), từng mẫu (SGD) hoặc mini-batch."""
    n, d1 = Xb.shape
    b = n if kich_thuoc_batch is None else kich_thuoc_batch
    rng = np.random.default_rng(seed)
    w = np.zeros(d1) if w0 is None else np.array(w0, dtype=float)   # mặc định w = 0
    lich_su = [mat_mat(Xb, y, w, lam)]                 # J sau mỗi epoch
    for epoch in range(so_epoch):
        idx = rng.permutation(n) if b < n else np.arange(n)
        for k in range(0, n, b):
            lo = idx[k:k + b]                          # chỉ số của mini-batch hiện tại
            w = w - eta * gradient(Xb[lo], y[lo], w, lam)
        lich_su.append(mat_mat(Xb, y, w, lam))
    return w, lich_su

def chia_train_test(X, y, ti_le_test=0.3, seed=0):
    """Xáo trộn rồi chia dữ liệu thành tập huấn luyện và tập kiểm tra."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = int(len(y) * ti_le_test)
    return X[idx[n_test:]], X[idx[:n_test]], y[idx[n_test:]], y[idx[:n_test]]

def chuan_hoa(X_tr, X_khac=None):
    """Chuẩn hóa z = (x - mu) / s, với mu và s tính CHỈ từ tập huấn luyện."""
    mu, s = X_tr.mean(axis=0), X_tr.std(axis=0)
    Z_tr = (X_tr - mu) / s
    return (Z_tr, mu, s) if X_khac is None else (Z_tr, (X_khac - mu) / s, mu, s)
