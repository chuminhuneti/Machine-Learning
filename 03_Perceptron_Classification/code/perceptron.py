import numpy as np

def them_bias(X):
    """Thêm cột 1 vào đầu ma trận đặc trưng: (n, d) -> (n, d+1)."""
    return np.hstack([np.ones((X.shape[0], 1)), X])

def du_doan(Xb, w):
    """Nhãn dự đoán y_hat = sign(Xw), với quy ước sign(0) = +1."""
    s = Xb @ w
    return np.where(s >= 0, 1, -1)

def huan_luyen(Xb, y, eta=1.0, t_max=100):
    """Thuật toán Perceptron: cập nhật ngay sau mỗi mẫu khi y_i * (x_i w) <= 0."""
    n, d1 = Xb.shape
    w = np.zeros(d1)                      # khởi tạo w = 0
    lich_su = []                          # số lần cập nhật ở mỗi epoch
    quy_dao = []                          # w sau mỗi lần cập nhật
    for epoch in range(t_max):
        so_cap_nhat = 0
        for i in range(n):
            m = y[i] * (Xb[i] @ w)        # margin của mẫu i
            if m <= 0:
                w = w + eta * y[i] * Xb[i]
                quy_dao.append(w.copy())
                so_cap_nhat += 1
        lich_su.append(so_cap_nhat)
        if so_cap_nhat == 0:              # một epoch không cập nhật: hội tụ
            break
    return w, lich_su, quy_dao

def huan_luyen_tui(Xb, y, eta=1.0, t_max=100, seed=0):
    """Perceptron + thuật toán túi: giữ lại w có ít mẫu sai nhất."""
    n = Xb.shape[0]
    rng = np.random.default_rng(seed)
    w = np.zeros(Xb.shape[1])
    w_tui = w.copy()
    loi_tui = np.sum(du_doan(Xb, w) != y)
    lich_su = []                          # (số mẫu sai của w, số mẫu sai của w_tui)
    for epoch in range(t_max):
        for i in rng.permutation(n):      # xáo trộn thứ tự mẫu ở mỗi epoch
            if y[i] * (Xb[i] @ w) <= 0:
                w = w + eta * y[i] * Xb[i]
                loi = np.sum(du_doan(Xb, w) != y)
                if loi < loi_tui:         # tốt hơn túi hiện tại: bỏ vào túi
                    w_tui, loi_tui = w.copy(), loi
        lich_su.append((np.sum(du_doan(Xb, w) != y), loi_tui))
    return w, w_tui, lich_su

def chia_train_test(X, y, ti_le_test=0.3, seed=0):
    """Xáo trộn rồi chia dữ liệu thành tập huấn luyện và tập kiểm tra."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = int(len(y) * ti_le_test)
    return X[idx[n_test:]], X[idx[:n_test]], y[idx[n_test:]], y[idx[:n_test]]

def ma_tran_nham_lan(y, y_hat):
    """Ma trận nhầm lẫn C = [[TN, FP], [FN, TP]] (hàng: nhãn thực, cột: nhãn dự đoán)."""
    tp = np.sum((y == 1) & (y_hat == 1))
    tn = np.sum((y == -1) & (y_hat == -1))
    fp = np.sum((y == -1) & (y_hat == 1))
    fn = np.sum((y == 1) & (y_hat == -1))
    return np.array([[tn, fp], [fn, tp]])
