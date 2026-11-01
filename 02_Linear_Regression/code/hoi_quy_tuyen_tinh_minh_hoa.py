"""
MÃ MINH HỌA CHO BÀI GIẢNG: MÔ HÌNH HỒI QUY TUYẾN TÍNH

Ký hiệu (thống nhất với bài giảng):
    Xb : ma trận đặc trưng X, kích thước n x (d+1), cột đầu toàn số 1
    y  : vector giá trị thực, độ dài n
    w  : vector tham số, độ dài d+1 (w[0] là hệ số chệch)
    e  : vector sai số = Xb @ w - y

Cách chạy:  python hoi_quy_tuyen_tinh_minh_hoa.py
Yêu cầu :  numpy, scikit-learn  (pip install numpy scikit-learn)
"""
import numpy as np
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, SGDRegressor,
                                  QuantileRegressor)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# =====================================================================
# PHẦN 1. CÁC HÀM CƠ BẢN (mục 2 và mục 3)
# =====================================================================
def them_bias(X):
    """Thêm cột 1 vào đầu ma trận đặc trưng: (n, d) -> (n, d+1)."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


def du_doan(Xb, w):
    """Giá trị dự đoán  y_hat = Xw."""
    return Xb @ w


def mse(Xb, y, w):
    """Hàm mất mát MSE:  J(w) = (1/n) ||Xw - y||^2."""
    e = Xb @ w - y
    return np.mean(e ** 2)


def gradient(Xb, y, w):
    """Gradient của MSE:  (2/n) X^T (Xw - y)."""
    n = len(y)
    return (2 / n) * Xb.T @ (Xb @ w - y)


def huan_luyen_gd(Xb, y, eta=0.02, t_max=1000, eps=1e-12):
    """Batch Gradient Descent (mục 3.4). Trả về (w, lịch sử J)."""
    w = np.zeros(Xb.shape[1])                    # khởi tạo w = 0
    lich_su = [mse(Xb, y, w)]
    for _ in range(t_max):
        w = w - eta * gradient(Xb, y, w)         # cập nhật tham số
        lich_su.append(mse(Xb, y, w))
        if abs(lich_su[-2] - lich_su[-1]) < eps:  # điều kiện dừng
            break
    return w, lich_su


def phuong_trinh_chuan(Xb, y):
    """Nghiệm đóng (mục 3.7): giải (X^T X) w = X^T y."""
    return np.linalg.solve(Xb.T @ Xb, Xb.T @ y)


def ridge(Xb, y, lam):
    """Ridge nghiệm đóng (mục 5.2): (X^T X + n*lam*P) w = X^T y, P = diag(0,1,...,1)."""
    n, d1 = Xb.shape
    P = np.eye(d1)
    P[0, 0] = 0                                  # không điều chuẩn w0
    return np.linalg.solve(Xb.T @ Xb + n * lam * P, Xb.T @ y)


def danh_gia(y, y_hat):
    """Trả về (MSE, RMSE, MAE, R2)."""
    e = y_hat - y
    mse_ = np.mean(e ** 2)
    return (mse_, np.sqrt(mse_), np.mean(np.abs(e)),
            1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2))


def tieu_de(s):
    print("\n" + "=" * 70 + "\n" + s + "\n" + "=" * 70)


# =====================================================================
# PHẦN 2. VÍ DỤ 5 CĂN NHÀ: khớp với ví dụ tính tay (mục 1, 2, 3)
# =====================================================================
def phan_2_nam_can_nha():
    tieu_de("PHẦN 2. Ví dụ 5 căn nhà (tính tay ở mục 3)")
    X = np.array([[3], [4], [5], [6], [7]], dtype=float)   # diện tích, đơn vị 10 m2
    y = np.array([1.2, 1.9, 2.4, 3.2, 3.3])                # giá, tỷ đồng
    Xb = them_bias(X)

    # Mục 2.2: dự đoán và sai số của đường A
    w_A = np.array([-0.35, 0.55])
    print("y_hat đường A :", du_doan(Xb, w_A))
    print("sai số e      :", np.round(Xb @ w_A - y, 3))
    print("J đường A     :", round(mse(Xb, y, w_A), 4))
    for ten, w in [("B", [0.2, 0.35]), ("C", [-1.0, 0.75])]:
        print("J đường %s     : %.4f" % (ten, mse(Xb, y, np.array(w))))

    # Mục 3.5: hai lần lặp đầu của Gradient Descent (eta = 0.02, w = 0)
    w1, J = huan_luyen_gd(Xb, y, eta=0.02, t_max=3, eps=0)
    print("\nGD 3 lần lặp  : w =", np.round(w1, 5), "| J qua các lần lặp =", np.round(J, 4))

    # Mục 3.7: phương trình chuẩn
    w_pt = phuong_trinh_chuan(Xb, y)
    print("PT chuẩn      : w* =", w_pt, "| J =", round(mse(Xb, y, w_pt), 4))
    e = Xb @ w_pt - y
    print("Kiểm tra X^T e = 0:", np.round(Xb.T @ e, 10))

    # Mục 3.5: GD chậm khi chưa chuẩn hóa
    w_cham, J_cham = huan_luyen_gd(Xb, y, eta=0.02, t_max=1000, eps=0)
    print("GD 1000 lần   : w =", np.round(w_cham, 4), "(vẫn chưa tới w*)")

    # Mục 3.5: chuẩn hóa đặc trưng giúp hội tụ nhanh
    tb, sd = X.mean(axis=0), X.std(axis=0)
    Xs = them_bias((X - tb) / sd)
    w_s, J_s = huan_luyen_gd(Xs, y, eta=0.3, t_max=1000)
    w_goc = np.array([w_s[0] - w_s[1] * tb[0] / sd[0], w_s[1] / sd[0]])
    print("Sau chuẩn hóa : w' =", np.round(w_s, 4), "| số lần lặp =", len(J_s) - 1)
    print("Quy về thang gốc: w =", np.round(w_goc, 4))

    # Mục 3.6: ảnh hưởng của tốc độ học
    print("\nTốc độ học (đặc trưng chưa chuẩn hóa), J sau 40 lần lặp:")
    for eta in [0.001, 0.02, 0.04]:
        _, J_ = huan_luyen_gd(Xb, y, eta=eta, t_max=40, eps=0)
        print("  eta = %-6s J = %.4g" % (eta, J_[-1]))
    L = np.linalg.eigvalsh(2 / len(y) * Xb.T @ Xb).max()
    print("Ngưỡng ổn định 2/L = %.4f" % (2 / L))


# =====================================================================
# PHẦN 3. MAE VÀ NGOẠI LỆ (mục 5.1)
# =====================================================================
def phan_3_ngoai_le():
    tieu_de("PHẦN 3. MSE và MAE khi có ngoại lệ (mục 5.1)")
    x = np.array([3, 4, 5, 6, 7, 8], dtype=float)
    y = np.array([1.2, 1.9, 2.4, 3.2, 3.3, 6.0])    # căn thứ 6 là ngoại lệ
    Xb = them_bias(x[:, None])
    w_mse = phuong_trinh_chuan(Xb, y)
    # MAE (hồi quy phân vị 0,5, alpha = 0): cực tiểu hóa tổng |e|
    w_mae = QuantileRegressor(quantile=0.5, alpha=0, fit_intercept=False,
                              solver="highs").fit(Xb, y).coef_
    Xb5, y5 = Xb[:5], y[:5]
    print("Không ngoại lệ, MSE : w = [-0.35, 0.55]")
    for ten, w in [("MSE có ngoại lệ", w_mse), ("MAE có ngoại lệ", w_mae)]:
        mae5 = np.mean(np.abs(Xb5 @ w - y5))
        print("%-16s: w = %s | MAE trên 5 nhà bình thường = %.4f" % (ten, np.round(w, 3), mae5))


# =====================================================================
# PHẦN 4. BIẾN ĐỔI ĐẶC TRƯNG, QUÁ KHỚP, RIDGE, LASSO (mục 4 và 5)
# =====================================================================
def phan_4_da_thuc():
    tieu_de("PHẦN 4. Đa thức, quá khớp, Ridge và Lasso (mục 4.3 và 5.2)")
    rng = np.random.default_rng(7)
    f = lambda t: 0.5 + 0.03 * t ** 2               # quan hệ thật (đường cong)
    x_tr = np.linspace(3, 12, 10)
    y_tr = f(x_tr) + rng.normal(0, 0.25, 10)
    x_te = np.linspace(3, 12, 200)
    y_te = f(x_te) + rng.normal(0, 0.25, 200)

    def mo_hinh(bac, reg):
        return make_pipeline(PolynomialFeatures(bac, include_bias=False), StandardScaler(), reg)

    n = len(x_tr)
    cau_hinh = [
        ("Bậc 1", 1, LinearRegression()),
        ("Bậc 2", 2, LinearRegression()),
        ("Bậc 8", 8, LinearRegression()),
        ("Bậc 8 + Ridge (lambda=0,01)", 8, Ridge(alpha=n * 0.01)),   # alpha = n*lambda
        ("Bậc 8 + Lasso (lambda=0,04)", 8, Lasso(alpha=0.04 / 2, max_iter=200000)),  # alpha = lambda/2
    ]
    print("%-30s %-12s %-12s %-14s %s" % ("Mô hình", "MSE huấn luyện", "MSE kiểm tra",
                                           "||w||", "Số w khác 0"))
    for ten, bac, reg in cau_hinh:
        m = mo_hinh(bac, reg).fit(x_tr[:, None], y_tr)
        w = m[-1].coef_
        print("%-30s %-14.4f %-12.4f %-14.1f %d" % (
            ten, mean_squared_error(y_tr, m.predict(x_tr[:, None])),
            mean_squared_error(y_te, m.predict(x_te[:, None])),
            np.linalg.norm(w), np.sum(np.abs(w) > 1e-6)))


# =====================================================================
# PHẦN 5. DỮ LIỆU 300 CĂN NHÀ: huấn luyện, đánh giá, Ridge, scikit-learn (mục 6)
# =====================================================================
def phan_5_du_lieu_lon():
    tieu_de("PHẦN 5. 300 căn nhà: chia dữ liệu, đánh giá, scikit-learn (mục 6)")
    rng = np.random.default_rng(42)
    n = 300
    dien_tich = rng.uniform(30, 120, n)             # x1: m2
    khoang_cach = rng.uniform(0.5, 15, n)           # x2: km tới trung tâm
    gia = 0.8 + 0.045 * dien_tich - 0.08 * khoang_cach + rng.normal(0, 0.4, n)
    X = np.column_stack([dien_tich, khoang_cach])

    idx = rng.permutation(n)
    n_tr = int(0.8 * n)
    tr, te = idx[:n_tr], idx[n_tr:]
    X_tr, y_tr, X_te, y_te = X[tr], gia[tr], X[te], gia[te]
    print("Số mẫu huấn luyện: %d | kiểm tra: %d" % (len(tr), len(te)))

    # Chuẩn hóa: tham số CHỈ ước lượng trên tập huấn luyện (tránh rò rỉ)
    tb, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
    Xs_tr = them_bias((X_tr - tb) / sd)
    Xs_te = them_bias((X_te - tb) / sd)

    w_gd, J = huan_luyen_gd(Xs_tr, y_tr, eta=0.1, t_max=5000, eps=1e-12)
    w_pt = phuong_trinh_chuan(Xs_tr, y_tr)
    print("GD: %d lần lặp | w = %s" % (len(J) - 1, np.round(w_gd, 4)))
    print("PT chuẩn       | w = %s" % np.round(w_pt, 4))
    print("Sai khác tối đa giữa hai nghiệm: %.2e" % np.max(np.abs(w_gd - w_pt)))

    w1_g = w_pt[1:] / sd
    w0_g = w_pt[0] - np.sum(w_pt[1:] * tb / sd)
    print("Hệ số trên thang đo gốc: w0 = %.4f, w1 = %.4f, w2 = %.4f  (thật: 0.8, 0.045, -0.08)"
          % (w0_g, *w1_g))

    print("\nĐánh giá (tự cài đặt):")
    for ten, Xm, ym in [("Huấn luyện", Xs_tr, y_tr), ("Kiểm tra  ", Xs_te, y_te)]:
        m, r, a, r2 = danh_gia(ym, du_doan(Xm, w_pt))
        print("  %s: MSE = %.4f | RMSE = %.4f | MAE = %.4f | R2 = %.4f" % (ten, m, r, a, r2))
    base = danh_gia(y_te, np.full(len(y_te), y_tr.mean()))[3]
    print("  Mô hình 'đoán trung bình': R2 trên tập kiểm tra = %.4f" % base)

    print("\nRidge cài từ đầu:")
    for lam in [0, 0.1, 1, 10]:
        w_r = ridge(Xs_tr, y_tr, lam)
        print("  lambda = %-4s w = %s | MSE kiểm tra = %.4f"
              % (lam, np.round(w_r, 4), danh_gia(y_te, du_doan(Xs_te, w_r))[0]))

    # scikit-learn: lưu ý quy ước tham số alpha
    lam = 0.1
    mo_hinh = {
        "LinearRegression": LinearRegression(),
        "Ridge (lambda=0,1)": Ridge(alpha=n_tr * lam),            # alpha = n * lambda
        "Lasso (lambda=0,1)": Lasso(alpha=lam / 2),               # alpha = lambda / 2
        "SGDRegressor": SGDRegressor(loss="squared_error", penalty=None,
                                     learning_rate="constant", eta0=0.01,
                                     max_iter=2000, tol=1e-6, random_state=0),
    }
    print("\nscikit-learn:")
    print("  %-20s %-9s %-9s %-7s" % ("Mô hình", "MSE", "MAE", "R2"))
    for ten, m in mo_hinh.items():
        pipe = make_pipeline(StandardScaler(), m).fit(X_tr, y_tr)
        y_hat = pipe.predict(X_te)
        print("  %-20s %-9.4f %-9.4f %-7.4f" % (ten, mean_squared_error(y_te, y_hat),
              mean_absolute_error(y_te, y_hat), r2_score(y_te, y_hat)))


if __name__ == "__main__":
    phan_2_nam_can_nha()
    phan_3_ngoai_le()
    phan_4_da_thuc()
    phan_5_du_lieu_lon()
