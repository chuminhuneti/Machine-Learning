import numpy as np
from logistic import *
from softmax import *
from mang import *
np.set_printoptions(precision=4, suppress=True)

print("--- 6.3: Softmax Regression tren Iris")
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
d = load_iris(); X_iris, y_iris = d.data, d.target
X_tr, X_te, y_tr, y_te = train_test_split(X_iris, y_iris, test_size=0.3, random_state=0, stratify=y_iris)
Z_tr, Z_te, mu, s = chuan_hoa(X_tr, X_te)
Zb_tr, Zb_te = them_bias(Z_tr), them_bias(Z_te)
Y_tr = np.eye(3)[y_tr]                                   # nhãn one-hot, kích thước n x K
W, ls = huan_luyen_softmax(Zb_tr, Y_tr, eta=0.5, so_epoch=2000)
print("J ban dau =", round(ls[0], 4), "= ln 3 =", round(np.log(3), 4), "| J cuoi =", round(ls[-1], 4))
print("W =")
print(W)
P_te = softmax(Zb_te @ W)
print("accuracy train =", np.mean(softmax(Zb_tr @ W).argmax(axis=1) == y_tr).round(4),
      "| accuracy test =", np.mean(P_te.argmax(axis=1) == y_te).round(4))
print(confusion_matrix(y_te, P_te.argmax(axis=1)))

print("--- 6.3: mang no-ron hoc XOR")
X4 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y4 = np.array([0, 1, 1, 0], dtype=float)
for so_an in (2, 3):
    thanh_cong = 0
    for seed in range(10):
        W1, w2, ls = mang_hai_lop(X4, y4, so_an=so_an, eta=2.0, so_epoch=10000, seed=seed)
        thanh_cong += np.all((du_doan_mang(X4, W1, w2) >= 0.5) == y4)
    print(f"{so_an} no-ron an: hoc dung XOR o {thanh_cong}/10 lan khoi tao")
W1, w2, ls = mang_hai_lop(X4, y4, so_an=3, eta=2.0, so_epoch=10000, seed=0)
print("p =", du_doan_mang(X4, W1, w2).round(3), "| J cuoi =", round(ls[-1], 4))
