import numpy as np
from logistic import *
np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(42); n_lop = 200
dau = np.c_[rng.normal(8, 2.2, n_lop), rng.normal(1.5, 1.2, n_lop)]
rot = np.c_[rng.normal(4, 2.2, n_lop), rng.normal(3.5, 1.5, n_lop)]
X = np.clip(np.vstack([dau, rot]), 0, None); y = np.r_[np.ones(n_lop), np.zeros(n_lop)].astype(int)
X_tr, X_te, y_tr, y_te = chia_train_test(X, y, ti_le_test=0.3, seed=0)
Z_tr, Z_te, mu, s = chuan_hoa(X_tr, X_te)
Zb_tr, Zb_te = them_bias(Z_tr), them_bias(Z_te)
w_z, _ = huan_luyen(Zb_tr, y_tr, eta=5.0, so_epoch=300)
w_goc = np.r_[w_z[0] - np.sum(w_z[1:] * mu / s), w_z[1:] / s]
print("--- 5.2/5.3: mo hinh tu cai dat")
p_tr, p_te = sigmoid(Zb_tr @ w_z), sigmoid(Zb_te @ w_z)
print("J train =", mat_mat(Zb_tr, y_tr, w_z).round(4), "| J test =", mat_mat(Zb_te, y_te, w_z).round(4))
print("acc train =", np.mean((p_tr >= 0.5) == y_tr).round(4), "| acc test =", np.mean((p_te >= 0.5) == y_te).round(4))
def ma_tran_nham_lan(y, y_hat):
    tp = np.sum((y == 1) & (y_hat == 1)); tn = np.sum((y == 0) & (y_hat == 0))
    fp = np.sum((y == 0) & (y_hat == 1)); fn = np.sum((y == 1) & (y_hat == 0))
    return np.array([[tn, fp], [fn, tp]])
C = ma_tran_nham_lan(y_te, (p_te >= 0.5).astype(int)); print(C)
tn, fp, fn, tp = C.ravel()
print("precision =", round(tp / (tp + fp), 3), "| recall =", round(tp / (tp + fn), 3), "| do dac hieu =", round(tn / (tn + fp), 3), "| F1 =", round(2 * tp / (2 * tp + fp + fn), 3))
print("--- 5.5: scikit-learn")
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=1e10, max_iter=10000).fit(X_tr, y_tr)
print("intercept_ =", clf.intercept_, "| coef_ =", clf.coef_, "| classes_ =", clf.classes_)
print("tu cai dat (thang goc): w =", w_goc)
p_sk = clf.predict_proba(X_te)[:, 1]
p_ours = sigmoid(them_bias(X_te) @ w_goc)
print("max |p_sk - p_ours| =", np.abs(p_sk - p_ours).max().round(4))
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, confusion_matrix, classification_report
clf_mac_dinh = LogisticRegression().fit(X_tr, y_tr)
print(f"{'tu cai dat':24s} train {np.mean((sigmoid(them_bias(X_tr) @ w_goc) >= 0.5) == y_tr):.4f} | test {np.mean((p_ours >= 0.5) == y_te):.4f} | log-loss {log_loss(y_te, p_ours):.4f} | AUC {roc_auc_score(y_te, p_ours):.4f}")
for ten, m in [("sklearn C=1e10", clf), ("sklearn mac dinh (C=1)", clf_mac_dinh)]:
    p = m.predict_proba(X_te)[:, 1]
    print(f"{ten:24s} train {m.score(X_tr, y_tr):.4f} | test {m.score(X_te, y_te):.4f} | log-loss {log_loss(y_te, p):.4f} | AUC {roc_auc_score(y_te, p):.4f}")
print("mac dinh: intercept_ =", clf_mac_dinh.intercept_, "| coef_ =", clf_mac_dinh.coef_)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for Cc in (0.001, 0.01, 0.1, 1, 10, 100, 1000):
    mo_hinh = make_pipeline(StandardScaler(), LogisticRegression(C=Cc))
    acc = cross_val_score(mo_hinh, X_tr, y_tr, cv=cv)
    ll = -cross_val_score(mo_hinh, X_tr, y_tr, cv=cv, scoring="neg_log_loss")
    print(f"C = {Cc:<6} CV accuracy = {acc.mean():.4f} +- {acc.std():.4f} | CV log-loss = {ll.mean():.4f}")
tot = make_pipeline(StandardScaler(), LogisticRegression(C=10)).fit(X_tr, y_tr)
print("C = 10 tren tap kiem tra: accuracy =", round(tot.score(X_te, y_te), 4), "| log-loss =", round(log_loss(y_te, tot.predict_proba(X_te)[:, 1]), 4))
print(confusion_matrix(y_te, clf.predict(X_te), labels=[0, 1]))
print(classification_report(y_te, (p_ours >= 0.5).astype(int), labels=[0, 1], target_names=["rot", "dau"], digits=3))
