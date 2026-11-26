"""
Online Resource 1 - Reproducible analysis script
Machine learning for burnout detection in software professionals:
a Monte Carlo simulation study.

Input : df_software.csv (6,000 synthetic cases generated from the covariance
        structure of 2,500 real questionnaire responses; see Method).
Output: all statistics reported in the article and Figs. 2-4 / S1-S3.
Requires: pandas, numpy, scipy, scikit-learn, statsmodels, matplotlib.
"""
import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (cross_val_score, cross_validate, train_test_split,
                                     StratifiedKFold, KFold)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, classification_report,
                             adjusted_rand_score, silhouette_score)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.contingency_tables import mcnemar

SEED = 42
path = sys.argv[1] if len(sys.argv) > 1 else "simulated_data.csv"
figdir = sys.argv[2] if len(sys.argv) > 2 else "."
df = pd.read_csv(path)
F = ["Long Working Hours", "Role Misalignment", "Work Overload", "Workplace Harassment",
     "Employee Turnover", "G_Anxiety_Disorder", "Depression"]
LAB = ["Long working hours", "Role misalignment", "Work overload", "Workplace harassment",
       "Employee turnover", "Generalized anxiety", "Depression"]
X, y = df[F], df["Burnout"]
res = {}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})

# ---- Fig. 2: distribution and correlation matrix ----
cuts = y.quantile([1/3, 2/3]).values
fig, ax = plt.subplots(2, 1, figsize=(6.5, 10), gridspec_kw={"height_ratios": [1, 1.6]})
ax[0].hist(y, bins=np.arange(24, 93, 2), color="#4878A8", edgecolor="white")
for c in cuts: ax[0].axvline(c, ls="--", color="black", lw=1)
ax[0].set_xlabel("Burnout score"); ax[0].set_ylabel("Simulated cases")
ax[0].text(-0.1, 1.02, "a", transform=ax[0].transAxes, fontsize=13, fontweight="bold")
corr = df[["Burnout"] + F].corr().values
im = ax[1].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
names = ["Burnout"] + LAB
ax[1].set_xticks(range(8)); ax[1].set_xticklabels(names, rotation=40, ha="right")
ax[1].set_yticks(range(8)); ax[1].set_yticklabels(names)
for i in range(8):
    for j in range(8):
        ax[1].text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                   color="white" if abs(corr[i, j]) > .6 else "black")
ax[1].spines[:].set_visible(False)
fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04, label="Pearson r")
ax[1].text(-0.42, 1.02, "b", transform=ax[1].transAxes, fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(f"{figdir}/Fig2.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"}); fig.savefig(f"{figdir}/Fig2.png", dpi=300); plt.close(fig)

# ---- associations and redundancy ----
n = len(df)
def fisher_ci(r):
    z, se = np.arctanh(r), 1/np.sqrt(n-3); return np.tanh(z-1.96*se), np.tanh(z+1.96*se)
res["r_with_burnout"] = {f: [round(df[f].corr(y), 2), *[round(v, 2) for v in fisher_ci(df[f].corr(y))]] for f in F}
def r2(cols): return LinearRegression().fit(df[cols], y).score(df[cols], y)
res["R2_single"] = {f: round(r2([f]), 3) for f in F}
res["R2_RM_WO"] = round(r2(["Role Misalignment", "Work Overload"]), 3)
res["R2_all7"] = round(r2(F), 4)
Z = StandardScaler().fit_transform(X)
res["r_sum_z"] = round(np.corrcoef(Z.sum(1), y)[0, 1], 3)
pcs = PCA().fit(Z); P = pcs.transform(Z)
res["PC_var"] = [round(v, 3) for v in pcs.explained_variance_ratio_[:2]]
res["r_PC1_PC2"] = [round(abs(np.corrcoef(P[:, 0], y)[0, 1]), 3), round(np.corrcoef(P[:, 1], y)[0, 1], 3)]

# ---- Fig. 3: logistic regression ----
yb = (y > 50).astype(int)
lr = LogisticRegression().fit(df[["Long Working Hours"]], yb)
b0, b1 = lr.intercept_[0], lr.coef_[0][0]
res["logit"] = {"b0": round(b0, 3), "b1": round(b1, 3), "x_p50": round(-b0/b1, 2)}
xs = np.linspace(0, 10.6, 300); ps = 1/(1+np.exp(-(b0+b1*xs)))
rng = np.random.default_rng(SEED); jit = rng.uniform(-.03, .03, n)
fig, ax = plt.subplots(figsize=(6.5, 3.4))
lw = df["Long Working Hours"]
ax.scatter(lw[yb == 0], yb[yb == 0]+jit[yb == 0], s=3, alpha=.35, color="#4878A8", label="No burnout (score ≤ 50)")
ax.scatter(lw[yb == 1], yb[yb == 1]+jit[yb == 1], s=3, alpha=.35, color="#8B1A3A", label="Burnout (score > 50)")
ax.plot(xs, ps, color="black", lw=2, label="Estimated probability")
ax.axhline(.5, ls="--", color="gray", lw=1, label="Threshold 0.5"); ax.axvline(-b0/b1, ls=":", color="gray", lw=1)
ax.set_xlabel("Long working hours score"); ax.set_ylabel("Probability of burnout")
ax.legend(frameon=False, loc="center right", fontsize=8, markerscale=3)
fig.tight_layout(); fig.savefig(f"{figdir}/Fig3.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"}); fig.savefig(f"{figdir}/Fig3.png", dpi=300); plt.close(fig)

# ---- polynomial regression (5-fold CV) ----
kf = KFold(5, shuffle=True, random_state=SEED); res["poly_cv"] = {}
for d in [1, 2, 3, 5]:
    s = cross_validate(make_pipeline(PolynomialFeatures(d), LinearRegression()), df[["Work Overload"]], y, cv=kf,
                       scoring=("r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"))
    res["poly_cv"][d] = [round(s["test_r2"].mean(), 3), round(-s["test_neg_mean_absolute_error"].mean(), 2),
                         round(-s["test_neg_root_mean_squared_error"].mean(), 2)]

# ---- classification ----
labs = ["Low", "Medium", "High"]
lev = pd.qcut(y, 3, labels=labs)
res["tertile_cuts"] = [round(c, 1) for c in cuts]
models = {"K-NN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
          "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1, gamma="scale"))}
skf = StratifiedKFold(5, shuffle=True, random_state=SEED); cv = {}
for k, m in models.items():
    s = cross_val_score(m, X, lev, cv=skf); cv[k] = s
    res[f"cv_{k}"] = [round(s.mean(), 3), round(s.std(), 3), round(s.min(), 3), round(s.max(), 3)]
Xtr, Xte, ytr, yte = train_test_split(X, lev, test_size=.2, stratify=lev, random_state=SEED)
preds, cms = {}, {}
def summarize(name, pr):
    cm = confusion_matrix(yte, pr, labels=labs); k = int(np.trace(cm))
    lo, hi = proportion_confint(k, len(yte), method="wilson")
    rep = classification_report(yte, pr, labels=labs, output_dict=True)
    res[f"test_{name}"] = {"correct": k, "acc": round(k/len(yte), 4), "ci95": [round(lo, 3), round(hi, 3)],
                           "macroF1": round(rep["macro avg"]["f1-score"], 3), "cm": cm.tolist(),
                           "levels": {l: [round(rep[l][m], 2) for m in ("precision", "recall", "f1-score")] for l in labs}}
    cms[name] = cm
for k, m in models.items():
    preds[k] = m.fit(Xtr, ytr).predict(Xte); summarize(k, preds[k])
a = int(((preds["SVM"] == yte.values) & (preds["K-NN"] != yte.values)).sum())
b = int(((preds["SVM"] != yte.values) & (preds["K-NN"] == yte.values)).sum())
res["mcnemar"] = {"svm_only": a, "knn_only": b, "p": float(mcnemar([[0, a], [b, 0]], exact=True).pvalue)}
sc = StandardScaler().fit(Xtr); pca2 = PCA(2).fit(sc.transform(Xtr))
svm2 = SVC(kernel="rbf", C=1, gamma="scale").fit(pca2.transform(sc.transform(Xtr)), ytr)
summarize("SVM_2PC", svm2.predict(pca2.transform(sc.transform(Xte))))

# ---- Fig. 4: confusion matrices + PC plane ----
fig, axs = plt.subplots(2, 2, figsize=(7, 6.4))
for ax, (nm, cmap, letter) in zip(axs.flat[:3], [("SVM", "Purples", "a"), ("K-NN", "Blues", "b"), ("SVM_2PC", "Purples", "c")]):
    cm = cms[nm]; ax.imshow(cm, cmap=cmap, vmin=0, vmax=400)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > 200 else "black", fontsize=10)
    ax.set_xticks(range(3)); ax.set_xticklabels(labs); ax.set_yticks(range(3)); ax.set_yticklabels(labs)
    ax.set_xlabel("Predicted level"); ax.set_ylabel("Observed level"); ax.spines[:].set_visible(False)
    ax.text(-0.35, 1.04, letter, transform=ax.transAxes, fontsize=13, fontweight="bold")
ax = axs[1, 1]; colors = {"Low": "#3B75AF", "Medium": "#E3A21A", "High": "#8B1A3A"}
for l in labs:
    msk = (lev == l).values; ax.scatter(P[msk, 0], P[msk, 1], s=2, alpha=.4, color=colors[l], label=l)
ax.set_xlabel("First principal component"); ax.set_ylabel("Second principal component")
ax.legend(title="Burnout level", frameon=False, markerscale=4, fontsize=8)
ax.text(-0.25, 1.04, "d", transform=ax.transAxes, fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(f"{figdir}/Fig4.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"}); fig.savefig(f"{figdir}/Fig4.png", dpi=300); plt.close(fig)

# ---- Fig. S1: cross-validated accuracy ----
fig, axs = plt.subplots(1, 2, figsize=(7, 3.2))
axs[0].bar(["K-NN", "SVM"], [cv["K-NN"].mean(), cv["SVM"].mean()], yerr=[cv["K-NN"].std(), cv["SVM"].std()],
           capsize=6, color=["#8C8C8C", "#3B75AF"], edgecolor="black")
axs[0].axhline(1/3, ls=":", color="black"); axs[0].set_ylim(0, 1); axs[0].set_ylabel("Mean accuracy (5-fold CV)")
axs[1].boxplot([cv["K-NN"], cv["SVM"]], tick_labels=["K-NN", "SVM"])
for i, k in enumerate(["K-NN", "SVM"], 1): axs[1].scatter([i]*5, cv[k], color="black", s=12, zorder=3)
axs[1].set_ylabel("Accuracy per fold")
for ax, l in zip(axs, "ab"): ax.text(-0.2, 1.03, l, transform=ax.transAxes, fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(f"{figdir}/FigS1.png", dpi=300); plt.close(fig)

# ---- clustering, Figs. S2-S3 ----
inert, sil = [], []
for k in range(2, 7):
    km = KMeans(k, random_state=SEED, n_init=10).fit(Z); inert.append(km.inertia_); sil.append(silhouette_score(Z, km.labels_))
res["inertia"] = [round(v) for v in inert]; res["silhouette"] = [round(v, 3) for v in sil]
km3 = KMeans(3, random_state=SEED, n_init=10).fit(Z)
res["crosstab_k3"] = pd.crosstab(km3.labels_, lev).values.tolist(); res["ARI"] = round(adjusted_rand_score(lev, km3.labels_), 3)
fig, axs = plt.subplots(1, 2, figsize=(7, 3.0))
axs[0].plot(range(2, 7), inert, marker="o", color="#3B75AF"); axs[0].set_xlabel("Number of clusters (k)"); axs[0].set_ylabel("Within-cluster inertia")
axs[1].plot(range(2, 7), sil, marker="o", color="#3B75AF"); axs[1].axhline(.25, ls="--", color="gray")
axs[1].set_xlabel("Number of clusters (k)"); axs[1].set_ylabel("Mean silhouette")
for ax, l in zip(axs, "ab"): ax.text(-0.25, 1.03, l, transform=ax.transAxes, fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(f"{figdir}/FigS2.png", dpi=300); plt.close(fig)
fig, ax = plt.subplots(figsize=(5, 3.8)); cc = ["#1B9E77", "#E7298A", "#8C8C8C"]
for k in range(3):
    msk = km3.labels_ == k; ax.scatter(P[msk, 0], P[msk, 1], s=2, alpha=.4, color=cc[k], label=f"Cluster {k+1}")
cen = pcs.transform(km3.cluster_centers_); ax.scatter(cen[:, 0], cen[:, 1], marker="X", s=120, color="black", label="Centroids")
ax.set_xlabel("First principal component"); ax.set_ylabel("Second principal component")
lg = ax.legend(frameon=False, fontsize=8)
for h in lg.legend_handles[:3]: h.set_sizes([25])
lg.legend_handles[3].set_sizes([60])
fig.tight_layout(); fig.savefig(f"{figdir}/FigS3.png", dpi=300); plt.close(fig)

json.dump(res, open(f"{figdir}/results.json", "w"), indent=1)
print(json.dumps(res, indent=1))

# ---- neural network regression (MLP) ----
from sklearn.neural_network import MLPRegressor
sets = {"LWH": ["Long Working Hours"], "RM": ["Role Misalignment"], "WO": ["Work Overload"],
        "RM+WO": ["Role Misalignment", "Work Overload"], "all7": F}
res["mlp"] = {}
for k, cols in sets.items():
    mk = lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu",
                               solver="adam", max_iter=2000, early_stopping=False, random_state=SEED))
    full = mk().fit(df[cols], y).score(df[cols], y)
    cvs = cross_val_score(mk(), df[cols], y, cv=KFold(5, shuffle=True, random_state=SEED), scoring="r2").mean()
    res["mlp"][k] = [round(full, 3), round(cvs, 3)]
print("MLP", res["mlp"])
json.dump(res, open(f"{figdir}/results.json", "w"), indent=1)


# ---------------------------------------------------------------------------
# Monte Carlo simulation procedure (documentation).
# This function generated df_software.csv from the 2,500 real responses.
# It is NOT run here because the real responses are not public.
# ---------------------------------------------------------------------------
def simulate(real: pd.DataFrame, n_sim: int = 6000, seed: int = 42) -> pd.DataFrame:
    """Parametric Monte Carlo simulation from the real-sample covariance structure."""
    num = real.select_dtypes(include=[np.number]).columns          # 22 items, 7 factors, burnout
    np.random.seed(seed)
    sim = pd.DataFrame(np.random.multivariate_normal(real[num].mean().values,
                                                     real[num].cov().values, size=n_sim), columns=num)
    for c in [c for c in num if c.startswith("Item")]:             # items: round and truncate to observed range
        sim[c] = np.clip(np.round(sim[c]), real[c].min(), real[c].max()).astype(int)
    for c in F:                                                     # factor scores: truncate negatives at 0
        sim[c] = np.clip(sim[c], 0, None)
    # Note: in the dataset provided as Online Resource 3, the Age and Country labels were drawn
    # from a preliminary version of the data and do not reproduce the real-sample proportions
    # (Mexican 50%, Indian 25%, US 25%). They are not used in any analysis.
    for c in ["Age", "Country"]:                                    # assigned at random, independent of scores
        p = real[c].value_counts(normalize=True)
        sim[c] = np.random.choice(p.index, size=n_sim, p=p.values)
    return sim[real.columns]
