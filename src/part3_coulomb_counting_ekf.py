#============================================================
# PART 3 (v2) - COULOMB COUNTING vs EXTENDED KALMAN FILTER
# Upload: model_parameters.json (from Part 2 v3) + DST 80%/50% + FUDS 80%/50% files
# Tests: T1 nominal | T2 +/-10% initial SOC | T3 current bias+noise | T4 voltage noise | T5 DST profile (combined errors)
# ============================================================
import re, json, glob
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from time import perf_counter
try:
    from google.colab import files; IN_COLAB = True
except ImportError:
    IN_COLAB = False
if IN_COLAB: print(list(files.upload()))
 
# ---------- 1. find files ----------
fs = glob.glob("*.json") + glob.glob("*.xls") + glob.glob("*.xlsx")
norm = lambda f: f.lower().replace("_", "").replace(" ", "").replace("-", "")
def pick(*words, bad=()):
    h = sorted([f for f in fs if all(w in norm(f) for w in words) and not any(b in norm(f) for b in bad)],
               key=lambda x: ("(" in x, len(x)))
    return h[0] if h else None
MODEL = pick("modelparameters")
FILES = {(p, l): pick(p, f"{l}soc", bad=("bjdst",)) for p in ("dst", "fuds") for l in ("80", "50")}
print("Model:", MODEL); print("Data :", FILES)
if MODEL is None or None in FILES.values():
    raise ValueError("Need model_parameters.json plus DST 80/50 and FUDS 80/50 files - see the printout above.")
 
# ---------- 2. model from Part 2 ----------
P = json.load(open(MODEL))
R0, R1, TAU, DV, Q = (float(P[k]) for k in ("R0_ohm", "R1_ohm", "tau_s", "dV_volt", "Q_ah"))
POLY = np.array(P["ocv_poly"], float); DPOLY = np.polyder(POLY)
G = np.linspace(0.05, 1.0, 1901); OCVG = np.polyval(POLY, G); DOCVG = np.polyval(DPOLY, G)   # look-up tables (fast)
lut = lambda tab, s: tab[min(max(int((s - 0.05) / 0.0005 + 0.5), 0), 1900)]
SOC_MIN = 0.15      # the model was fitted and validated above 15 % SOC; the filter is only run/evaluated there
print(f"Model: Q={Q:.3f} Ah, R0={R0*1e3:.1f} mOhm, R1={R1*1e3:.1f} mOhm, tau={TAU:.0f} s, dV={DV*1e3:.1f} mV")
 
# ---------- 3. data: same reference SOC / segmentation as Part 2 v3 ----------
def load(path):
    sh = pd.read_excel(path, sheet_name=None); raw = sh[max(sh, key=lambda k: len(sh[k]))]
    col = lambda pats: next(c for p in pats for c in raw.columns if re.search(p, str(c).lower()))
    t, v, i = (pd.to_numeric(raw[col(p)], errors="coerce").to_numpy(float) for p in ([r"test.?time", r"time"], [r"volt"], [r"curr"]))
    ok = ~(np.isnan(t) | np.isnan(v) | np.isnan(i)); t, v, i = t[ok], v[ok], i[ok]
    t, idx = np.unique(t, return_index=True); tu = np.arange(t[0], t[-1], 1.0)
    return pd.DataFrame({"t": tu - tu[0], "V": np.interp(tu, t, v[idx]), "I": -np.interp(tu, t, i[idx])})   # I > 0 = discharge
 
def segment(path):
    d = load(path); I, V = d.I.to_numpy(), d.V.to_numpy()
    f = np.where(V >= 4.18)[0][0]; k = f + np.argmax(I[f:] > 0.5); anchor = k - 1          # SOC = 100 % just before the set-up discharge
    pe = k + np.argmax(I[k:] < 0.5); a2 = pe + 120; j = a2 + np.argmax(np.abs(I[a2:]) > 0.05)  # dynamic test starts after the rest
    s = max(pe, j - 120)
    soc = 1 - np.r_[0, np.cumsum(I[anchor:-1])] / 3600 / Q
    seg = d.iloc[s:].reset_index(drop=True); seg["SOC"] = soc[s - anchor:]; seg["t"] -= seg.t[0]
    low = np.where(seg.SOC.to_numpy() <= SOC_MIN)[0]
    return seg.iloc[:low[0]] if len(low) else seg
 
SEG = {key: segment(path) for key, path in FILES.items()}
for key, s in SEG.items(): print(f"{key[0].upper()} {key[1]}%: {len(s)} s, SOC {s.SOC.iloc[0]*100:.1f} -> {s.SOC.iloc[-1]*100:.1f} %")
 
# ---------- 4. estimators ----------
def cc(I, s0):                                    # Coulomb counting
    return s0 - np.r_[0, np.cumsum(I[:-1])] / 3600 / Q
 
def ekf(I, V, s0, qs, qv, sv, p0s=0.05, p0v=0.005):
    """x = [SOC, V1]; same time indexing as the Part 2 model: SOC[k] uses I[k-1], V1[k] and Vt[k] use I[k].
    V1 starts at 0 with a small uncertainty because every test starts after a long rest."""
    n = len(I); a = np.exp(-1.0 / TAU); b = R1 * (1 - a); cs = 1 / (3600 * Q); R = sv * sv
    s, v1, Pss, Psv, Pvv = s0, 0.0, p0s**2, 0.0, p0v**2
    out = np.empty(n); out[0] = s0
    for k in range(1, n):
        s = min(max(s - I[k-1] * cs, 0.0), 1.0); v1 = a * v1 + b * I[k]            # predict
        Pss += qs; Psv *= a; Pvv = a * a * Pvv + qv
        h = lut(DOCVG, s); y = V[k] - (lut(OCVG, s) + DV - R0 * I[k] - v1)            # innovation, H = [h, -1]
        g1, g2 = h * Pss - Psv, h * Psv - Pvv; S = h * g1 - g2 + R
        k1, k2 = g1 / S, g2 / S
        s += k1 * y; v1 += k2 * y                                                      # correct
        Pss = max(Pss - k1 * g1, 1e-12); Psv -= k1 * g2; Pvv = max(Pvv - k2 * g2, 1e-12)
        out[k] = s = min(max(s, 0.0), 1.0)
    return out
 
def stats(ref, est, t):
    e = (est - ref) * 100
    def conv(band, hold=60):
        ok = np.convolve(np.abs(e) <= band, np.ones(hold), "valid") == hold
        return t[np.argmax(ok)] if ok.any() else np.nan
    return dict(RMSE=np.sqrt(np.mean(e**2)), MAE=np.mean(np.abs(e)), Max=np.max(np.abs(e)), Final=e[-1], Conv2=conv(2), Conv5=conv(5))
 
# ---------- 5. tune Q and R on the DST data (not on the FUDS test data) ----------
print("\nTuning EKF noise settings on DST ...")
best = None
for qs in (1e-8, 1e-7, 1e-6, 1e-5):
    for qv in (1e-8, 1e-6):
        for sv in (0.01, 0.02, 0.04):
            r = np.mean([stats(sg.SOC.to_numpy(), ekf(sg.I.to_numpy() + bias, sg.V.to_numpy(), np.clip(sg.SOC.iloc[0] + off, 0, 1), qs, qv, sv),
                               sg.t.to_numpy())["RMSE"] for key, sg in SEG.items() if key[0] == "dst"
                         for off, bias in ((0, 0), (0.1, 0), (-0.1, 0), (0, 0.05))])     # nominal, +/-10 % start, 50 mA current bias
            if best is None or r < best[0]: best = (r, qs, qv, sv)
_, QS, QV, SV = best
print(f"Chosen: q_soc={QS:g}, q_v1={QV:g}, sigma_v={SV*1e3:.0f} mV  (mean DST RMSE {best[0]:.2f} % SOC)")
 
# ---------- 6. experiments ----------
rng = np.random.default_rng(1)
#            (SOC offset, current bias A, current noise A, voltage noise V)
TESTS = {"T1 Nominal": (0, 0, 0, 0), "T2a +10% initial SOC": (0.10, 0, 0, 0), "T2b -10% initial SOC": (-0.10, 0, 0, 0),
         "T3 Current bias 50 mA + noise 20 mA": (0, 0.05, 0.02, 0), "T4 Voltage noise 20 mV": (0, 0, 0, 0.02)}
RUNS = [(t, "fuds", l) for t in TESTS for l in ("80", "50")] + [("T5 DST: +10% SOC, 50 mA bias, 10 mV noise", "dst", l) for l in ("80", "50")]
TESTS["T5 DST: +10% SOC, 50 mA bias, 10 mV noise"] = (0.10, 0.05, 0.02, 0.01)
 
rows, PL = [], {}
for test, prof, lvl in RUNS:
    off, bias, ni, nv = TESTS[test]; sg = SEG[(prof, lvl)]
    ref, t = sg.SOC.to_numpy(), sg.t.to_numpy()
    Im = sg.I.to_numpy() + bias + rng.normal(0, ni, len(sg)) if (bias or ni) else sg.I.to_numpy()
    Vm = sg.V.to_numpy() + rng.normal(0, nv, len(sg)) if nv else sg.V.to_numpy()
    s0 = float(np.clip(ref[0] + off, 0, 1))
    c = cc(Im, s0); t0 = perf_counter(); e = ekf(Im, Vm, s0, QS, QV, SV); us = (perf_counter() - t0) / len(sg) * 1e6
    PL[(test, prof, lvl)] = (t, ref, c, e)
    for name, est in (("Coulomb counting", c), ("EKF", e)):
        rows.append(dict(Test=test, Profile=f"{prof.upper()} {lvl}%", Estimator=name, **stats(ref, est, t), us_per_step=us if name == "EKF" else np.nan))
RES = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 45)
print("\n" + "=" * 100 + "\nPART 3 RESULTS  (SOC error in %, evaluated for SOC above 15 %; Conv = seconds until the error stays within +/-2 % or +/-5 % for 60 s)\n" + "=" * 100)
print(RES.round(2).to_string(index=False))
print("\nNOTE: the reference SOC is itself integrated current, so in T1 Coulomb counting equals the reference (error 0) by construction.\n"
      "T1 therefore shows only the EKF's own error floor. The EKF's benefit shows in T2-T5, where Coulomb counting cannot correct its inputs.")
RES.to_csv("part3_results.csv", index=False)
 
# ---------- 7. figures ----------
def fig(test, fname, title):
    keys = [k for k in PL if k[0] == test and k[1] == "fuds"]
    f, ax = plt.subplots(len(keys), 2, figsize=(11, 3.6 * len(keys)), squeeze=False)
    for r, k in enumerate(keys):
        t, ref, c, e = PL[k]; t = t / 3600
        ax[r, 0].plot(t, ref * 100, label="Reference"); ax[r, 0].plot(t, c * 100, "--", label="Coulomb counting"); ax[r, 0].plot(t, e * 100, label="EKF")
        ax[r, 0].set_ylabel("SOC (%)"); ax[r, 0].set_title(f"FUDS {k[2]}%"); ax[r, 0].legend(); ax[r, 0].grid(alpha=.3)
        ax[r, 1].plot(t, (c - ref) * 100, "--", label="Coulomb counting"); ax[r, 1].plot(t, (e - ref) * 100, label="EKF")
        ax[r, 1].axhline(2, ls=":", c="k", lw=.8); ax[r, 1].axhline(-2, ls=":", c="k", lw=.8)
        ax[r, 1].set_ylabel("SOC error (%)"); ax[r, 1].legend(); ax[r, 1].grid(alpha=.3)
    ax[-1, 0].set_xlabel("Time (h)"); ax[-1, 1].set_xlabel("Time (h)"); f.suptitle(title); f.tight_layout(); f.savefig(fname, dpi=200); plt.show()
fig("T2a +10% initial SOC", "fig5_initial_soc_error.png", "Fig. 5  Recovery from +10 % initial SOC error")
fig("T3 Current bias 50 mA + noise 20 mA", "fig6_current_bias.png", "Fig. 6  Current-sensor bias and noise")
if IN_COLAB:
    for f in ("part3_results.csv", "fig5_initial_soc_error.png", "fig6_current_bias.png"): files.download(f)


