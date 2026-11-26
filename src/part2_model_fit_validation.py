# SOC estimation project - Part 2, VERSION 3 (Colab).
# HOW TO USE: paste this WHOLE file into ONE Colab cell and run it. When the upload box appears, select ALL your files together:
#   the low-current OCV file + DST 80% + DST 50% + FUDS 80% + FUDS 50% (missing ones are skipped).
# Tip: use Runtime > Restart session first, so old duplicate uploads like '(1)' are cleared.
 
# %% [markdown]
# # SOC estimation project - Part 2 (v2): reference SOC, Thevenin model fit and validation
# Run the cells **in order**. Needed files from the CALCE INR 18650-20R page (25 C): the **low-current OCV** file and the **DST** and **FUDS** files (80% and 50% versions). Missing levels are skipped automatically.
#
# Each DST/FUDS file starts with a set-up sequence (charge to 4.2 V, hold, 1 A discharge to the 80% / 50% level, rest) and only then runs the dynamic test. This notebook anchors SOC = 100 % at the end of the full charge, integrates the current from there, and cuts out the dynamic part for fitting.
 
# %%
import re, json, glob
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.optimize import least_squares
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
print("Running in Colab:", IN_COLAB)
 
# %% [markdown]
# ## 1. Upload the files\nSelect all your files together (OCV, DST 80/50, FUDS 80/50).
 
# %%
if IN_COLAB:
    uploaded = files.upload()
    print(list(uploaded.keys()))
 
# %% [markdown]
# ## 2. Match files by name\nCheck the printout. If something is wrong, set the name by hand where indicated.
 
# %%
all_files = sorted(glob.glob("*.xlsx") + glob.glob("*.xls") + glob.glob("*.xlsm") + glob.glob("*.csv"), key=lambda f: (len(f), f))
print("Files in this session:")
for f in all_files: print("   ", repr(f))
 
def norm(f): return f.lower().replace("_", "").replace(" ", "")
ocv_hits = [f for f in all_files if "ocv" in f.lower() and "incremental" not in f.lower()]
OCV_FILE = ocv_hits[0] if ocv_hits else None
DYN = {}
for prof in ("DST", "FUDS"):
    for lvl in ("80", "50"):
        hits = [f for f in all_files if prof.lower() in f.lower() and "bjdst" not in f.lower() and f"{lvl}soc" in norm(f)]
        if hits: DYN[(prof, lvl)] = hits[0]
 
# To override a guess, uncomment and edit, e.g.:
# OCV_FILE = "my_ocv_file.xlsx"
# DYN[("DST", "80")] = "my_dst80_file.xls"
 
print("\nLow-current OCV file:", OCV_FILE)
for k, v in DYN.items(): print(f"{k[0]} {k[1]}% file:", v)
if OCV_FILE is None or not any(k[0] == "DST" for k in DYN) or not any(k[0] == "FUDS" for k in DYN):
    raise ValueError("Need the OCV file, at least one DST file and at least one FUDS file. Look at the list above and set the names by hand, or upload the missing file(s).")
 
# %% [markdown]
# ## 3. Settings\nOnly change these if a later printout tells you to.
 
# %%
DISCHARGE_NEGATIVE = True   # raw CALCE files: discharge current is negative. Set False if your plots show the opposite.
Q_AH = None                 # capacity in Ah for SOC. None = use the C/20 capacity measured in the OCV file (keeps SOC and the OCV curve on the same scale)
DT = 1.0                    # seconds (data is resampled to this step)
POLY_ORDER = 9              # OCV polynomial order
DYN_THRESH_A = 1.1          # the dynamic test starts at the first |I| above this (the set-up discharge is 1.0 A)
PRE_S = 120                 # keep this many seconds of rest before the dynamic test starts
FIT_SOC_MIN = 0.15          # fit and report the model only above this SOC (the cell voltage collapses near cut-off, which this simple model cannot follow)
TAU_MAX = 600               # upper bound for the polarisation time constant (s); typical values are tens to a few hundred seconds
 
# %% [markdown]
# ## 4. Loader\nReads a CALCE file, finds time / voltage / current columns by name, converts units, and makes **discharge current positive**.
 
# %%
def _find(cols, patterns):
    for p in patterns:
        for c in cols:
            if re.search(p, str(c).lower()): return c
    return None
 
def load_calce(path, sheet=None):
    if path.lower().endswith((".xlsx", ".xlsm", ".xls")):
        sheets = pd.read_excel(path, sheet_name=None)
        name = sheet or max(sheets, key=lambda k: len(sheets[k]))
        raw = sheets[name]
        print(f"{path}: using sheet '{name}'")
    else:
        raw = pd.read_csv(path)
    cols = list(raw.columns)
    tc = _find(cols, [r"test.?time", r"duration", r"^time", r"time"])
    vc = _find(cols, [r"\bmv\b", r"volt"])
    ic = _find(cols, [r"\bma\b", r"curr"])
    if None in (tc, vc, ic): raise ValueError(f"Could not find time/voltage/current columns in {cols}")
    print(f"   time='{tc}', voltage='{vc}', current='{ic}'")
    t, v, i = (pd.to_numeric(raw[c], errors="coerce") for c in (tc, vc, ic))
    ok = t.notna() & v.notna() & i.notna()
    t, v, i = t[ok].to_numpy(float), v[ok].to_numpy(float), i[ok].to_numpy(float)
    if re.search(r"\bmv\b", str(vc).lower()) or np.abs(v).max() > 20: v = v / 1000.0
    if re.search(r"\bma\b", str(ic).lower()) or np.abs(i).max() > 20: i = i / 1000.0
    return pd.DataFrame({"t": t, "V": v, "I_raw": i})
 
def prepare(df, dt=DT):
    t, idx = np.unique(df["t"].to_numpy(float), return_index=True)
    v, i = df["V"].to_numpy()[idx], df["I_raw"].to_numpy()[idx]
    tu = np.arange(t[0], t[-1], dt)
    V, I = np.interp(tu, t, v), np.interp(tu, t, i)
    if DISCHARGE_NEGATIVE: I = -I
    return pd.DataFrame({"t": tu - tu[0], "V": V, "I": I})   # I > 0 means discharge
 
# %% [markdown]
# ## 5. OCV-SOC curve from the low-current test
 
# %%
def longest_run(mask):
    m = np.r_[False, mask, False].astype(int)
    d = np.diff(m); s = np.where(d == 1)[0]; e = np.where(d == -1)[0]
    k = np.argmax(e - s); return s[k], e[k]
 
ocv_d = prepare(load_calce(OCV_FILE))
s1, e1 = longest_run(ocv_d.I.to_numpy() > 0.03); s2, e2 = longest_run(ocv_d.I.to_numpy() < -0.03)
dis, chg = ocv_d.iloc[s1:e1], ocv_d.iloc[s2:e2]
Q_dis = dis.I.sum() * DT / 3600; Q_chg = (-chg.I).sum() * DT / 3600
print(f"C/20 discharge capacity {Q_dis:.3f} Ah | charge capacity {Q_chg:.3f} Ah")
soc_dis = 1 - np.cumsum(dis.I.to_numpy()) * DT / 3600 / Q_dis
soc_chg = np.cumsum(-chg.I.to_numpy()) * DT / 3600 / Q_chg
grid = np.linspace(0, 1, 501)
v_dis = np.interp(grid, soc_dis[::-1], dis.V.to_numpy()[::-1]); v_chg = np.interp(grid, soc_chg, chg.V.to_numpy())
ocv = 0.5 * (v_dis + v_chg); W = grid >= 0.05
poly = np.polyfit(grid[W], ocv[W], POLY_ORDER)
rmse_fit = np.sqrt(np.mean((np.polyval(poly, grid[W]) - ocv[W])**2)) * 1000
print(f"OCV polynomial (order {POLY_ORDER}) fit error {rmse_fit:.2f} mV")
def ocv_fun(soc): return np.polyval(poly, np.clip(soc, 0.05, 1.0))
_s = np.linspace(0.05, 1.0, 951); _o = np.maximum.accumulate(ocv_fun(_s))
def soc_from_ocv(v): return float(np.interp(v, _o, _s))
if Q_AH is None: Q_AH = Q_dis
print("Capacity used for SOC:", round(Q_AH, 3), "Ah")
plt.figure(figsize=(6, 4)); plt.plot(grid*100, v_dis, label="discharge"); plt.plot(grid*100, v_chg, label="charge")
plt.plot(grid*100, ocv, "k--", label="average"); plt.plot(grid[W]*100, ocv_fun(grid[W]), "r:", label="fit")
plt.xlabel("SOC (%)"); plt.ylabel("Voltage (V)"); plt.title("Fig. 1  OCV-SOC curve"); plt.legend(); plt.grid(alpha=.3); plt.show()
 
# %% [markdown]
# ## 6. Build the dynamic segments with a reference SOC
# Steps per file: (1) find the end of the full charge (voltage reached about 4.2 V), (2) set **SOC = 100 %** just before the first discharge after it, (3) integrate current from there, (4) keep only the dynamic test (plus a short rest before it).
# **Note:** CALCE gives no separate SOC measurement, so this reference is a current integration. Later experiments therefore inject errors into the estimators' inputs.
 
# %%
def build_segment(path, label):
    d = prepare(load_calce(path)); I, V = d.I.to_numpy(), d.V.to_numpy()
    full = np.where(V >= 4.18)[0]
    if len(full) == 0: raise ValueError(f"{label}: voltage never reaches 4.18 V, cannot find the full-charge point")
    k = full[0] + np.argmax(I[full[0]:] > 0.5)
    if not I[k] > 0.5: raise ValueError(f"{label}: no discharge found after the full charge")
    anchor = k - 1
    soc = np.full(len(d), np.nan)
    soc[anchor:] = 1.0 - np.r_[0, np.cumsum(I[anchor:-1])] * DT / 3600 / Q_AH
    pulse_end = k + np.argmax(I[k:] < 0.5)              # end of the set-up discharge pulse
    after = pulse_end + 120
    j = after + np.argmax(np.abs(I[after:]) > 0.05)      # first current after the rest = start of the dynamic test
    s = max(pulse_end, j - PRE_S)
    seg = d.iloc[s:].copy(); seg["SOC"] = soc[s:]; seg = seg.reset_index(drop=True); seg["t"] = seg.t - seg.t.iloc[0]
    rest_ok = np.abs(seg.I.to_numpy()[:30]).max() < 0.02
    soc_rest = soc_from_ocv(np.median(seg.V.to_numpy()[:30]))
    print(f"{label}: full charge at t={d.t[anchor]/3600:.2f} h (V={V[anchor]:.3f} V) -> rest after the set-up pulse, dynamic test starts t={d.t[j]/3600:.2f} h, lasts {seg.t.iloc[-1]/3600:.2f} h")
    print(f"   SOC at start of dynamic test: {seg.SOC.iloc[0]*100:.1f} % (integrated) | {soc_rest*100:.1f} % (from rest voltage, rough) | SOC at end {seg.SOC.iloc[-1]*100:.1f} %, end voltage {seg.V.iloc[-1]:.2f} V")
    if not rest_ok: print("   WARNING: the segment does not start at rest - the voltage-based SOC check above is not meaningful for this file")
    if seg.SOC.max() > 1.02 or seg.SOC.min() < 0: print("   WARNING: SOC leaves 0-100 %. Check DISCHARGE_NEGATIVE and the capacity.")
    return seg
 
SEG = {key: build_segment(path, f"{key[0]} {key[1]}%") for key, path in DYN.items()}
fit_keys  = [k for k in SEG if k[0] == "DST"]
test_keys = [k for k in SEG if k[0] == "FUDS"]
 
fig, ax = plt.subplots(len(SEG), 2, figsize=(12, 2.6 * len(SEG)), squeeze=False)
for r, (key, seg) in enumerate(SEG.items()):
    ax[r, 0].plot(seg.t/3600, seg.I, lw=.5); ax[r, 0].set_ylabel(f"{key[0]} {key[1]}%\nI (A, + = discharge)")
    ax[r, 1].plot(seg.t/3600, seg.V, lw=.5);  ax[r, 1].set_ylabel("V (V)")
    for c in (0, 1): ax[r, c].grid(alpha=.3)
ax[-1, 0].set_xlabel("time (h)"); ax[-1, 1].set_xlabel("time (h)")
fig.suptitle("Fig. 2  Dynamic test segments used (after the set-up sequence)"); fig.tight_layout(); plt.show()
 
# %% [markdown]
# ## 7. Fit the Thevenin model on the DST segments
# `V1[k] = a*V1[k-1] + R1*(1-a)*I[k]`, `a = exp(-dt/tau)`, `Vt = OCV(SOC) + dV - I*R0 - V1`.
# `dV` is a constant offset that absorbs the difference between the averaged OCV curve and the OCV on the discharge side (hysteresis). It is fitted together with R0, R1 and tau.
 
# %%
def simulate_v(params, seg):
    R0, R1, tau, dV = params
    a = np.exp(-DT / tau); I = seg.I.to_numpy()
    V1 = lfilter([R1 * (1 - a)], [1, -a], I)
    return ocv_fun(seg.SOC.to_numpy()) + dV - I * R0 - V1
 
def residuals(x, keys):
    out = []
    for k in keys:
        seg = SEG[k]; m = seg.SOC.to_numpy() > FIT_SOC_MIN
        out.append((simulate_v(x, seg) - seg.V.to_numpy())[m])
    return np.concatenate(out)
 
best = None
for tau0 in (20, 80, 300):
    r = least_squares(residuals, [0.12, 0.05, tau0, -0.03], args=(fit_keys,),
                      bounds=([0.005, 0.001, 3, -0.15], [1.0, 1.0, TAU_MAX, 0.15]), x_scale=[0.1, 0.05, 50, 0.05])
    if best is None or r.cost < best.cost: best = r
R0, R1, TAU, DV = best.x; C1 = TAU / R1
print(f"R0 = {R0*1000:.1f} mOhm | R1 = {R1*1000:.1f} mOhm | C1 = {C1:.0f} F | tau = {TAU:.1f} s | OCV offset dV = {DV*1000:.1f} mV")
print("Fitted on:", [f"{k[0]} {k[1]}%" for k in fit_keys], f"| only samples with SOC > {FIT_SOC_MIN*100:.0f} %")
if TAU > 0.97 * TAU_MAX or TAU < 3.2: print("WARNING: tau sits at its bound, so the single RC branch is not describing a real polarisation time constant. Say so in the report or send me this printout.")
 
STATS = {}
def report(key):
    seg = SEG[key]; vm = simulate_v((R0, R1, TAU, DV), seg); soc = seg.SOC.to_numpy()
    m = soc > FIT_SOC_MIN; e = (vm - seg.V.to_numpy())[m]
    rm, ma, mx = np.sqrt(np.mean(e**2))*1000, np.mean(np.abs(e))*1000, np.max(np.abs(e))*1000
    m2 = soc > 0.07; e2 = (vm - seg.V.to_numpy())[m2]; rm2 = np.sqrt(np.mean(e2**2))*1000
    tag = "fit" if key in fit_keys else "validation"
    print(f"{key[0]} {key[1]}% ({tag}): SOC>{FIT_SOC_MIN*100:.0f}%: RMSE {rm:.1f} mV | MAE {ma:.1f} mV | max {mx:.0f} mV   || incl. end-of-discharge (SOC>7%): RMSE {rm2:.1f} mV")
    STATS[f"{key[0]}_{key[1]}"] = dict(role=tag, rmse_mV=float(rm), mae_mV=float(ma), max_mV=float(mx), rmse_incl_end_mV=float(rm2))
    return vm
VM = {k: report(k) for k in SEG}
 
# %% [markdown]
# ## 8. Validation on FUDS (data the fit never saw)
 
# %%
fig, ax = plt.subplots(2 * len(test_keys), 1, figsize=(11, 4 * len(test_keys)), sharex=False)
ax = np.atleast_1d(ax)
for n, k in enumerate(test_keys):
    seg = SEG[k]
    ax[2*n].plot(seg.t/3600, seg.V, lw=.6, label="measured"); ax[2*n].plot(seg.t/3600, VM[k], lw=.6, label="model", alpha=.8)
    ax[2*n].set_ylabel("Terminal voltage (V)"); ax[2*n].legend(); ax[2*n].grid(alpha=.3); ax[2*n].set_title(f"FUDS {k[1]}%: measured vs model")
    ax[2*n+1].plot(seg.t/3600, (VM[k] - seg.V)*1000, lw=.5); ax[2*n+1].set_ylabel("Model error (mV)"); ax[2*n+1].set_xlabel("time (h)"); ax[2*n+1].grid(alpha=.3)
fig.suptitle("Fig. 3  Model validation on FUDS"); fig.tight_layout(); fig.savefig("fig3_validation_fuds.png", dpi=200); plt.show()
 
# %% [markdown]
# ## 9. Coulomb counting with injected errors (preview of your experiments)
# Clean Coulomb counting equals the reference by construction. The interesting case is wrong inputs: a wrong starting SOC and a small current-sensor bias.
 
# %%
rng = np.random.default_rng(1)
SOC_OFFSET, BIAS_A, NOISE_A = 0.10, 0.02, 0.01     # +10 % start error, 20 mA bias, 10 mA RMS noise
seg = SEG[test_keys[0]]
I_meas = seg.I.to_numpy() + BIAS_A + rng.normal(0, NOISE_A, len(seg))
soc_ref = seg.SOC.to_numpy()
soc_cc = (soc_ref[0] + SOC_OFFSET) - np.r_[0, np.cumsum(I_meas[:-1])] * DT / 3600 / Q_AH
err = (soc_cc - soc_ref) * 100
print(f"Coulomb counting with errors on FUDS {test_keys[0][1]}%: final error {err[-1]:.1f} % SOC | max |error| {np.max(np.abs(err)):.1f} % | RMSE {np.sqrt(np.mean(err**2)):.1f} %")
STATS["CC_with_errors"] = dict(on=f"FUDS_{test_keys[0][1]}", final_pct=float(err[-1]), max_pct=float(np.max(np.abs(err))),
                               rmse_pct=float(np.sqrt(np.mean(err**2))), soc_offset=SOC_OFFSET, bias_A=BIAS_A, noise_A=NOISE_A)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(seg.t/3600, soc_ref*100, label="reference"); ax[0].plot(seg.t/3600, soc_cc*100, label="Coulomb counting + errors")
ax[0].set_xlabel("time (h)"); ax[0].set_ylabel("SOC (%)"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(seg.t/3600, err); ax[1].set_xlabel("time (h)"); ax[1].set_ylabel("SOC error (%)"); ax[1].grid(alpha=.3)
fig.suptitle("Fig. 4  Coulomb counting drift"); fig.tight_layout(); fig.savefig("fig4_cc_drift.png", dpi=200); plt.show()
 
# %% [markdown]
# ## 10. Save and summarise\nKeep `model_parameters.json` - the EKF notebook will load it.
 
# %%
res = dict(R0_ohm=R0, R1_ohm=R1, C1_farad=C1, tau_s=TAU, dV_volt=DV, Q_ah=Q_AH, poly_order=POLY_ORDER, ocv_poly=list(map(float, poly)),
           ocv_fit_rmse_mV=float(rmse_fit), fit_on=[f"{k[0]}_{k[1]}" for k in fit_keys], stats=STATS)
json.dump(res, open("model_parameters.json", "w"), indent=2)
print("="*62); print("RESULTS TO COPY INTO YOUR REPORT"); print("="*62)
print(f"Capacity used (C/20, for SOC)  : {Q_AH:.3f} Ah")
print(f"OCV polynomial fit error       : {rmse_fit:.2f} mV (order {POLY_ORDER}, valid 5-100 % SOC)")
print(f"R0 = {R0*1000:.1f} mOhm, R1 = {R1*1000:.1f} mOhm, C1 = {C1:.0f} F, tau = {TAU:.1f} s, dV = {DV*1000:.1f} mV")
for k, v in STATS.items():
    if k != "CC_with_errors": print(f"Voltage RMSE {k:8s} ({v['role']:10s}): {v['rmse_mV']:.1f} mV (SOC>{FIT_SOC_MIN*100:.0f}%) | {v['rmse_incl_end_mV']:.1f} mV (SOC>7%, incl. end collapse)")
cc = STATS["CC_with_errors"]; print(f"Coulomb counting with injected errors: final {cc['final_pct']:.1f} %, max {cc['max_pct']:.1f} % SOC")
if IN_COLAB:
    for f in ["model_parameters.json", "fig3_validation_fuds.png", "fig4_cc_drift.png"]: files.download(f)


