"""Step 1: build the OCV-SOC curve from the CALCE INR 18650-20R low-current (C/20) OCV file.

Input : the Excel file downloaded from the CALCE page (columns: time [s], step, voltage [mV], current [mA], temp)
Output: ocv_soc_curve.csv, ocv_soc_curve.png, and printed fit statistics.
Convention in this file: discharge current is NEGATIVE, charge current is POSITIVE.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILE = "data/11_5_2015_low_current_OCV_test_SP20-1.xlsx"  

df = pd.read_excel(FILE)
df.columns = ["t", "step", "mV", "mA", "T"]
df["V"] = df["mV"] / 1000.0          # volts
df["I"] = df["mA"] / 1000.0          # amps (discharge < 0)
df["dt"] = df["t"].diff().fillna(0)  # seconds
df["Ah"] = df["I"] * df["dt"] / 3600 # signed charge increment

# Step 3 = C/20 discharge, step 5 = C/20 charge (check this against the printed step summary)
dis = df[df["step"] == 3].copy()
chg = df[df["step"] == 5].copy()

# normalised SOC for each branch (0..1) from its own integrated capacity
dis["Q"] = (-dis["Ah"]).cumsum()
Q_dis = dis["Q"].iloc[-1]
dis["SOC"] = 1 - dis["Q"] / Q_dis
chg["Q"] = chg["Ah"].cumsum()
Q_chg = chg["Q"].iloc[-1]
chg["SOC"] = chg["Q"] / Q_chg   # each branch normalised by its own capacity
print(f"Discharge capacity (C/20): {Q_dis:.3f} Ah | charge capacity: {Q_chg:.3f} Ah")

# put both branches on a common SOC grid and average (the CALCE page averages charge and discharge)
grid = np.linspace(0.0, 1.0, 501)
v_dis = np.interp(grid, dis["SOC"][::-1], dis["V"][::-1])
v_chg = np.interp(grid, chg["SOC"], chg["V"])
ocv = 0.5 * (v_dis + v_chg)

# The curve below ~5 % SOC collapses (cell polarisation near cut-off), so fit the useful window 5-100 %
W = grid >= 0.05
for n in range(4, 13):
    p = np.polyfit(grid[W], ocv[W], n)
    rmse = np.sqrt(np.mean((np.polyval(p, grid[W]) - ocv[W]) ** 2)) * 1000
    print(f"order {n:2d}: fit RMSE (5-100% SOC) = {rmse:6.2f} mV")
ORDER = 9   # lowest order that stays monotonic (slope > 0) with ~2 mV error; also keep the lookup table (np.interp) as an alternative
p = np.polyfit(grid[W], ocv[W], ORDER)
dp = np.polyder(p)
slope = np.polyval(dp, grid[W])
print(f"Chosen order {ORDER}. dOCV/dSOC range: {slope.min():.3f} to {slope.max():.3f} V per unit SOC")
m = (grid >= 0.10) & (grid <= 0.95)
print(f"Charge-discharge gap 10-95% SOC: mean {np.mean(np.abs(v_chg - v_dis)[m])*1000:.0f} mV, max {np.max(np.abs(v_chg - v_dis)[m])*1000:.0f} mV (IR drop + hysteresis; averaging cancels most of the IR part)")

pd.DataFrame({"SOC": grid, "OCV_V": ocv, "OCV_fit_V": np.where(W, np.polyval(p, grid), np.nan),
              "V_discharge": v_dis, "V_charge": v_chg}).to_csv("ocv_soc_curve.csv", index=False)
np.save("ocv_poly_coeffs.npy", p)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(grid * 100, v_dis, label="C/20 discharge")
ax[0].plot(grid * 100, v_chg, label="C/20 charge")
ax[0].plot(grid * 100, ocv, "k--", label="average (OCV)")
ax[0].plot(grid[W] * 100, np.polyval(p, grid[W]), "r:", label=f"polynomial fit (order {ORDER})")
ax[0].set_xlabel("SOC (%)"); ax[0].set_ylabel("Voltage (V)"); ax[0].set_title("Fig. A. OCV–SOC curve (INR 18650-20R)")
ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(grid[W] * 100, slope)
ax[1].set_xlabel("SOC (%)"); ax[1].set_ylabel("dOCV/dSOC (V per unit SOC)"); ax[1].set_title("Fig. B. Slope used by the EKF")
ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig("ocv_soc_curve.png", dpi=200)
print("saved ocv_soc_curve.csv, ocv_soc_curve.png, ocv_poly_coeffs.npy")
