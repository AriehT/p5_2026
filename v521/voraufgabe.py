import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ------------------------------------------------------------
# Modell: Gauß und linearer Untergrund

def gauss_with_bg(x, A, mu, sigma, m, b):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + (m * x + b)

# ------------------------------------------------------------
# Peak-Fit-Funktion
def fit_peak(x, y, fit_min, fit_max):
    # Fitbereich 
    mask = (x >= fit_min) & (x <= fit_max)
    x_fit = x[mask]
    y_fit = y[mask]

    # Startwerte 
    A0 = np.max(y_fit) - np.min(y_fit)
    mu0 = x_fit[np.argmax(y_fit)]
    sigma0 = (fit_max - fit_min) / 10
    m0 = 0
    b0 = np.min(y_fit)

    p0 = [A0, mu0, sigma0, m0, b0]

    #durchführen
    popt, pcov = curve_fit(gauss_with_bg, x_fit, y_fit, p0=p0)
    perr = np.sqrt(np.diag(pcov))

    return popt, perr, x_fit, y_fit

# ------------------------------------------------------------
# Daten laden

data = np.loadtxt("spectrum.txt")
x = data[:, 0]
y = data[:, 1]

# ------------------------------------------------------------
# Peak bei Kanal zB 511 fitten 

fit_min = 480
fit_max = 540

popt, perr, xf, yf = fit_peak(x, y, fit_min, fit_max)

A, mu, sigma, m, b = popt
dA, dmu, dsigma, dm, db = perr

print("----- Fit-Ergebnisse -----")
print(f"Amplitude A       = {A:.2f} ± {dA:.2f}")
print(f"Peakposition mu   = {mu:.2f} ± {dmu:.2f}")
print(f"Breite sigma      = {sigma:.2f} ± {dsigma:.2f}")
print(f"Untergrund m      = {m:.4f} ± {dm:.4f}")
print(f"Untergrund b      = {b:.2f} ± {db:.2f}")
print(f"Peak-Integral     = {A * sigma * np.sqrt(2*np.pi):.2f}")

# ------------------------------------------------------------
# Plot

plt.figure(figsize=(10,5))
plt.plot(x, y, label="Spektrum", color="black")
plt.plot(xf, gauss_with_bg(xf, *popt), label="Fit", color="red", linewidth=2)
plt.axvline(mu, color="blue", linestyle="--", label=f"Peak bei {mu:.1f}")
plt.xlabel("Kanal")
plt.ylabel("Zählrate")
plt.legend()
plt.title("Peak-Fit (Gauß + Untergrund)")
plt.show()
