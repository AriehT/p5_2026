import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def lade_spektrum(dateiname):
    daten = np.loadtxt(dateiname)

    if daten.ndim == 1:
        counts = daten
        kanal = np.arange(len(counts))
    else:
        kanal = daten[:, 0]
        counts = daten[:, 1]

    return kanal.astype(float), counts.astype(float)


def gauss_untergrund(x, A, mu, sigma, m, b):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + m * x + b


def chi2_red(y, y_fit, dy, anzahl_parameter):
    chi2 = np.sum(((y - y_fit) / dy) ** 2)
    freiheitsgrade = len(y) - anzahl_parameter
    return chi2 / freiheitsgrade


def fit_gauss_untergrund(kanal, counts, start, ende):
    maske = (kanal >= start) & (kanal <= ende)

    x = kanal[maske]
    y = counts[maske]

    A0 = np.max(y) - np.min(y)
    mu0 = x[np.argmax(y)]
    sigma0 = (ende - start) / 10
    m0 = 0
    b0 = np.min(y)

    dy = np.sqrt(np.maximum(y, 1))

    popt, pcov = curve_fit(
        gauss_untergrund,
        x,
        y,
        p0=[A0, mu0, sigma0, m0, b0],
        sigma=dy,
        absolute_sigma=True,
        maxfev=50000
    )

    fehler = np.sqrt(np.diag(pcov))

    A, mu, sigma, m, b = popt
    dA, dmu, dsigma, dm, db = fehler

    fit = gauss_untergrund(x, *popt)

    chi_red = chi2_red(
        y,
        fit,
        dy,
        anzahl_parameter=5
    )

    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
    peakflaeche = A * sigma * np.sqrt(2 * np.pi)

    return {
        "x": x,
        "y": y,
        "fit": fit,
        "A": A,
        "dA": dA,
        "mu": mu,
        "dmu": dmu,
        "sigma": sigma,
        "dsigma": dsigma,
        "m": m,
        "dm": dm,
        "b": b,
        "db": db,
        "FWHM": fwhm,
        "peakflaeche": peakflaeche,
        "chi_red": chi_red,
        "start": start,
        "ende": ende
    }


dateiname = "spectrum.txt"

kanal, counts = lade_spektrum(dateiname)

fit_bereiche = [
    (200, 450),
    (475, 700),
    (750, 1100),
    (3000, 3550),
    (3600, 4200)
]

ergebnisse = []

for start, ende in fit_bereiche:
    try:
        erg = fit_gauss_untergrund(
            kanal,
            counts,
            start,
            ende
        )
        ergebnisse.append(erg)

    except Exception as e:
        print(f"Fehler im Bereich {start}-{ende}: {e}")


print("\nFit-Ergebnisse")
print("====================================")

for i, erg in enumerate(ergebnisse):
    print(f"\nPeak {i+1}")
    print(f"Fitbereich : {erg['start']} - {erg['ende']}")
    print(f"mu         = {erg['mu']:.2f} ± {erg['dmu']:.2f}")
    print(f"sigma      = {erg['sigma']:.2f} ± {erg['dsigma']:.2f}")
    print(f"FWHM       = {erg['FWHM']:.2f}")
    print(f"A          = {erg['A']:.2f} ± {erg['dA']:.2f}")
    print(f"Fläche     = {erg['peakflaeche']:.2f}")
    print(f"chi_red    = {erg['chi_red']:.2f}")


plt.figure(figsize=(12, 6))

plt.plot(
    kanal,
    counts,
    ".",
    markersize=1,
    linestyle="none",
    label="Beispiel-Spektrum"
)

farben = [
    "orange",
    "red",
    "cyan",
    "magenta",
    "gold"
]

for i, erg in enumerate(ergebnisse):
    farbe = farben[i]

    plt.plot(
        erg["x"],
        erg["fit"],
        color=farbe,
        linewidth=1.5,
        label=rf"Peak {i+1}: $\chi^2_{{red}}={erg['chi_red']:.2f}$"
    )

plt.title("Voraufgabe")
plt.xlabel("Kanalnummer / 1")
plt.ylabel("MCA-Zählrate / 1")

plt.xlim(0, 8200)
plt.ylim(0, 600)

plt.grid()
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()