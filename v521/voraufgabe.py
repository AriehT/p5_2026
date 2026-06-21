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


# Gaußfunktion mit Fläche F als Parameter
# Integral über Gauß = F
def gauss_integral(x, F, mu, sigma):
    return F / (np.sqrt(2 * np.pi) * sigma) * np.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )


def gauss_untergrund(x, F, mu, sigma, m, b):
    return gauss_integral(x, F, mu, sigma) + m * x + b


def doppel_gauss_untergrund(x, F1, mu1, sigma1, F2, mu2, sigma2, m, b):
    return (
        gauss_integral(x, F1, mu1, sigma1)
        + gauss_integral(x, F2, mu2, sigma2)
        + m * x
        + b
    )


def chi2_red(y, y_fit, dy, anzahl_parameter):
    chi2 = np.sum(((y - y_fit) / dy) ** 2)
    freiheitsgrade = len(y) - anzahl_parameter
    return chi2 / freiheitsgrade


def fit_einzelpeak(kanal, counts, start, ende):
    maske = (kanal >= start) & (kanal <= ende)
    x = kanal[maske]
    y = counts[maske]

    dy = np.sqrt(np.maximum(y, 1))

    mu0 = x[np.argmax(y)]
    sigma0 = (ende - start) / 10
    F0 = (np.max(y) - np.min(y)) * sigma0 * np.sqrt(2 * np.pi)
    m0 = 0
    b0 = np.min(y)

    popt, pcov = curve_fit(
        gauss_untergrund,
        x,
        y,
        p0=[F0, mu0, sigma0, m0, b0],
        sigma=dy,
        absolute_sigma=True,
        maxfev=50000
    )

    fehler = np.sqrt(np.diag(pcov))

    F, mu, sigma, m, b = popt
    dF, dmu, dsigma, dm, db = fehler

    fit = gauss_untergrund(x, *popt)
    chi_red = chi2_red(y, fit, dy, 5)

    return {
        "typ": "einzel",
        "x": x,
        "fit": fit,
        "F": F,
        "dF": dF,
        "mu": mu,
        "dmu": dmu,
        "sigma": sigma,
        "dsigma": dsigma,
        "FWHM": 2 * np.sqrt(2 * np.log(2)) * sigma,
        "m": m,
        "dm": dm,
        "b": b,
        "db": db,
        "chi_red": chi_red,
        "start": start,
        "ende": ende
    }


def fit_doppelpeak(kanal, counts, start, ende):
    maske = (kanal >= start) & (kanal <= ende)
    x = kanal[maske]
    y = counts[maske]

    dy = np.sqrt(np.maximum(y, 1))

    # Startwerte für Peak 4 und Peak 5
    p0 = [
        30000, 3250, 100,
        70000, 3900, 180,
        0, np.min(y)
    ]

    grenzen_unten = [
        0, start, 1,
        0, start, 1,
        -np.inf, -np.inf
    ]

    grenzen_oben = [
        np.inf, ende, ende - start,
        np.inf, ende, ende - start,
        np.inf, np.inf
    ]

    popt, pcov = curve_fit(
        doppel_gauss_untergrund,
        x,
        y,
        p0=p0,
        sigma=dy,
        absolute_sigma=True,
        bounds=(grenzen_unten, grenzen_oben),
        maxfev=100000
    )

    fehler = np.sqrt(np.diag(pcov))

    F1, mu1, sigma1, F2, mu2, sigma2, m, b = popt
    dF1, dmu1, dsigma1, dF2, dmu2, dsigma2, dm, db = fehler

    fit = doppel_gauss_untergrund(x, *popt)
    chi_red = chi2_red(y, fit, dy, 8)

    return {
        "typ": "doppel",
        "x": x,
        "fit": fit,
        "F1": F1,
        "dF1": dF1,
        "mu1": mu1,
        "dmu1": dmu1,
        "sigma1": sigma1,
        "dsigma1": dsigma1,
        "FWHM1": 2 * np.sqrt(2 * np.log(2)) * sigma1,
        "F2": F2,
        "dF2": dF2,
        "mu2": mu2,
        "dmu2": dmu2,
        "sigma2": sigma2,
        "dsigma2": dsigma2,
        "FWHM2": 2 * np.sqrt(2 * np.log(2)) * sigma2,
        "m": m,
        "dm": dm,
        "b": b,
        "db": db,
        "chi_red": chi_red,
        "start": start,
        "ende": ende
    }


dateiname = "spectrum.txt"

kanal, counts = lade_spektrum(dateiname)

ergebnisse = []

# Peak 1
ergebnisse.append(fit_einzelpeak(kanal, counts, 200, 450))

# Peak 2 wird ausgelassen, weil dort kein echter Peak ist

# Peak 3
ergebnisse.append(fit_einzelpeak(kanal, counts, 600, 1100))

# Peak 4 und Peak 5 gemeinsam mit gleichem Untergrund
ergebnisse.append(fit_doppelpeak(kanal, counts, 3000, 4200))


print("\nFit-Ergebnisse")
print("====================================")

peaknummer = 1

for erg in ergebnisse:

    if erg["typ"] == "einzel":
        print(f"\nPeak {peaknummer}")
        print(f"Fitbereich : {erg['start']} - {erg['ende']}")
        print(f"mu         = {erg['mu']:.2f} ± {erg['dmu']:.2f}")
        print(f"sigma      = {erg['sigma']:.2f} ± {erg['dsigma']:.2f}")
        print(f"FWHM       = {erg['FWHM']:.2f}")
        print(f"Fläche F   = {erg['F']:.2f} ± {erg['dF']:.2f}")
        print(f"chi_red    = {erg['chi_red']:.2f}")
        peaknummer += 1

    else:
        print(f"\nPeak {peaknummer}")
        print(f"Fitbereich : {erg['start']} - {erg['ende']}")
        print(f"mu         = {erg['mu1']:.2f} ± {erg['dmu1']:.2f}")
        print(f"sigma      = {erg['sigma1']:.2f} ± {erg['dsigma1']:.2f}")
        print(f"FWHM       = {erg['FWHM1']:.2f}")
        print(f"Fläche F   = {erg['F1']:.2f} ± {erg['dF1']:.2f}")
        print(f"chi_red    = {erg['chi_red']:.2f}")
        peaknummer += 1

        print(f"\nPeak {peaknummer}")
        print(f"Fitbereich : {erg['start']} - {erg['ende']}")
        print(f"mu         = {erg['mu2']:.2f} ± {erg['dmu2']:.2f}")
        print(f"sigma      = {erg['sigma2']:.2f} ± {erg['dsigma2']:.2f}")
        print(f"FWHM       = {erg['FWHM2']:.2f}")
        print(f"Fläche F   = {erg['F2']:.2f} ± {erg['dF2']:.2f}")
        print(f"chi_red    = {erg['chi_red']:.2f}")
        peaknummer += 1


plt.figure(figsize=(12, 6))

plt.plot(
    kanal,
    counts,
    ".",
    markersize=1,
    linestyle="none",
    label="Beispiel-Spektrum"
)

farben = ["orange", "cyan", "magenta"]

for i, erg in enumerate(ergebnisse):
    plt.plot(
        erg["x"],
        erg["fit"],
        linewidth=1.5,
        color=farben[i],
        label=rf"Fit {i+1}: $\chi^2_{{red}}={erg['chi_red']:.2f}$"
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
