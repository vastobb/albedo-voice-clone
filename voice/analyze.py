#!/usr/bin/env python3
"""Acoustic analysis of a voice clip: pitch, energy, spectrum, stereo field."""
import json
import sys

import numpy as np
import soundfile as sf


def autocorr_f0(frame: np.ndarray, sr: int, fmin: float, fmax: float):
    """Estimate F0 of one frame via autocorrelation. Returns (f0, confidence)."""
    frame = frame - frame.mean()
    n = len(frame)
    w = np.hanning(n)
    x = frame * w
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    X = np.fft.rfft(x, nfft)
    ac = np.fft.irfft(X * np.conj(X), nfft)[:n]
    ac = ac / (ac[0] + 1e-12)
    lag_min = int(sr / fmax)
    lag_max = min(int(sr / fmin), n - 1)
    seg = ac[lag_min:lag_max]
    if len(seg) < 3:
        return 0.0, 0.0
    k = int(np.argmax(seg))
    peak = seg[k]
    if 0 < k < len(seg) - 1:
        a, b, c = seg[k - 1], seg[k], seg[k + 1]
        denom = (a - 2 * b + c)
        delta = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
    else:
        delta = 0.0
    lag = lag_min + k + delta
    if lag <= 0:
        return 0.0, 0.0
    return sr / lag, float(peak)


def main():
    path = sys.argv[1]
    x, sr = sf.read(path, always_2d=True, dtype="float64")
    mono = x.mean(axis=1)
    n = len(mono)
    dur = n / sr

    out = {"file": path, "sr": sr, "channels": x.shape[1], "duration_s": round(dur, 3)}

    peak = float(np.max(np.abs(mono)))
    rms_all = float(np.sqrt(np.mean(mono**2)))
    out["peak_dbfs"] = round(20 * np.log10(peak + 1e-12), 2)
    out["rms_dbfs"] = round(20 * np.log10(rms_all + 1e-12), 2)
    out["crest_db"] = round(20 * np.log10(peak / (rms_all + 1e-12)), 2)

    if x.shape[1] == 2:
        corr = float(np.corrcoef(x[:, 0], x[:, 1])[0, 1])
        mid = (x[:, 0] + x[:, 1]) / 2
        side = (x[:, 0] - x[:, 1]) / 2
        out["stereo_correlation"] = round(corr, 3)
        out["side_vs_mid_db"] = round(20 * np.log10(
            (np.sqrt(np.mean(side**2)) + 1e-12) / (np.sqrt(np.mean(mid**2)) + 1e-12)), 2)

    hop = int(0.010 * sr)
    win = 4 * hop
    nfr = max(1, (n - win) // hop + 1)
    rms_env = np.array([
        np.sqrt(np.mean(mono[i * hop: i * hop + win] ** 2) + 1e-15)
        for i in range(nfr)
    ])
    env_db = 20 * np.log10(rms_env + 1e-12)
    out["rms_env_db"] = {"max": round(float(env_db.max()), 1),
                         "p50": round(float(np.percentile(env_db, 50)), 1),
                         "p10": round(float(np.percentile(env_db, 10)), 1)}

    thr = env_db.max() - 40
    out["silence_ratio_pct"] = round(100 * float(np.mean(env_db < thr)), 1)

    from scipy.signal import find_peaks
    thr2 = env_db.max() - 18
    pk, _ = find_peaks(env_db, height=thr2, distance=int(0.08 / 0.01))
    out["energy_peaks"] = int(len(pk))
    if dur > 0:
        out["approx_peaks_per_s"] = round(len(pk) / dur, 2)

    fwin = 2048
    fhop = int(0.010 * sr)
    f0s, confs = [], []
    for i in range(0, n - fwin, fhop):
        fr = mono[i: i + fwin]
        r = np.sqrt(np.mean(fr**2))
        if 20 * np.log10(r + 1e-12) < env_db.max() - 45:
            continue
        f0, conf = autocorr_f0(fr, sr, 70, 500)
        f0s.append(f0)
        confs.append(conf)
    f0s = np.array(f0s)
    confs = np.array(confs)
    voiced = (confs > 0.25) & (f0s > 0)
    f0v = f0s[voiced]
    if len(f0v) > 0:
        out["f0_hz"] = {
            "n_voiced_frames": int(len(f0v)),
            "voiced_ratio_pct": round(100 * len(f0v) / max(1, len(f0s)), 1),
            "mean": round(float(np.mean(f0v)), 1),
            "median": round(float(np.median(f0v)), 1),
            "p5": round(float(np.percentile(f0v, 5)), 1),
            "p95": round(float(np.percentile(f0v, 95)), 1),
        }

    from scipy.signal import stft
    f, t, Z = stft(mono, fs=sr, nperseg=2048, noverlap=1536, window="hann")
    mag = np.abs(Z) + 1e-12
    freqs = f[:, None]
    centroid = (np.sum(freqs * mag, axis=0) / np.sum(mag, axis=0))
    gmean = np.exp(np.mean(np.log(mag), axis=0))
    amean = np.mean(mag, axis=0)
    flatness = gmean / amean
    power = mag**2
    total = np.sum(power, axis=0) + 1e-15
    band = lambda a, b: np.sum(power[(f >= a) & (f < b)], axis=0) / total
    out["spectral"] = {
        "centroid_mean_hz": round(float(np.mean(centroid)), 1),
        "flatness_mean": round(float(np.mean(flatness)), 4),
        "band_energy_frac": {
            "0-150": round(float(np.mean(band(0, 150))), 4),
            "150-400": round(float(np.mean(band(150, 400))), 4),
            "400-1k": round(float(np.mean(band(400, 1000))), 4),
            "1k-4k": round(float(np.mean(band(1000, 4000))), 4),
            "4k-8k": round(float(np.mean(band(4000, 8000))), 4),
            "8k+": round(float(np.mean(band(8000, sr / 2))), 4),
        },
    }

    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
