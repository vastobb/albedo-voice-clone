#!/usr/bin/env python3
"""Prepare reference clips for voice cloning.

- decodes the input mp3 -> work/refs/ref_raw.wav (mono 44.1 kHz)
- cleans it (denoise + compressor + limiter) -> work/refs/ref_clean.wav
- builds emotion-morphed conditioning references ref_<emotion>.wav using
  formant-preserving pitch/tempo shifts + EQ. Prefers the ffmpeg rubberband
  filter; falls back to the rubberband CLI.
"""
import argparse
import math
import shutil
import subprocess
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
REFS = PROJ / "work" / "refs"

# emotion -> (pitch_factor, tempo, extra ffmpeg filters after the shift)
EMO_SPECS = {
    "happy":     (1.06, 1.05, "highshelf=f=4000:g=2.5"),
    "sad":       (0.93, 0.90, "highshelf=f=3500:g=-2.5,lowshelf=f=250:g=1.5"),
    "angry":     (0.97, 1.06, "equalizer=f=2500:width_type=o:width=1.5:g=3,"
                              "acompressor=threshold=-18dB:ratio=4:attack=5:release=80:makeup=3"),
    "teasing":   (0.96, 0.92, "highshelf=f=6000:g=-1.5"),
    "surprised": (1.08, 1.10, "highshelf=f=5000:g=3"),
}


def run(cmd):
    subprocess.run(cmd, check=True)


def have_ffmpeg_filter(name: str) -> bool:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"],
                         capture_output=True, text=True).stdout
    return name in out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(PROJ / "ref" / "albedo-1.mp3"))
    args = ap.parse_args()

    REFS.mkdir(parents=True, exist_ok=True)
    rb_filter = have_ffmpeg_filter("rubberband")
    print("ffmpeg rubberband filter:", "yes" if rb_filter else "no (using rubberband CLI)")

    raw = REFS / "ref_raw.wav"
    run(["ffmpeg", "-y", "-v", "error", "-i", args.input,
         "-ac", "1", "-ar", "44100", "-c:a", "pcm_f32le", str(raw)])

    clean = REFS / "ref_clean.wav"
    run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-af",
         "highpass=f=60,afftdn=nr=8:nf=-35,"
         "acompressor=threshold=-20dB:ratio=2.5:attack=8:release=120:makeup=2,"
         "alimiter=limit=0.95",
         "-ar", "44100", "-ac", "1", str(clean)])
    print("wrote", clean)

    for emo, (pf, tempo, extra) in EMO_SPECS.items():
        out = REFS / f"ref_{emo}.wav"
        if rb_filter:
            chain = (f"rubberband=pitch={pf}:tempo={tempo}:formant=preserved:"
                     f"pitchq=quality:channels=together,{extra}")
            run(["ffmpeg", "-y", "-v", "error", "-i", str(clean), "-af", chain, str(out)])
        else:
            tmp = REFS / f"_rb_{emo}.wav"
            semitones = 12.0 * math.log2(pf)
            run(["rubberband", "--time", str(tempo), "--pitch", f"{semitones:.4f}",
                 "--formant", str(clean), str(tmp)])
            tmp2 = REFS / f"_rb2_{emo}.wav"
            run(["ffmpeg", "-y", "-v", "error", "-i", str(tmp), "-af", extra, str(tmp2)])
            shutil.move(str(tmp2), str(out))
            tmp.unlink(missing_ok=True)
        print("wrote", out)

    for name in ("ref_raw", "ref_clean"):
        run(["ffmpeg", "-y", "-v", "error", "-i", str(REFS / f"{name}.wav"),
             "-lavfi", "showspectrumpic=s=1400x500:legend=1:scale=log:color=magma",
             str(REFS / f"{name}_spectrogram.png")])

    print("prep OK:", len(EMO_SPECS) + 2, "reference clips")


if __name__ == "__main__":
    main()
