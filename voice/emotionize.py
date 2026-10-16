#!/usr/bin/env python3
"""Emotion shaping stage — raw XTTS output -> emotion-consistent delivery.

Stage 2 of the emotion pipeline (stage 1 is the morphed conditioning reference
in prep_refs.py). Uses the ffmpeg rubberband filter for formant-preserving
pitch/tempo when available, else the rubberband CLI. Then EQ, dynamics and
EBU R128 loudness normalisation, and a final resample to 44.1 kHz.

Usage: emotionize.py --in raw.wav --emotion happy --out stage2.wav
"""
import argparse
import math
import os
import subprocess
from pathlib import Path

# emotion -> (pitch_factor | None, tempo | None, remaining ffmpeg chain)
SPECS = {
    # straight, clean, even
    "neutral": (None, None,
                "highpass=f=45,loudnorm=I=-16:TP=-1.5:LRA=11,aresample=44100"),
    # brighter, faster, lifted pitch -> energetic cheer
    "happy": (1.03, 1.03,
              "highpass=f=45,highshelf=f=4000:g=2,"
              "equalizer=f=9000:width_type=o:width=1:g=1,"
              "loudnorm=I=-16:TP=-1.5:LRA=11,aresample=44100"),
    # slower, lower, darker, softer -> sorrowful weight
    "sad": (0.94, 0.90,
            "highpass=f=45,highshelf=f=3500:g=-2.5,lowshelf=f=250:g=1.5,"
            "loudnorm=I=-16:TP=-1.5:LRA=11,aresample=44100"),
    # snappier tempo, presence boost, compression + soft clip -> fury
    "angry": (0.99, 1.05,
              "highpass=f=45,equalizer=f=2500:width_type=o:width=1.5:g=3,"
              "acompressor=threshold=-18dB:ratio=4:attack=5:release=80:makeup=3,"
              "asoftclip=type=tanh:threshold=0.9,"
              "loudnorm=I=-15:TP=-1.5:LRA=9,aresample=44100"),
    # slow, slightly low + softened highs -> the classic 'ara ara' tease
    "teasing": (0.97, 0.94,
                "highpass=f=45,highshelf=f=6000:g=-1.5,"
                "loudnorm=I=-16:TP=-1.5:LRA=11,aresample=44100"),
    # fast, high, bright -> startle/shock
    "surprised": (1.06, 1.08,
                  "highpass=f=45,highshelf=f=5000:g=3,"
                  "loudnorm=I=-15.5:TP=-1.5:LRA=9,aresample=44100"),
}

_RB_FILTER = None


def rb_filter_available() -> bool:
    global _RB_FILTER
    if _RB_FILTER is None:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"],
                             capture_output=True, text=True).stdout
        _RB_FILTER = "rubberband" in out
    return _RB_FILTER


def process(inp: str, emotion: str, outp: str) -> None:
    pf, tempo, rest = SPECS[emotion]
    os.makedirs(os.path.dirname(os.path.abspath(outp)), exist_ok=True)
    if pf is None:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", inp, "-af", rest, outp],
                       check=True)
        return
    if rb_filter_available():
        chain = (f"rubberband=pitch={pf}:tempo={tempo}:formant=preserved:"
                 f"pitchq=quality:channels=together,{rest}")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", inp, "-af", chain, outp],
                       check=True)
    else:
        tmp = str(Path(outp).with_suffix(".rb.wav"))
        semitones = 12.0 * math.log2(pf)
        subprocess.run(["rubberband", "--time", str(tempo), "--pitch", f"{semitones:.4f}",
                        "--formant", inp, tmp], check=True)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmp, "-af", rest, outp],
                       check=True)
        Path(tmp).unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--emotion", required=True, choices=sorted(SPECS))
    ap.add_argument("--out", dest="outp", required=True)
    args = ap.parse_args()
    process(args.inp, args.emotion, args.outp)
    print(f"OK {args.emotion}: {args.inp} -> {args.outp}")


if __name__ == "__main__":
    main()
