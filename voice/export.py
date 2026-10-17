#!/usr/bin/env python3
"""Pick the best candidate per emotion+language, trim, shape, and export finals.

Ranking: candidates whose Whisper ASR score passes --min-asr are preferred;
among them (or all candidates if none pass) the highest speaker similarity wins.

Reads work/raw/run_meta.json + work/eval_stage1.json, trims leading/trailing
silence, applies the emotion DSP (voice/emotionize.py), and writes
output/<lang>/albedo_ara_sayonara_<emotion>.{wav,mp3,png} plus a per-language
demo reel, summary.json and README.md.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
RAW = PROJ / "work" / "raw"
OUT = PROJ / "output"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from emotionize import process as emotionize  # noqa: E402

EMO_ORDER = ["neutral", "happy", "sad", "angry", "teasing", "surprised"]


def run(cmd):
    subprocess.run(cmd, check=True)


def trim_silence(src: str, dst: str) -> None:
    """Trim leading/trailing silence, keeping 50 ms of padding."""
    chain = ("silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.05,"
             "areverse,"
             "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.05,"
             "areverse")
    run(["ffmpeg", "-y", "-v", "error", "-i", src, "-af", chain, dst])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-asr", type=float, default=0.35,
                    help="minimum ASR phrase score to prefer a candidate")
    args = ap.parse_args()

    meta = json.load(open(RAW / "run_meta.json"))
    ev = json.load(open(PROJ / "work" / "eval_stage1.json"))
    info = {os.path.basename(r["file"]): r for r in ev["results"]}

    gens = meta["gens"]
    langs = []
    for g in gens:
        if g["lang"] not in langs:
            langs.append(g["lang"])
    langs.sort(key=lambda l: (l != "ja", l))  # ja first

    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for lang in langs:
        lang_dir = OUT / lang
        lang_dir.mkdir(parents=True, exist_ok=True)

        picks = {}
        for g in gens:
            if g["lang"] != lang:
                continue
            base = os.path.basename(g["file"])
            r = info.get(base, {})
            emo = g["emotion"]
            cand = {
                "file": g["file"],
                "sim": r.get("spk_sim", 0.0),
                "asr": r.get("asr_score"),
                "transcript": r.get("transcript", ""),
                "seed": g["seed"],
                "dur_s": g["dur_s"],
            }
            if emo not in picks:
                picks[emo] = cand
                continue
            # prefer ASR-passing takes; then speaker similarity
            cur, new = picks[emo], cand
            cur_ok = (cur["asr"] or 0) >= args.min_asr
            new_ok = (new["asr"] or 0) >= args.min_asr
            if (new_ok and not cur_ok) or (new_ok == cur_ok and new["sim"] > cur["sim"]):
                picks[emo] = new

        reel_parts = []
        ordered = [e for e in EMO_ORDER if e in picks] + \
                  [e for e in picks if e not in EMO_ORDER]
        for emo in ordered:
            src = picks[emo]["file"]
            trimmed = PROJ / "work" / "stage2" / f"trim_{lang}_{emo}.wav"
            proc = PROJ / "work" / "stage2" / f"{lang}_{emo}.wav"
            trimmed.parent.mkdir(parents=True, exist_ok=True)
            trim_silence(src, str(trimmed))
            emotionize(str(trimmed), emo, str(proc))

            wav_out = lang_dir / f"albedo_ara_sayonara_{emo}.wav"
            mp3_out = lang_dir / f"albedo_ara_sayonara_{emo}.mp3"
            png_out = lang_dir / f"albedo_ara_sayonara_{emo}.png"
            run(["ffmpeg", "-y", "-v", "error", "-i", str(proc),
                 "-c:a", "pcm_s16le", str(wav_out)])
            run(["ffmpeg", "-y", "-v", "error", "-i", str(proc),
                 "-c:a", "libmp3lame", "-b:a", "320k", str(mp3_out)])
            run(["ffmpeg", "-y", "-v", "error", "-i", str(proc), "-lavfi",
                 "showspectrumpic=s=1600x520:legend=1:scale=log:color=magma",
                 str(png_out)])
            reel_parts.append(proc)
            summary.append({"lang": lang, "emotion": emo,
                            "source": os.path.basename(src),
                            "seed": picks[emo]["seed"],
                            "spk_sim": picks[emo]["sim"],
                            "asr_score": picks[emo]["asr"],
                            "transcript": picks[emo]["transcript"],
                            "gen_dur_s": picks[emo]["dur_s"]})
            print(f"[export] {lang}/{emo}: {os.path.basename(src)} "
                  f"(sim={picks[emo]['sim']:.3f}, asr={picks[emo]['asr']}, "
                  f"'{picks[emo]['transcript'][:40]}')")

        if reel_parts:
            gap = PROJ / "work" / "stage2" / "_gap.wav"
            run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                 "-i", "anullsrc=r=44100:cl=mono", "-t", "0.5", str(gap)])
            lst = PROJ / "work" / "stage2" / f"reel_{lang}.txt"
            with open(lst, "w") as fh:
                for w in reel_parts:
                    fh.write(f"file '{w}'\nfile '{gap}'\n")
            reel = lang_dir / "albedo_ara_sayonara_demo_reel.mp3"
            run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
                 "-i", str(lst), "-c:a", "libmp3lame", "-b:a", "320k", str(reel)])
            print("[export] reel:", reel)

    with open(OUT / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)

    lines = ["# Voice outputs", "",
             "Generated by the `Clone Voice — Albedo (XTTS v2)` workflow.", "",
             "| language | emotion | seed | speaker-sim | ASR score | transcript | gen duration (s) |",
             "|---|---|---|---|---|---|---|"]
    for r in summary:
        asr = f"{r['asr_score']:.2f}" if r.get("asr_score") is not None else "–"
        tr = (r.get("transcript") or "").replace("|", "/")[:40]
        lines.append(f"| {r['lang']} | {r['emotion']} | {r['seed']} | "
                     f"{r['spk_sim']:.3f} | {asr} | {tr} | {r['gen_dur_s']} |")
    lines += ["",
              "Per emotion: `.wav` (16-bit 44.1 kHz), `.mp3` (320 kbps), "
              "`.png` (spectrogram). `*_demo_reel.mp3` plays all emotions in sequence."]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print("[export] done:", len(summary), "final clips")


if __name__ == "__main__":
    main()
