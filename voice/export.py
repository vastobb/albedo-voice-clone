#!/usr/bin/env python3
"""Pick the best candidate per emotion+language, trim, shape, and export finals.

Selection (per emotion+language):
1. "valid" candidates = ASR pass (says the phrase, no rambling) AND sane
   duration (0.6–7 s). Best valid candidate by speaker similarity wins.
2. If none is valid, fall back to the highest ASR score, then similarity.
3. As a last resort, the highest similarity wins.

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
MIN_DUR, MAX_DUR = 0.6, 7.0


def run(cmd):
    subprocess.run(cmd, check=True)


def trim_silence(src: str, dst: str) -> None:
    """Trim leading/trailing silence, keeping 50 ms of padding."""
    chain = ("silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.05,"
             "areverse,"
             "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.05,"
             "areverse")
    run(["ffmpeg", "-y", "-v", "error", "-i", src, "-af", chain, dst])


def pick_best(cands):
    valid = [c for c in cands
             if c.get("asr_pass") and MIN_DUR <= c["dur_s"] <= MAX_DUR]
    if valid:
        return max(valid, key=lambda c: c["sim"]), "valid"
    sane = [c for c in cands if MIN_DUR <= c["dur_s"] <= MAX_DUR]
    pool = sane or cands
    nonloop = [c for c in pool if not c.get("loop")]
    pool = nonloop or pool
    return max(pool, key=lambda c: ((c.get("asr") or 0.0), c["sim"])), "fallback"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT))
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

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    summary = []
    for lang in langs:
        lang_dir = out_root / lang
        lang_dir.mkdir(parents=True, exist_ok=True)

        cands_by_emo = {}
        for g in gens:
            if g["lang"] != lang:
                continue
            base = os.path.basename(g["file"])
            r = info.get(base, {})
            cands_by_emo.setdefault(g["emotion"], []).append({
                "file": g["file"],
                "sim": r.get("spk_sim", 0.0),
                "asr": r.get("asr_score"),
                "asr_pass": bool(r.get("asr_pass", False)),
                "loop": bool(r.get("asr_loop", False)),
                "transcript": r.get("transcript", ""),
                "seed": g["seed"],
                "dur_s": g["dur_s"],
            })

        picks = {}
        for emo, cands in cands_by_emo.items():
            picks[emo], how = pick_best(cands)
            picks[emo]["how"] = how

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
                            "asr_pass": picks[emo]["asr_pass"],
                            "transcript": picks[emo]["transcript"],
                            "gen_dur_s": picks[emo]["dur_s"],
                            "selected_as": picks[emo]["how"]})
            print(f"[export] {lang}/{emo}: {os.path.basename(src)} "
                  f"[{picks[emo]['how']}] (sim={picks[emo]['sim']:.3f}, "
                  f"asr={picks[emo]['asr']}, '{picks[emo]['transcript'][:40]}')")

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

    with open(out_root / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)

    lines = ["# Voice outputs", "",
             "Generated by the `Clone Voice — Albedo (XTTS v2)` workflow.", "",
             "| language | emotion | seed | speaker-sim | ASR | passed | transcript | gen duration (s) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in summary:
        asr = f"{r['asr_score']:.2f}" if r.get("asr_score") is not None else "–"
        ok = "✓" if r.get("asr_pass") else ("~" if r.get("selected_as") == "fallback" else "✗")
        tr = (r.get("transcript") or "").replace("|", "/")[:40]
        lines.append(f"| {r['lang']} | {r['emotion']} | {r['seed']} | "
                     f"{r['spk_sim']:.3f} | {asr} | {ok} | {tr} | {r['gen_dur_s']} |")
    lines += ["",
              "Per emotion: `.wav` (16-bit 44.1 kHz), `.mp3` (320 kbps), "
              "`.png` (spectrogram). `*_demo_reel.mp3` plays all emotions in sequence.",
              "", "✓ = ASR-verified take, ~ = best available (no take fully passed ASR)."]
    (out_root / "README.md").write_text("\n".join(lines) + "\n")
    print("[export] done:", len(summary), "final clips")


if __name__ == "__main__":
    main()
