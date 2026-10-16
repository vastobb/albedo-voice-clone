#!/usr/bin/env python3
"""Generate candidate voice clips with XTTS v2 — fully local, CPU-only.

Usage: run_xtts.py --jobs jobs.json
jobs.json: [{"emotion": "happy", "lang": "ja", "seeds": [1, 2],
             "text": "optional override", "ref": "optional ref override",
             "sound_norm": false, "tag": "optional filename tag"}, ...]

The conditioning reference is work/refs/ref_<emotion>.wav
(falls back to ref_clean.wav when no morphed ref exists, e.g. 'neutral').
Outputs 24 kHz wavs into work/raw/ plus run_meta.json.
"""
import argparse
import json
import os
import resource
import time

import numpy as np
import soundfile as sf
import torch

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJ, "models", "xtts-v2")
RAW_DIR = os.path.join(PROJ, "work", "raw")
REFS_DIR = os.path.join(PROJ, "work", "refs")

LANG_TEXT = {
    "ja": "あらあら、さよなら。",
    "en": "Ara ara... sayonara.",
}

EMO_SPEED = {
    "neutral": 1.0, "happy": 1.06, "sad": 0.9,
    "angry": 1.05, "teasing": 0.95, "surprised": 1.1,
}


def load_model():
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    config = XttsConfig()
    config.load_json(os.path.join(MODEL_DIR, "config.json"))
    model = Xtts.init_from_config(config)
    try:
        model.load_checkpoint(config, checkpoint_dir=MODEL_DIR, use_deepspeed=False, eval=True)
    except TypeError:
        model.load_checkpoint(
            config,
            checkpoint_path=os.path.join(MODEL_DIR, "model.pth"),
            vocab_path=os.path.join(MODEL_DIR, "vocab.json"),
            speaker_file_path=os.path.join(MODEL_DIR, "speakers_xtts.pth"),
            use_deepspeed=False,
            eval=True,
        )
    model.eval()
    if hasattr(model, "to"):
        model.to("cpu")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--temperature", type=float, default=0.75)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.85)
    ap.add_argument("--repetition-penalty", type=float, default=5.0)
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 4)
    os.makedirs(RAW_DIR, exist_ok=True)

    t0 = time.time()
    model = load_model()
    load_s = time.time() - t0
    print(f"[run_xtts] model loaded in {load_s:.1f}s", flush=True)
    meta = {"load_s": round(load_s, 1), "gens": []}

    for job in json.load(open(args.jobs)):
        emo = job["emotion"]
        lang = job["lang"]
        ref = os.path.join(REFS_DIR, f"ref_{emo}.wav")
        if not os.path.exists(ref):
            ref = os.path.join(REFS_DIR, "ref_clean.wav")
        if job.get("ref"):
            ref = os.path.join(REFS_DIR, job["ref"])
        text = job.get("text") or LANG_TEXT.get(lang, "")

        t1 = time.time()
        try:
            gpt_cond, spk_emb = model.get_conditioning_latents(
                audio_path=[ref], sound_norm_refs=bool(job.get("sound_norm", False))
            )
        except TypeError:
            gpt_cond, spk_emb = model.get_conditioning_latents(audio_path=[ref])
        cond_s = time.time() - t1

        for seed in job["seeds"]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            t2 = time.time()
            out = model.inference(
                text=text,
                language=lang,
                gpt_cond_latent=gpt_cond,
                speaker_embedding=spk_emb,
                temperature=args.temperature,
                length_penalty=1.0,
                repetition_penalty=args.repetition_penalty,
                top_k=args.top_k,
                top_p=args.top_p,
                speed=EMO_SPEED.get(emo, 1.0),
                enable_text_splitting=False,
            )
            wav = np.asarray(out["wav"], dtype=np.float32)
            tag = job.get("tag")
            fn = os.path.join(RAW_DIR, f"{emo}_{lang}_s{seed}" + (f"_{tag}" if tag else "") + ".wav")
            sf.write(fn, wav, 24000)
            g = {
                "emotion": emo, "lang": lang, "seed": seed,
                "ref": os.path.basename(ref), "text": text,
                "speed": EMO_SPEED.get(emo, 1.0),
                "dur_s": round(len(wav) / 24000, 3),
                "gen_s": round(time.time() - t2, 1),
                "cond_s": round(cond_s, 1),
                "file": fn,
            }
            meta["gens"].append(g)
            print(json.dumps(g), flush=True)

    meta["maxrss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    meta["total_s"] = round(time.time() - t0, 1)
    with open(os.path.join(RAW_DIR, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("META", json.dumps({k: v for k, v in meta.items() if k != "gens"}), flush=True)


if __name__ == "__main__":
    main()
