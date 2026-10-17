#!/usr/bin/env python3
"""Build work/jobs.json for run_xtts.py from workflow inputs."""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emotions", default="neutral,happy,sad,angry,teasing,surprised")
    ap.add_argument("--seeds", default="1,2,3,4")
    ap.add_argument("--languages", default="ja,en")
    ap.add_argument("--text-ja", default="あらあら、さよなら。")
    ap.add_argument("--text-en", default="Ara ara... sayonara.")
    ap.add_argument("--out", default="work/jobs.json")
    args = ap.parse_args()

    texts = {"ja": args.text_ja, "en": args.text_en}
    emotions = [e.strip() for e in args.emotions.split(",") if e.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    langs = [l.strip() for l in args.languages.split(",") if l.strip()]

    jobs = []
    for lang in langs:
        for emo in emotions:
            jobs.append({"emotion": emo, "lang": lang, "seeds": seeds,
                         "text": texts.get(lang, "")})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(jobs, f, indent=1)
    print(f"wrote {args.out}: {len(jobs)} jobs "
          f"({len(emotions)} emotions × {len(seeds)} seeds × {len(langs)} languages)")


if __name__ == "__main__":
    main()
