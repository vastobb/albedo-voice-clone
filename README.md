# Albedo Voice Clone — local AI on GitHub Actions

Zero-shot **voice cloning** of the reference clip in [`ref/albedo-1.mp3`](ref/albedo-1.mp3)
(Albedo, *Overlord*), speaking **"ara ara sayonara"** (あらあら、さよなら) in six
emotions — **neutral, happy, sad, angry, teasing, surprised** — plus a romaji
(English) rendering of the same phrase.

Everything runs **inside this repository's GitHub Actions runner** — no external
AI services, no cloud TTS APIs. The model (XTTS v2) is downloaded onto the
runner and runs locally there, on CPU.

## ▶️ How to run (the button)

1. Go to the **Actions** tab of this repo.
2. Select **"Clone Voice — Albedo (XTTS v2)"** in the left sidebar.
3. Click **"Run workflow"** → **"Run workflow"**.
4. Wait ~15–25 min (first run is slower; the XTTS model is cached afterwards).
5. Get the results from either:
   - the **run's Artifacts** (`albedo-voice-runN`), or
   - the **`outputs` branch** (always the latest results — force-pushed each run).

You can adjust text, emotions, seeds, temperature, etc. in the Run-workflow form.

## 🔊 Outputs

For every language (`ja` = authentic Japanese, `en` = romaji reading):

| file | what |
|---|---|
| `albedo_ara_sayonara_<emotion>.wav` | 16-bit 44.1 kHz |
| `albedo_ara_sayonara_<emotion>.mp3` | 320 kbps |
| `albedo_ara_sayonara_<emotion>.png` | spectrogram |
| `albedo_ara_sayonara_demo_reel.mp3` | all emotions in sequence |
| `summary.json` / `README.md` | seeds + speaker-similarity scores |

## 🧠 Pipeline (all local, all on the runner)

1. **Reference prep** (`voice/prep_refs.py`) — decode the mp3, mono 44.1 kHz,
   gentle `afftdn` denoise + limiter → `ref_clean`. Then build
   *emotion-morphed* conditioning references with **librubberband**
   (formant-preserving pitch/tempo + EQ), e.g. a lifted, brighter ref for
   *happy*, a lowered, slower, darker ref for *sad*.
2. **Zero-shot cloning** (`voice/run_xtts.py`) — **XTTS v2** (Coqui) on CPU
   conditions on each ref variant and generates multiple seeded candidates per
   emotion/language (`temperature`, `speed`, repetition penalty per emotion).
3. **Ranking** (`voice/evaluate.py`) — a local **Whisper** ASR check scores
   every take against the phrase (character similarity + key-word check) and
   *passes* only takes that contain "さよなら"/"sayonara" with sane length
   (ramble/loop rejection). Among passing takes with sane duration (0.6–7 s),
   **ECAPA** speaker similarity picks the winner; otherwise the best ASR score
   is used as fallback.
4. **Emotion shaping** (`voice/emotionize.py`) — second stage DSP per emotion:
   formant-preserving pitch/tempo shifts, shelf EQ, compressor/soft-clip
   (*angry*), loudness normalisation (EBU R128).
5. **Export** (`voice/export.py`) — silence trimming (50 ms padding),
   WAV/MP3/spectrograms + demo reels + `summary.json`, committed to the
   `outputs` branch and uploaded as artifacts.

## 📊 Reference clip analysis (done locally with Python/NumPy + ffmpeg)

- 4.96 s, 44.1 kHz stereo, 256 kbps MP3 (cut from a "music+vocals" source).
- Voice-dominant; envelope periodicity test shows **no strong musical beat**
  (max autocorr ≈ 0.16) and non-stationary spectral lines → speech, not music.
- Bright, breathy timbre: ~42% of energy in 1–4 kHz, airy up to 16 kHz,
  almost no bass (<1% below 150 Hz).
- Pitch (autocorrelation): median ≈ 414 Hz, p5–p95 ≈ 180–500 Hz — a high,
  expressive anime register.

## ⚙️ Requirements & notes

- Runs on `ubuntu-latest` (CPU only). XTTS v2 model (~1.9 GB) is cached
  between runs via `actions/cache`; the ECAPA model (~80 MB) is fetched once
  per run. Optionally set a repo secret `HF_TOKEN` if Hugging Face rate-limits.
- **Personal / creative use.** The reference audio is a short user-provided
  clip of a fictional character; Albedo © KADOKAWA / *Overlord*. XTTS v2 is
  released under the Coqui Public Model License (non-commercial).
