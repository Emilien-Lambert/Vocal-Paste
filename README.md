# Vocal Paste

Voice-to-clipboard for macOS. Press a key, speak, paste.

100% local — your voice never leaves your Mac. Powered by the [mlx-community/Voxtral-Mini-4B-Realtime-6bit](https://huggingface.co/mlx-community/Voxtral-Mini-4B-Realtime-6bit) model, running on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).

Perfect for driving AI agents (Claude Code, Codex, Gemini CLI...) with your voice — just speak and paste your prompt.

## Features

- **Multilingual Support:** Transcribe in 13 languages with high accuracy.
- **Real-Time Performance:** Low-latency transcription optimized for Apple Silicon.
- **Privacy First:** No cloud processing; everything happens on-device.
- **Easy Workflow:** Copy-to-clipboard or auto-paste (optional).

## Supported Languages

Vocal Paste supports **13 languages** for real-time speech transcription:

|                    |                     |                       |                    |
|:-------------------|:--------------------|:----------------------|:-------------------|
| 🇺🇸 **English**   | 🇫🇷 **French**     | 🇪🇸 **Spanish**      | 🇩🇪 **German**    |
| 🇮🇹 **Italian**   | 🇳🇱 **Dutch**      | 🇵🇹 **Portuguese**   | 🇷🇺 **Russian**   |
| 🇨🇳 **Chinese**   | 🇯🇵 **Japanese**   | 🇰🇷 **Korean**       | 🇮🇳 **Hindi**     |

## Requirements

- macOS (Apple Silicon)
- Python 3.10+
- [FFmpeg](https://formulae.brew.sh/formula/ffmpeg) (for file transcription): `brew install ffmpeg`

## Model Details

This project uses **Voxtral-Mini-4B-Realtime**, a model developed by **Mistral AI**. It combines a ~3.4B parameter Language Model with a ~970M parameter Audio Encoder to achieve state-of-the-art real-time Automatic Speech Recognition (ASR).

- **Official Model:** [mistralai/Voxtral-Mini-4B-Realtime-2602](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)
- **MLX Optimized Version:** [mlx-community/Voxtral-Mini-4B-Realtime-6bit](https://huggingface.co/mlx-community/Voxtral-Mini-4B-Realtime-6bit)
- **Framework:** [Apple MLX](https://github.com/ml-explore/mlx)

## How it works

### Live voice recording

1. Press **Right ⌘** — recording starts
2. Speak naturally — transcription runs in real-time
3. Release — text is copied to clipboard (and optionally auto-pasted)

### File transcription

Drop an audio file (`.wav`, `.flac`, `.ogg`, `.mp3`, `.m4a`, `.aac`, `.wma`) into the `audios/` folder and start the app. It will automatically transcribe the file and save the result to `transcript.txt` at the project root before entering live recording mode.

> **Note:** Do not start a voice recording while a file is being transcribed.

## Install

Create and activate a virtual environment to manage dependencies properly:

```bash
git clone https://github.com/Emilien-Lambert/Vocal-Paste.git
cd Vocal-Paste
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Hugging Face Setup

A token is required to download the model without rate limits:

1. Create one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Add it to your `.env`:

```
HF_TOKEN=your_token_here
```

On first run, the model (~3 GB) downloads automatically.

## Run

```bash
python main.py
```

Verbose mode (see transcription live):

```bash
python main.py -v
```

> **macOS permissions:** System Settings → Privacy & Security → grant **Accessibility** and **Microphone** access to your terminal.

## Configuration

Edit `.env`:

```
# Time in seconds before unloading the model from RAM
# Lower values free RAM faster, higher values keep the model ready for instant use
MODEL_TIMEOUT=30

# Hold key to record (true) or toggle on/off (false)
HOLD_TO_TALK=false

# Auto-paste after transcription (requires Accessibility permission)
AUTO_PASTE=false
```

## License

MIT
