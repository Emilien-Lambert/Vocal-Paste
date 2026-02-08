# Vocal Paste

Voice-to-clipboard for macOS. Press a key, speak, paste.

100% local — your voice never leaves your Mac. Powered by [Voxtral](https://huggingface.co/mlx-community/Voxtral-Mini-4B-Realtime-6bit) running on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).

Perfect for driving AI agents (Claude Code, Codex, Gemini CLI...) with your voice — just speak and paste your prompt.

## How it works

1. Press **Right ⌘** — recording starts
2. Speak naturally — transcription runs in real-time
3. Release — text is copied to clipboard (and optionally auto-pasted)

That's it.

## Install

```bash
git clone https://github.com/YOUR_USERNAME/Vocal-Paste.git
cd Vocal-Paste
pip install -r requirements.txt
cp .env.example .env
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
MODEL_TIMEOUT=300

# Hold key to record (true) or toggle on/off (false)
HOLD_TO_TALK=false

# Auto-paste after transcription (requires Accessibility permission)
AUTO_PASTE=false
```

## Requirements

- macOS (Apple Silicon)
- Python 3.10+

## License

MIT
