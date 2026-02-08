import subprocess
import src.config as config

def log(message: str, verbose_only: bool = False) -> None:
    """Standardized logging function."""
    if not verbose_only or config.VERBOSE:
        print(message)

def paste_text_to_system() -> None:
    """Simulates Cmd+V to paste content on macOS."""
    if not config.AUTO_PASTE:
        return
    try:
        script = 'tell application "System Events" to keystroke "v" using {command down}'
        subprocess.run(['osascript', '-e', script], check=False)
    except Exception as e:
        log(f"ERROR: Failed to auto-paste: {e}")
