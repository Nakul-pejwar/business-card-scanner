# Business Card Scanner 🚀

Yo, tired of typing contacts like it's 1999? This Python beast uses your webcam to snap cards, OCR-extracts the deets (name, company, phone, email, etc.), and dumps 'em into a tidy Excel sheet. Built for hoarders like us—scan a stack, profit.

## Quick Start (Clone & Roll)
1. Clone this repo: `git clone https://github.com/yourusername/business-card-scanner.git`
2. Hop in: `cd business-card-scanner`
3. If you’re using VS Code with a virtual environment:
python -m venv venv
source venv/Scripts/activate  # if using Git Bash
4. Install deps: `pip install -r requirements.txt` (New? Deps = dependencies, the libraries we lean on.)
5. Tesseract setup: Download from [here](https://github.com/tesseract-ocr/tesseract) (free OCR engine). On Windows/Mac, installer does it; Linux: `sudo apt install tesseract-ocr` (sudo = superuser, god-mode for installs—careful, it's like handing keys to a toddler).
5. Run: `python card_scanner.py` – Webcam pops, spacebar to snap, 'q' to save.

## What It Does
- Live cam viewfinder (feels like your phone's camera app).
- Parses raw text with regex magic (emails/phones auto-hunted; name/company via line heuristics—80% win rate on clean cards).
- Outputs to `contact_cards.xlsx`: Columns for Timestamp, Name, Company, Phone, Email, Address, Website, Raw_Text.

## Tweaks & Troubleshooting
- Blurry OCR? Better lighting or add contrast in code (line ~45: `cv2.convertScaleAbs` boost).
- Batch mode? Ping me—we'll loop over image folders.
- Errors? Check console—common: Tesseract path (add `pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'` for Win).

## License
MIT—fork it, break it, love it.

Made with ❤️ and zero sleep. Contributions? PRs welcome—let's make it scan minds next. 😆pip 