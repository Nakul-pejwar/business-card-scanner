import cv2  # OpenCV: Your camera's best friend for live feeds and snaps
import pytesseract  # OCR engine—turns pixels into words
import pandas as pd  # Pandas: The data ninja for Excel wrangling
import re  # Regex: Pattern hunter for emails/phones (import it, it's built-in magic)
from datetime import datetime  # Timestamps, 'cause we love order
import numpy as np  # NumPy: Quick math for image tweaks (also built-in)

# Your Excel file—grows with each scan
excel_file = 'contact_cards.xlsx'

# Regex patterns for parsing (explained below—new? Think of 'em as treasure maps for text)
email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Grabs emails like john@company.com
phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'  # Phones: 123-456-7890 or +1 (555) 1234
website_pattern = r'\b(https?://)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/[a-zA-Z0-9.-]*)*'  # Sites: company.com or full URLs

def extract_text_from_image(image):
    """Grab raw text from the image—grayscale for better OCR reads."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # BGR to gray: Strips color, sharpens text edges
    text = pytesseract.image_to_string(gray, lang='eng')  # OCR blast—'eng' for English cards
    return text.strip()  # Trim whitespace, no fluffy extras

def parse_contact_info(raw_text):
    """Smart parse: Hunt for fields in the messy text. Returns a dict of goodies."""
    info = {
        'Name': '',
        'Company': '',
        'Phone': '',
        'Email': '',
        'Address': '',
        'Website': '',
        'Raw_Text': raw_text
    }
    
    # Emails & Phones: Easy peasy with regex—findall grabs all matches
    emails = re.findall(email_pattern, raw_text)
    info['Email'] = emails[0] if emails else ''  # First one's usually the main
    
    phones = re.findall(phone_pattern, raw_text)
    info['Phone'] = phones[0] if phones else ''  # Same for phone
    
    websites = re.findall(website_pattern, raw_text)
    info['Website'] = websites[0] if websites else ''
    
    # Name & Company: Hacky but works—split lines, grab likely suspects
    lines = [line.strip() for line in raw_text.split('\n') if line.strip()]  # Clean lines
    if lines:
        # Assume line 1-ish is name (capitals, short), line 2-3 company (longer words)
        info['Name'] = lines[0] if lines[0].isupper() or len(lines[0].split()) <= 3 else 'Unknown'
        for line in lines[1:5]:  # Check next few lines for company vibes (all caps or "Inc./LLC")
            if any(word in line.upper() for word in ['INC', 'LLC', 'CORP', 'CO']) or line.isupper():
                info['Company'] = line
                break
            elif len(line.split()) >= 2 and not re.search(email_pattern, line) and not re.search(phone_pattern, line):
                info['Company'] = line  # Fallback: Non-email/phone line
    
    # Address: Anything with commas/numbers that ain't phone/email
    addr_candidates = [line for line in lines if ',' in line and not re.search(email_pattern | phone_pattern, line)]
    info['Address'] = addr_candidates[0] if addr_candidates else ''
    
    return info

# Setup Excel—create if new, load if exists
try:
    df = pd.read_excel(excel_file)
except FileNotFoundError:
    df = pd.DataFrame(columns=['Timestamp', 'Name', 'Company', 'Phone', 'Email', 'Address', 'Website', 'Raw_Text'])
    df.to_excel(excel_file, index=False)

# Webcam launch (0 = built-in cam; swap to 1 for USB)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Window size—cozy, not fullscreen chaos
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Camera's primed, legend! Point at a card, mash SPACEBAR to snap & extract. 'Q' to wrap and save. Let's hoard those contacts. 📸")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cam hiccup—plug it back in or yell at your laptop.")
        break
    
    # Show live preview
    display_frame = frame.copy()
    cv2.putText(display_frame, "SPACE to Capture | Q to Quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Card Scanner Cam', display_frame)
    
    key = cv2.waitKey(1) & 0xFF  # Grab key press—non-blocking, so feed stays smooth
    
    if key == ord(' '):  # Spacebar = SNAP! (Like old Polaroids, but digital)
        print("Click captured—OCR grinding...")
        raw_text = extract_text_from_image(frame)
        
        if raw_text:
            parsed = parse_contact_info(raw_text)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # New row for Excel
            new_row = pd.DataFrame({
                'Timestamp': [timestamp],
                'Name': [parsed['Name']],
                'Company': [parsed['Company']],
                'Phone': [parsed['Phone']],
                'Email': [parsed['Email']],
                'Address': [parsed['Address']],
                'Website': [parsed['Website']],
                'Raw_Text': [parsed['Raw_Text']]
            })
            
            df = pd.concat([df, new_row], ignore_index=True)
            
            # Flash success: Green tint on captured frame
            green_frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)  # Jet map for that glow—new? It's a color overlay trick
            cv2.putText(green_frame, f"Extracted: {parsed['Name']} @ {parsed['Company']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow('Success Snap!', green_frame)
            cv2.waitKey(1500)  # Pause 1.5 secs to admire
            
            print(f"Parsed: Name={parsed['Name']}, Company={parsed['Company']}, Phone={parsed['Phone']}, Email={parsed['Email']}")
        else:
            # No text? Red flash for "try again"
            red_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            red_frame[:, :] = [0, 100, 255]  # HSV red overlay—feels urgent
            cv2.imshow('No Text? Retry!', red_frame)
            cv2.waitKey(1000)
            print("Ghost card? Angle better or light it up.")
    
    elif key == ord('q'):
        break

# Cleanup: Save & shut down
df.to_excel(excel_file, index=False)
cap.release()
cv2.destroyAllWindows()
print(f"Session over, champ! {len(df)} contacts in '{excel_file}'. Open it up—sort by company for that power move. If parsing missed stuff (like funky fonts), snap a sample and we'll regex-tweak it next round.")
