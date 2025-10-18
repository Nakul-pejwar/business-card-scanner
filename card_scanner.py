import cv2
import pytesseract
import pandas as pd
import re
from datetime import datetime
import numpy as np
import os
from pathlib import Path
import logging
import threading
from collections import defaultdict

# ============================================================================
# LOGGING SETUP - Better debugging & error tracking
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('card_scanner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION - Centralized settings
# ============================================================================
CONFIG = {
    'excel_file': 'contact_cards.xlsx',
    'camera_id': 0,
    'frame_width': 1280,
    'frame_height': 720,
    'font_scale': 0.6,
    'font_thickness': 1,
    'capture_delay_ms': 1500,
    'retry_delay_ms': 1000,
    'min_text_length': 5,
}

# Improved regex patterns
PATTERNS = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
    'phone': r'(\+?[\d]{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
    'website': r'https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?',
    'address': r'\d+\s+[A-Za-z\s]+(?:St|Ave|Rd|Blvd|Dr|Lane)',
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def init_excel(filepath):
    """Initialize Excel file with proper structure."""
    try:
        if os.path.exists(filepath):
            df = pd.read_excel(filepath)
            logger.info(f"Loaded existing Excel: {len(df)} contacts")
        else:
            columns = ['Timestamp', 'Name', 'Company', 'Phone', 'Email', 'Address', 'Website', 'Confidence', 'Raw_Text']
            df = pd.DataFrame(columns=columns)
            df.to_excel(filepath, index=False)
            logger.info(f"Created new Excel file: {filepath}")
        return df
    except Exception as e:
        logger.error(f"Excel init failed: {e}")
        raise

def extract_text_from_image(image):
    """Extract text with preprocessing for better OCR."""
    try:
        # Preprocessing: Grayscale + contrast enhancement
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Threshold for cleaner text
        _, binary = cv2.threshold(enhanced, 150, 255, cv2.THRESH_BINARY)
        
        # OCR with better config
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(binary, lang='eng', config=custom_config)
        
        logger.debug(f"OCR extracted {len(text)} characters")
        return text.strip()
    except Exception as e:
        logger.error(f"Text extraction failed: {e}")
        return ""

def extract_field(raw_text, pattern_key):
    """Safely extract field using regex."""
    try:
        matches = re.findall(PATTERNS[pattern_key], raw_text, re.IGNORECASE)
        return matches[0] if matches else ''
    except Exception as e:
        logger.warning(f"Pattern extraction failed for {pattern_key}: {e}")
        return ''

def calculate_confidence(parsed_data):
    """Calculate extraction confidence (0-100)."""
    confidence = 0
    max_score = 0
    
    field_weights = {'Name': 20, 'Email': 25, 'Phone': 25, 'Company': 20, 'Website': 10}
    
    for field, weight in field_weights.items():
        max_score += weight
        if parsed_data.get(field):
            confidence += weight
    
    return int((confidence / max_score) * 100) if max_score > 0 else 0

def parse_contact_info(raw_text):
    """Smart parsing with improved heuristics."""
    if len(raw_text) < CONFIG['min_text_length']:
        return None
    
    info = {
        'Name': '',
        'Company': '',
        'Phone': '',
        'Email': '',
        'Address': '',
        'Website': '',
        'Raw_Text': raw_text
    }
    
    try:
        # Extract structured fields
        info['Email'] = extract_field(raw_text, 'email')
        info['Phone'] = extract_field(raw_text, 'phone')
        info['Website'] = extract_field(raw_text, 'website')
        info['Address'] = extract_field(raw_text, 'address')
        
        # Parse name & company from lines
        lines = [line.strip() for line in raw_text.split('\n') if line.strip() and len(line.strip()) > 2]
        
        if lines:
            # Name: Usually first line, short, mixed case
            first_line = lines[0]
            if not re.search(PATTERNS['email'], first_line) and not re.search(PATTERNS['phone'], first_line):
                info['Name'] = first_line if len(first_line) < 50 else 'Unknown'
            
            # Company: Look for keywords or longer lines
            for line in lines[1:6]:
                upper_line = line.upper()
                if any(keyword in upper_line for keyword in ['INC', 'LLC', 'CORP', 'CO', 'LTD', 'GROUP']):
                    info['Company'] = line
                    break
                elif len(line.split()) >= 2 and not re.search(PATTERNS['email'], line):
                    info['Company'] = line
                    break
        
        logger.info(f"Parsed: {info['Name']} | {info['Email']}")
        return info
    except Exception as e:
        logger.error(f"Parsing failed: {e}")
        return None

def is_duplicate(df, new_contact):
    """Check for duplicate contacts."""
    if df.empty or not new_contact['Email']:
        return False
    
    existing_emails = df['Email'].str.lower().values
    return new_contact['Email'].lower() in existing_emails

def draw_ui(frame, status, message, color):
    """Draw modern UI overlay."""
    overlay = frame.copy()
    h, w = frame.shape[:2]
    
    # Header bar
    cv2.rectangle(overlay, (0, 0), (w, 60), color, -1)
    cv2.putText(overlay, status, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Message bar
    cv2.rectangle(overlay, (0, h-50), (w, h), (40, 40, 40), -1)
    cv2.putText(overlay, message, (15, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 1)
    
    # Blend overlay
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Instruction text
    cv2.putText(frame, "SPACE: Capture | Q: Quit | S: Settings", (15, 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return frame

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    logger.info("=== Business Card Scanner Started ===")
    
    # Initialize
    df = init_excel(CONFIG['excel_file'])
    cap = cv2.VideoCapture(CONFIG['camera_id'])
    
    if not cap.isOpened():
        logger.error("Camera failed to open!")
        print("❌ Camera error: Check connection or permissions")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['frame_width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['frame_height'])
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    logger.info("Camera initialized successfully")
    print("\n✅ Camera Ready! | 📸 Point at card → SPACE to capture | Q to quit\n")
    
    capture_count = 0
    duplicate_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame capture failed")
                break
            
            # Draw live UI
            display_frame = draw_ui(frame, "🎥 LIVE", 
                                   f"Contacts: {len(df)} | Duplicates skipped: {duplicate_count}", 
                                   (25, 25, 112))
            
            cv2.imshow('📇 Business Card Scanner', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # SPACEBAR - CAPTURE
                logger.info("Capture triggered")
                raw_text = extract_text_from_image(frame)
                
                if raw_text:
                    parsed = parse_contact_info(raw_text)
                    
                    if parsed:
                        if is_duplicate(df, parsed):
                            duplicate_count += 1
                            logger.warning(f"Duplicate detected: {parsed['Email']}")
                            
                            # Red flash for duplicate
                            dup_frame = frame.copy()
                            dup_frame = draw_ui(dup_frame, "⚠️  DUPLICATE", 
                                               f"Email already exists: {parsed['Email']}", 
                                               (0, 0, 255))
                            cv2.imshow('Duplicate Alert', dup_frame)
                            cv2.waitKey(CONFIG['retry_delay_ms'])
                        else:
                            # Success - add to dataframe
                            confidence = calculate_confidence(parsed)
                            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            
                            new_row = pd.DataFrame({
                                'Timestamp': [timestamp],
                                'Name': [parsed['Name']],
                                'Company': [parsed['Company']],
                                'Phone': [parsed['Phone']],
                                'Email': [parsed['Email']],
                                'Address': [parsed['Address']],
                                'Website': [parsed['Website']],
                                'Confidence': [confidence],
                                'Raw_Text': [parsed['Raw_Text']]
                            })
                            
                            df = pd.concat([df, new_row], ignore_index=True)
                            capture_count += 1
                            
                            logger.info(f"Contact added: {parsed['Name']} ({confidence}% confidence)")
                            
                            # Green flash for success
                            success_frame = frame.copy()
                            success_frame = draw_ui(success_frame, "✅ SUCCESS", 
                                                   f"{parsed['Name']} @ {parsed['Company']} | Confidence: {confidence}%", 
                                                   (0, 200, 0))
                            cv2.imshow('Extraction Success', success_frame)
                            cv2.waitKey(CONFIG['capture_delay_ms'])
                    else:
                        logger.warning("Parsing returned None")
                        error_frame = draw_ui(frame, "❌ PARSE ERROR", "Could not parse contact info", (0, 0, 255))
                        cv2.imshow('Parse Error', error_frame)
                        cv2.waitKey(CONFIG['retry_delay_ms'])
                else:
                    logger.warning("No text detected in image")
                    no_text_frame = draw_ui(frame, "❌ NO TEXT", 
                                          "No text found. Better lighting? Check card angle.", 
                                          (0, 165, 255))
                    cv2.imshow('No Text Detected', no_text_frame)
                    cv2.waitKey(CONFIG['retry_delay_ms'])
            
            elif key == ord('q'):  # QUIT
                logger.info("Quit triggered by user")
                break
    
    except Exception as e:
        logger.error(f"Runtime error: {e}")
        print(f"❌ Error: {e}")
    
    finally:
        # Save & cleanup
        try:
            df.to_excel(CONFIG['excel_file'], index=False)
            logger.info(f"Saved {len(df)} total contacts to {CONFIG['excel_file']}")
        except Exception as e:
            logger.error(f"Save failed: {e}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n{'='*50}")
        print(f"✅ Session Complete!")
        print(f"📊 Total Contacts: {len(df)}")
        print(f"📸 New Captures: {capture_count}")
        print(f"⚠️  Duplicates Skipped: {duplicate_count}")
        print(f"💾 Saved to: {CONFIG['excel_file']}")
        print(f"📋 Log file: card_scanner.log")
        print(f"{'='*50}\n")

if __name__ == "__main__":
    main()