from fastapi import FastAPI, File, UploadFile
from typing import List, Dict, Any, Tuple
from PIL import Image
import numpy as np
import re
import uvicorn
import io
import torch
import time

# For CRAFT text detection (install craft-text-detector)
from craft_text_detector import Craft

# For TrOCR (install transformers)
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

app = FastAPI()

# CRAFT text detector - GPU if available, else CPU (very slow on CPU bro)
print("Loading CRAFT model, please wait yaar...")
craft_detector = Craft(output_dir=None, crop_type="poly", cuda=torch.cuda.is_available())

# TrOCR model - using base, but you can put your custom one here
try:
    print("Loading TrOCR model and processor, thoda time lagega...")
    TROCR_MODEL_PATH = "microsoft/trocr-base-printed"  # Change to your custom model if you trained one!
    trocr_processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_PATH)
    trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_PATH)
    trocr_model.eval()
    if torch.cuda.is_available():
        trocr_model = trocr_model.cuda()
    print("TrOCR loaded successfully!")
except Exception as e:
    print("Error loading TrOCR:", e)
    raise

def read_image_from_bytes(image_bytes: bytes) -> Image.Image:
    # Reads image from bytes and returns PIL Image (RGB)
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        print("Error reading image:", e)
        raise
    return img

def extract_text_blocks(img: Image.Image) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
    # Use CRAFT to detect text regions and crop them out
    # Returns list of (block_img, (x_min, y_min, x_max, y_max))
    print("Detecting text blocks using CRAFT...")
    craft_result = craft_detector.detect_text(np.array(img))
    boxes = craft_result["boxes"]  # List of [x_min, y_min, x_max, y_max]
    block_imgs = []
    for i, box in enumerate(boxes):
        # Sometimes CRAFT gives float coordinates, so convert to int
        x_min, y_min, x_max, y_max = [int(v) for v in box]
        # Make sure coordinates are valid
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img.width, x_max)
        y_max = min(img.height, y_max)
        if x_max > x_min and y_max > y_min:
            cropped = img.crop((x_min, y_min, x_max, y_max))
            block_imgs.append((cropped, (x_min, y_min, x_max, y_max)))
        else:
            print(f"Skipping invalid box {i}: {box}")
    print(f"Total blocks detected: {len(block_imgs)}")
    return block_imgs

def ocr_text_from_block(block_img: Image.Image) -> str:
    # Use TrOCR to recognize text from a cropped block image
    try:
        pixel_values = trocr_processor(block_img, return_tensors="pt").pixel_values
        if torch.cuda.is_available():
            pixel_values = pixel_values.cuda()
        with torch.no_grad():
            generated_ids = trocr_model.generate(pixel_values)
            text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()
    except Exception as e:
        print("Error in TrOCR OCR:", e)
        return ""

def combine_blocks_to_text(blocks: List[Tuple[Image.Image, Tuple[int, int, int, int]]]) -> str:
    # Sort blocks top to bottom, left to right (not always perfect but works mostly)
    print("Sorting blocks for logical reading order...")
    blocks_sorted = sorted(blocks, key=lambda b: (b[1][1], b[1][0]))
    full_text = ""
    for i, (block_img, bbox) in enumerate(blocks_sorted):
        print(f"OCR on block {i+1}/{len(blocks_sorted)} at {bbox}...")
        text = ocr_text_from_block(block_img)
        if text:
            full_text += text + "\n"
        else:
            print(f"Block {i+1} OCR gave empty result.")
    print("All blocks OCR done.")
    return full_text

def parse_reference_range(ref_range_str: str) -> Tuple[Any, Any]:
    # Parse reference range string and return (low, high) as floats if possible.
    # Handles cases like '12.0-15.0', '0-6', '19 - 45', '<11.3 mg/dL', '>10', etc.
    # Returns (low, high) or (None, None) if not parseable.
    ref_range_str = ref_range_str.replace('to', '-').replace('–', '-').replace(' ', '')
    # Remove common units
    for unit in ['mg/dl', 'g/dl', 'U/L', 'million/cu.mm', 'fl', '%', 'ml/min', 'IU/L', 'uIU/ml']:
        ref_range_str = ref_range_str.replace(unit, '')
    # Try range
    match = re.match(r'<?\s*([\d.]+)\s*-\s*>?\s*([\d.]+)', ref_range_str)
    if match:
        try:
            return float(match.group(1)), float(match.group(2))
        except:
            return (None, None)
    # Handle '<' or '>' only
    match = re.match(r'<\s*([\d.]+)', ref_range_str)
    if match:
        try:
            return (None, float(match.group(1)))
        except:
            return (None, None)
    match = re.match(r'>\s*([\d.]+)', ref_range_str)
    if match:
        try:
            return (float(match.group(1)), None)
        except:
            return (None, None)
    return (None, None)

def is_out_of_range(test_value: str, ref_range: Tuple[Any, Any]) -> bool:
    # Given a value and reference range tuple (low, high), determine if out of range.
    # Handles open-ended ranges too.
    try:
        value = float(re.sub(r"[^\d.]+", "", test_value))
    except Exception:
        return False  # If not a number, cannot determine
    low, high = ref_range
    if low is not None and value < low:
        return True
    if high is not None and value > high:
        return True
    return False

def extract_lab_tests_from_text(text: str) -> List[Dict[str, Any]]:
    # Extracts test name, value, unit, reference range from OCR text.
    # Handles multi-line, tabular, and free-form text.
    print("Extracting lab tests from text...")
    results = []
    # Regex for lines like: Test Name  Value  Unit  Reference Range
    # Example: "Urea 28 mg/dl 19-45"
    line_regex = re.compile(
        r'([A-Za-z0-9\(\)\-\[\]\/\s\.,%]+?)\s+([<>]?\d+\.?\d*)\s*([a-zA-Z\/%\.]*)\s*([<>\d\.\-\s]+)?'
    )
    lines = text.split('\n')
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line or len(line) < 4:
            continue
        # Try regex
        match = line_regex.match(line)
        if match:
            test_name = match.group(1).strip(" .:-")
            test_value = match.group(2).strip()
            test_unit = match.group(3).strip()
            bio_reference_range = match.group(4).strip() if match.group(4) else ""
            bio_reference_range = bio_reference_range.replace('mg/dl', '').replace('g/dl', '').replace('U/L', '').replace('million/cu.mm', '').replace('fl', '').replace('%', '').strip()
            ref_range_tuple = parse_reference_range(bio_reference_range)
            out_of_range = is_out_of_range(test_value, ref_range_tuple)
            if test_name and test_value:
                results.append({
                    "test_name": test_name.upper(),
                    "test_value": test_value,
                    "bio_reference_range": bio_reference_range,
                    "test_unit": test_unit,
                    "lab_test_out_of_range": out_of_range
                })
        else:
            # Try qualitative regex (NEGATIVE/POSITIVE etc.)
            qual_regex = re.compile(r'([A-Za-z0-9\(\)\-\[\]\/\s\.,%]+?)\s+(NEGATIVE|POSITIVE|REACTIVE|NONREACTIVE|PRESENT|ABSENT)\b', re.IGNORECASE)
            match2 = qual_regex.match(line)
            if match2:
                test_name = match2.group(1).strip(" .:-")
                test_value = match2.group(2).strip().upper()
                results.append({
                    "test_name": test_name.upper(),
                    "test_value": test_value,
                    "bio_reference_range": "",
                    "test_unit": "",
                    "lab_test_out_of_range": False
                })
            else:
                # Try splitting by tabs or multiple spaces
                parts = re.split(r'\s{2,}|\t', line)
                if len(parts) >= 3:
                    # Try to guess which part is what
                    test_name = parts[0].strip()
                    test_value = parts[1].strip()
                    test_unit = parts[2].strip() if len(parts) > 2 else ""
                    bio_reference_range = parts[3].strip() if len(parts) > 3 else ""
                    ref_range_tuple = parse_reference_range(bio_reference_range)
                    out_of_range = is_out_of_range(test_value, ref_range_tuple)
                    if test_name and test_value:
                        results.append({
                            "test_name": test_name.upper(),
                            "test_value": test_value,
                            "bio_reference_range": bio_reference_range,
                            "test_unit": test_unit,
                            "lab_test_out_of_range": out_of_range
                        })
    print(f"Extracted {len(results)} tests from text.")
    return results

@app.post("/get-lab-tests")
async def get_lab_tests(file: UploadFile = File(...)):
    t0 = time.time()
    try:
        # Step 1: Read and decode image
        print("Reading uploaded image...")
        image_bytes = await file.read()
        image = read_image_from_bytes(image_bytes)
        print("Image read successfully.")

        # Step 2: Segment image into text blocks (CRAFT)
        blocks = extract_text_blocks(image)

        # Step 3: OCR each block (TrOCR)
        full_text = combine_blocks_to_text(blocks)
        print("Full OCR text:\n", full_text)

        # Step 4: Extract lab tests from text
        lab_tests = extract_lab_tests_from_text(full_text)

        t1 = time.time()
        print(f"Total time taken: {t1-t0:.2f} seconds")

        return {
            "is_success": True,
            "data": lab_tests
        }
    except Exception as e:
        print("Exception in API:", e)
        return {
            "is_success": False,
            "error": str(e),
            "data": []
        }

if __name__ == "__main__":
    print("Starting FastAPI app on localhost:8080 ...")
    uvicorn.run(app, host="0.0.0.0", port=8080)

