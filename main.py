from fastapi import FastAPI, File, UploadFile
from typing import List, Dict, Any
import re
import uvicorn

app = FastAPI()

def extract_text_blocks(image):
    # Returns list of (block_image, block_bbox) for further OCR
    pass

def ocr_text_from_block(block_image):
    # Returns OCR text for the given image block
    pass

def parse_reference_range(ref_range_str):
    """
    Parse reference range string and return (low, high) as floats if possible.
    Handles cases like '12.0-15.0', '0-6', '19 - 45', '<11.3 mg/dL', '>10', etc.
    Returns (low, high) or (None, None) if not parseable.
    """
    ref_range_str = ref_range_str.replace('to', '-').replace('–', '-').replace('-', '-')
    match = re.match(r'<?\s*([\d.]+)\s*-\s*>?\s*([\d.]+)', ref_range_str)
    if match:
        return float(match.group(1)), float(match.group(2))
    # Handle '<' or '>' only
    match = re.match(r'<\s*([\d.]+)', ref_range_str)
    if match:
        return (None, float(match.group(1)))
    match = re.match(r'>\s*([\d.]+)', ref_range_str)
    if match:
        return (float(match.group(1)), None)
    return (None, None)

def is_out_of_range(test_value, ref_range):
    """
    Given a value and reference range tuple (low, high), determine if out of range.
    Handles open-ended ranges.
    """
    try:
        value = float(test_value)
    except Exception:
        return False  # If not a number, cannot determine

    low, high = ref_range
    if low is not None and value < low:
        return True
    if high is not None and value > high:
        return True
    return False

def extract_lab_tests_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extracts test name, value, unit, reference range from OCR text.
    Handles multi-line, tabular, and free-form text.
    """
    results = []
    # Common regex for lab test lines: Test Name  Value  Unit  Reference Range
    # Example: "Urea 28 mg/dl 19-45"
    # Handles possible extra spaces, tabs, or missing units/ranges
    line_regex = re.compile(
        r'([A-Za-z0-9\(\)\-\[\]\/\s\.,%]+?)\s+([<>]?\d+\.?\d*)\s*([a-zA-Z\/%\.]*)\s*([<>\d\.\-\s]+)?'
    )
    for line in text.split('\n'):
        line = line.strip()
        if not line or len(line) < 4:
            continue
        match = line_regex.match(line)
        if match:
            test_name = match.group(1).strip(" .:-")
            test_value = match.group(2).strip()
            test_unit = match.group(3).strip()
            bio_reference_range = match.group(4).strip() if match.group(4) else ""
            # Clean up reference range
            bio_reference_range = bio_reference_range.replace('mg/dl', '').replace('g/dl', '').replace('U/L', '').strip()
            ref_range_tuple = parse_reference_range(bio_reference_range)
            out_of_range = is_out_of_range(test_value, ref_range_tuple)
            # Only add if test_name and test_value are present
            if test_name and test_value:
                results.append({
                    "test_name": test_name.upper(),
                    "test_value": test_value,
                    "bio_reference_range": bio_reference_range,
                    "test_unit": test_unit,
                    "lab_test_out_of_range": out_of_range
                })
    return results

@app.post("/get-lab-tests")
async def get_lab_tests(file: UploadFile = File(...)):
    try:
        # Read image file
        image_bytes = await file.read()
        # You may need to decode image_bytes to an image object (e.g., using PIL or OpenCV)
        # image = decode_image(image_bytes)
        image = image_bytes  # Placeholder

        # Step 1: Segment image into text blocks (CRAFT)
        blocks = extract_text_blocks(image)

        # Step 2: OCR each block (TrOCR)
        full_text = ""
        for block_image, _ in blocks:
            block_text = ocr_text_from_block(block_image)
            full_text += block_text + "\n"

        # Step 3: Extract lab tests from text
        lab_tests = extract_lab_tests_from_text(full_text)

        return {
            "is_success": True,
            "data": lab_tests
        }
    except Exception as e:
        return {
            "is_success": False,
            "error": str(e),
            "data": []
        }

# For local testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
