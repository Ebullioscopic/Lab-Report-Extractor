# Lab Report Extraction API

## API Usage

- **Endpoint:**  
  `POST https://1hl9rpqb-8004.inc1.devtunnels.ms/get-lab-tests`
- **Parameter:**  
  `file` - The lab report image file (PNG, JPG, etc.)

---

## Example Request

```
curl -X POST \
  -F "file=@/path/to/lab_report.jpg" \
  https://1hl9rpqb-8004.inc1.devtunnels.ms/get-lab-tests
```

## Overview

This project provides an end-to-end solution for extracting structured lab test data from lab report images. It is built as a FastAPI service with a single POST endpoint:

```
POST https://1hl9rpqb-8004.inc1.devtunnels.ms/get-lab-tests
```

- **Input:** An image file (sent with the `file` parameter)
- **Output:** Structured JSON with all detected lab test names, values, units, reference ranges, and a flag indicating if the result is out of the normal range.

---

## Intermediate image

![Intermediate Image](/static/assets/intermediate_image.png)

---

## Pipeline Architecture

1. **Image Upload:** User uploads a lab report image to the API.
2. **Text Detection (CRAFT):** The image is processed using the CRAFT model to detect and segment regions containing text.
3. **Text Recognition (TrOCR):** Each detected text region is passed through a fine-tuned TrOCR model to recognize the text.
4. **Lab Test Extraction:** The recognized text is parsed using robust regular expressions and logic to extract test names, values, units, and reference ranges.
5. **Post-Processing:** Each test is checked against its reference range to flag out-of-range results.
6. **Structured Output:** Results are returned as structured JSON.

---

## Key Components

### CRAFT (Text Detection)

- **Role:** Detects and localizes text regions in the input image, even for complex layouts (tables, columns, free text).
- **How it works:** CRAFT (Character Region Awareness for Text detection) generates heatmaps and bounding boxes for text areas, enabling robust detection of arbitrarily oriented and curved text[6][7][10][11].
- **Why:** Segmentation ensures that each text region is processed cleanly by the OCR model, improving accuracy.

**Code Example:**
```
# Detect text regions using CRAFT
blocks = extract_text_blocks(image)  # Returns list of (block_image, block_bbox)
```

### TrOCR (Text Recognition)

- **Role:** Recognizes (reads) the text within each detected region.
- **How it works:** TrOCR is a transformer-based OCR model using a vision transformer encoder and a language transformer decoder. It is fine-tuned on lab report data for maximum accuracy[8][9][12].
- **Why:** Outperforms traditional OCR, especially for varied fonts, layouts, and handwritten text.

**Training Pseudocode:**
```
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Load pre-trained model and processor
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')

# Training loop (simplified)
for batch in train_dataloader:
    images, texts = batch
    pixel_values = processor(images=images, return_tensors="pt").pixel_values
    labels = processor.tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True).input_ids
    outputs = model(pixel_values=pixel_values, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
```

### Lab Test Extraction Logic

- **Role:** Converts raw OCR text into structured fields: test name, value, unit, reference range, and out-of-range flag.
- **How:** Uses regular expressions and rule-based parsing to extract and normalize information, handling variations in format, missing fields, and multi-line entries.

---

## Example Response

```
{
    "is_success": true,
    "data": [
        {
            "test_name": "UREA",
            "test_value": "28",
            "bio_reference_range": "19-45",
            "test_unit": "mg/dl",
            "lab_test_out_of_range": false
        },
        {
            "test_name": "CREATININE",
            "test_value": "1.04",
            "bio_reference_range": "0.67-1.17",
            "test_unit": "mg/dl",
            "lab_test_out_of_range": false
        }
    ]
}
```