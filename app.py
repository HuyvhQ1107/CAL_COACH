import base64
import io
import re

import torch
from flask import Flask, flash, redirect, render_template, request, url_for
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

app = Flask(__name__)
app.secret_key = "change-this-secret-key"  # cần cho flash()

### Step 1: Load model Qwen (chỉ load 1 lần khi khởi động)
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

GEN_PARAMS = {"max_new_tokens": 700, "do_sample": False}


def chat(prompt, image=None, system_prompt=None):
    """image: đường dẫn file ảnh hoặc đối tượng PIL.Image (có thể để None)."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    content = []
    images = []
    if image is not None:
        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB")
        image.thumbnail((640, 640))  # nhỏ hơn để chạy nhanh trên CPU/GPU yếu
        images.append(image)
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt})
    messages.append({"role": "user", "content": content})

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=images if images else None,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, **GEN_PARAMS)

    output = output[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(output, skip_special_tokens=True)[0]


### Step 3
def input_image_setup(uploaded_file):
    """Đọc file upload và mã hóa thành chuỗi base64."""
    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        return base64.b64encode(bytes_data).decode("utf-8")
    raise FileNotFoundError("No file uploaded")


### Step 4
def format_response(response_text):
    """Chuyển markdown đơn giản trong câu trả lời sang HTML."""
    response_text = re.sub(r"\*\*(.*?)\*\*", r"<p><strong>\1</strong></p>", response_text)
    response_text = re.sub(r"(?m)^\s*[\*\-]\s(.*)", r"<li>\1</li>", response_text)
    response_text = re.sub(
        r"(<li>.*?</li>)+",
        lambda match: f"<ul>{match.group(0)}</ul>",
        response_text,
        flags=re.DOTALL,
    )
    response_text = re.sub(r"</p>(?=<p>)", r"</p><br>", response_text)
    response_text = re.sub(r"(\n|\\n)+", r"<br>", response_text)
    return response_text


### Step 5
def generate_model_response(encoded_image, user_query, assistant_prompt):
    """Gửi ảnh + câu hỏi cho Qwen và trả về câu trả lời đã định dạng HTML."""
    try:
        # Giải mã base64 -> PIL.Image
        image = Image.open(io.BytesIO(base64.b64decode(encoded_image)))

        prompt = f"{assistant_prompt}\n\n{user_query or ''}".strip()
        raw_response = chat(prompt, image=image)

        return format_response(raw_response)
    except Exception as e:
        print(f"Error in generating response: {e}")
        return "<p>An error occurred while generating the response.</p>"


ASSISTANT_PROMPT = """
You are an expert nutritionist. Your task is to analyze the food items displayed in the image and provide a detailed nutritional assessment using the following format:

1. **Identification**: List each identified food item clearly, one per line.
2. **Portion Size & Calorie Estimation**: For each identified food item, specify the portion size and provide an estimated number of calories. Use bullet points with the following structure:
- **[Food Item]**: [Portion Size], [Number of Calories] calories

Example:
*   **Salmon**: 6 ounces, 210 calories
*   **Asparagus**: 3 spears, 25 calories

3. **Total Calories**: Provide the total number of calories for all food items.

Example:
Total Calories: [Number of Calories]

4. **Nutrient Breakdown**: Include a breakdown of key nutrients such as **Protein**, **Carbohydrates**, **Fats**, **Vitamins**, and **Minerals**. Use bullet points, and for each nutrient provide details about the contribution of each food item.

Example:
*   **Protein**: Salmon (35g), Asparagus (3g), Tomatoes (1g) = [Total Protein]

5. **Health Evaluation**: Evaluate the healthiness of the meal in one paragraph.

6. **Disclaimer**: Include the following exact text as a disclaimer:

The nutritional information and calorie estimates provided are approximate and are based on general food data.
Actual values may vary depending on factors such as portion size, specific ingredients, preparation methods, and individual variations.
For precise dietary advice or medical guidance, consult a qualified nutritionist or healthcare provider.

Format your response exactly like the template above to ensure consistency.
"""


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_query = request.form.get("user_query")
        uploaded_file = request.files.get("file")

        if uploaded_file and uploaded_file.filename:
            encoded_image = input_image_setup(uploaded_file)

            if not encoded_image:
                flash("Error processing the image. Please try again.", "danger")
                return redirect(url_for("index"))

            response = generate_model_response(encoded_image, user_query, ASSISTANT_PROMPT)
            return render_template("index.html", user_query=user_query, response=response)

        flash("Please upload an image file.", "danger")
        return redirect(url_for("index"))

    return render_template("index.html")


if __name__ == "__main__":
    # use_reloader=False: tránh Flask debug load model 2 lần (tốn gấp đôi VRAM)
    app.run(debug=True, use_reloader=False)