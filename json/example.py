import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from json_template_generation import generate_with_json_template
from transformers import AutoModel, AutoTokenizer

# 1. Load model
model_path = "resource/dream_dia"  # local model dir (DIA code + Dream-v0-Base-7B weights); or your path / HF hub id
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).to("cuda").eval()

# 2. Prepare input
input_text = """
Please generate a JSON object to extract the following information:
Albert Einstein: Born March 14, 1879. Theoretical physicist.

Now give your answer:
"""
input_ids = tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=True).to("cuda")

# 3. Generate with DIA (supports per-block length control)
with torch.no_grad():
    output = generate_with_json_template(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        initial_tokens_per_field=[2, 3, 4],
        max_tokens_per_field=[8, 10, 12],
        steps_per_block=8,
        verbose=True
    )

# 4. Decode and print
result = tokenizer.decode(output[0, input_ids.shape[1]:].tolist())
print("Final Output:")
print(result)
