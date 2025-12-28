import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "allenai/Olmo-3-7B-Think"
device = "cuda:0"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=0,                 # ✅ 모델 전체를 GPU 0에
    torch_dtype=torch.float16,    # 보통 VRAM 절약/속도에 유리 (가능하면)
)
model.eval()

prompt = "간단히 자기소개 해줘."
inputs = tokenizer(prompt, return_tensors="pt").to(device)  # ✅ 입력도 GPU로

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        temperature=0.6,
        top_p=0.95,
        max_new_tokens=256,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
