import sys
import io
# Windows 콘솔 인코딩 문제 해결 (최상단에 배치)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "allenai/Olmo-3-7B-Think"

# GPU가 진짜 있는지 체크 (GPU-only 하려면 필수)
assert torch.cuda.is_available(), "CUDA 사용 불가입니다. (GPU/드라이버/torch CUDA 설치 확인 필요)"

device = "cuda:0"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=0,            # ✅ GPU 0에 전부 올림 (여기서 accelerate 필요)
    dtype=torch.float16,     # ✅ torch_dtype 대신 dtype
)
model.eval()

prompt = "왜 머리가 터지는 거 같아?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)   # ✅ 입력도 GPU로

# 입력 프롬프트 길이 저장 (나중에 출력에서 제거하기 위해)
input_length = inputs['input_ids'].shape[1]

with torch.inference_mode():
    out = model.generate(
        **inputs,
        max_new_tokens=50,           # 답변 길이 제한 (너무 길게 생성 방지)
        do_sample=True,              # 샘플링 활성화
        temperature=0.3,             # 낮은 temperature로 더 일관된 답변
        top_p=0.9,                   # nucleus sampling
        repetition_penalty=1.2,      # 반복 방지 강화
        eos_token_id=tokenizer.eos_token_id,  # EOS 토큰 명시
        pad_token_id=tokenizer.eos_token_id,  # 패딩 토큰 설정
    )

# 출력에서 입력 프롬프트 제거하고 새로 생성된 부분만 디코딩
generated_tokens = out[0][input_length:]
response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

# 응답 정리: 첫 번째 완전한 문장만 추출 (불필요한 연속 질문 방지)
# 마침표, 느낌표, 물음표로 끝나는 첫 번째 문장만 추출
import re
# 문장 구분자로 분리
sentences = re.split(r'([.!?]\s+)', response)
if sentences:
    # 첫 번째 문장 + 구분자
    first_sentence = sentences[0] + (sentences[1] if len(sentences) > 1 else '')
    response_cleaned = first_sentence.strip()
else:
    response_cleaned = response

print(f"입력: {prompt}")
print(f"\n응답: {response_cleaned}")
