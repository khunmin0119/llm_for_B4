# this file is made for load olom(made in ai2) in huggingface transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
olmo = AutoModelForCausalLM.from_pretrained("allenai/Olmo-3-7B-Think")
tokenizer = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Think")
message = ["Who would win in a fight - a dinosaur or a cow named Moo Moo?"]
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
# optional verifying cuda
# inputs = {k: v.to('cuda') for k,v in inputs.items()}
# olmo = olmo.to('cuda')
response = olmo.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])
