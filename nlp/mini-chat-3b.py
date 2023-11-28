import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from conversation import get_default_conv_template

tokenizer = AutoTokenizer.from_pretrained("GeneZC/MiniChat-3B", use_fast=False, legacy=False)
model = AutoModelForCausalLM.from_pretrained("GeneZC/MiniChat-3B", use_cache=True, device_map="auto",
                                             torch_dtype=torch.float16).eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
conv = get_default_conv_template("minichat")

def generate_response(question):
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)

    prompt = conv.get_prompt()
    input_ids = tokenizer([prompt]).input_ids

    output_ids = model.generate(
        torch.as_tensor(input_ids).to(device),
        do_sample=True,
        temperature=0.7,
        max_new_tokens=1024,
    )

    output_ids = output_ids[0][len(input_ids[0]):]
    output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    return output

question = "Do you like memes?"
response = generate_response(question)
print(response)
