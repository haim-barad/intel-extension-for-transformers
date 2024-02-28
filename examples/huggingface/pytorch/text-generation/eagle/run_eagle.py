import time
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.ea_model import EaModel
from fastchat.model import get_conversation_template


"""
CPU = EmeraldRapids:
    base_model_path = "meta-llama/Llama-2-7b-chat-hf"
    ea_model_path = "yuhuili/EAGLE-llama2-chat-7B"
    With EAGLE:
        Total generated tokens: 6210
        Total time: 753.7117660045624
        TPS: 8.23922390507356
    Without EAGLE:
        Total generated tokens: 5807
        Total time: 1045.285539150238
        TPS: 5.555419818321398
"""

#######
# EAGLE
#######
base_model_path = "meta-llama/Llama-2-7b-chat-hf"
ea_model_path = "yuhuili/EAGLE-llama2-chat-7B"
use_eagle = True 
use_vicuna = False
use_llama_2_chat = True
max_new_tokens = 128
device = "cpu"
if device == "cpu":
    datatype = torch.float32
    cpu_only = True
elif device == "xpu":
    import intel_extension_for_pytorch as ipex
    datatype = torch.float16
    cpu_only = False
else:
    datatype = torch.float32
    cpu_only = False

if use_eagle:
    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        torch_dtype=datatype,
        # low_cpu_mem_usage=True,
        # device_map=device
        ).eval().to(device)
else:
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=base_model_path,
        torch_dtype=datatype,
        # low_cpu_mem_usage=True,
        # device_map=device
        ).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    # model = ipex.optimize_transformers(model, dtype=datatype, device=device, inplace=True)

your_message="Hello, can you give a story about a man and a dragon?"
if use_llama_2_chat:
    conv = get_conversation_template("llama-2-chat") 
    sys_p = ""
    conv.system_message = sys_p
    conv.append_message(conv.roles[0], your_message)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt() + " "

if use_vicuna:
    conv = get_conversation_template("vicuna")
    conv.append_message(conv.roles[0], your_message)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()


t_total = 0
total_new_tokens = 0
with torch.inference_mode():
    # Go through a number of iterations so make the statistics more accurate
    num_iterations = 10
    temperature = 0.5
    for _ in tqdm(range(num_iterations)):
        t_s = time.time()
        if use_eagle:
            input_ids = model.tokenizer([prompt], return_tensors="pt").input_ids.to(device)
            output_ids = model.eagenerate(
                input_ids,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                cpu_only=cpu_only,
                )
            output = model.tokenizer.decode(output_ids[0])
        else:
            input_ids = tokenizer([prompt], return_tensors="pt").input_ids.to(device)
            output_ids = model.generate(
                input_ids,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                )
            output = tokenizer.decode(output_ids[0])
        t_total += time.time() - t_s
        total_new_tokens += output_ids.shape[-1] - input_ids.shape[-1]
    

print(f"Total generated tokens: {total_new_tokens}")
print(f"Total time: {t_total}")
print(f"TPS: {total_new_tokens / t_total}")
print(output)
