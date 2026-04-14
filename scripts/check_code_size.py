from datasets import load_dataset
import time

print("Checking CodeSearchNet Python size...")
ds = load_dataset("code_search_net", "python", split="train", streaming=True)
count = 0
start = time.time()
for i, ex in enumerate(ds):
    code = ex.get('whole_func_code') or ex.get('func_code_string') or ex.get('code', "")
    count += len(code)
    if i >= 10000: break

elapsed = time.time() - start
print(f"10,000 functions = ~{count} characters (approx. {count // 4} tokens)")
print(f"Estimated time to load 200M tokens: {(200000000 / (count // 4)) * elapsed / 60:.1f} minutes")
