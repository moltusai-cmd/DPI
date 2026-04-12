import torch
import torch.nn as nn
from model import PID8Transformer
import re
from collections import Counter

def build_vocab(file_path, vocab_size=16384, max_lines=50000):
    counter = Counter()
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_lines: break
            tokens = re.findall(r"[\w']+|[.,!?;=]|@-@", line.lower())
            counter.update(tokens)
    
    most_common = counter.most_common(vocab_size - 2)
    vocab = {word: i + 2 for i, (word, _) in enumerate(most_common)}
    vocab['<pad>'] = 0
    vocab['<unk>'] = 1
    return vocab

def generate(model, vocab, inv_vocab, prompt, max_len=50, device='cpu', temperature=0.7):
    model.eval()
    tokens = [vocab.get(t, vocab['<unk>']) for t in re.findall(r"[\w']+|[.,!?;=]|@-@", prompt.lower())]
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    
    generated = tokens
    with torch.no_grad():
        for _ in range(max_len):
            logits = model(input_ids[:, -128:]) # Context window
            next_token_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            generated.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
            
            if next_token == vocab.get('.', -1): # Stop at period for brevity
                break
                
    return " ".join([inv_vocab.get(t, '<unk>') for t in generated])

def run_comparison():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_file = "wiki.train.raw"
    vocab = build_vocab(train_file)
    inv_vocab = {i: t for t, i in vocab.items()}
    
    # Load PID-8.1 Model
    model_pid = PID8Transformer(vocab_size=16384).to(device)
    model_pid.load_state_dict(torch.load("pid8_fertile_4epochs.pt", map_location=device))
    
    # Load Xavier Model
    model_xavier = PID8Transformer(vocab_size=16384).to(device)
    model_xavier.load_state_dict(torch.load("xavier_4epochs.pt", map_location=device))
    
    prompts = [
        "the game of",
        "the imperial unit",
        "valkyria chronicles is a",
        "the story follows"
    ]
    
    print("=== Generation Comparison ===\n")
    for p in prompts:
        print(f"Prompt: '{p}'")
        out_pid = generate(model_pid, vocab, inv_vocab, p, device=device)
        out_xavier = generate(model_xavier, vocab, inv_vocab, p, device=device)
        
        print(f"  [PID-8.1 (Loss 0.63)]: {out_pid}")
        print(f"  [Xavier  (Loss 5.54)]: {out_xavier}")
        print("-" * 30)

if __name__ == "__main__":
    run_comparison()
