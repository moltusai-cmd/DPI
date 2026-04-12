from tokenizers import ByteLevelBPETokenizer
import os

def train_tokenizer():
    paths = ["arxiv.train.raw"]
    
    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Customize training
    print("Starting BPE training on arXiv data...")
    tokenizer.train(files=paths, vocab_size=16384, min_frequency=2, show_progress=True, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask3>",
    ])

    # Save files to disk
    os.makedirs("bpe_tokenizer_arxiv", exist_ok=True)
    tokenizer.save_model("bpe_tokenizer_arxiv")
    print("BPE Tokenizer saved to bpe_tokenizer_arxiv/")

if __name__ == "__main__":
    train_tokenizer()
