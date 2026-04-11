from datasets import load_dataset
import os

def prepare_arxiv():
    print("Downloading arXiv abstracts from Hugging Face...")
    # We use a common arxiv subset (e.g. from Curation or similar)
    # 'arxiv_dataset' is a good candidate, but let's try 'scientific_papers' with 'arxiv' config
    try:
        dataset = load_dataset("ccdv/arxiv-summarization", split="train", streaming=True)
    except:
        print("Falling back to another source...")
        dataset = load_dataset("He-Xing-Jian/arxiv-abstracts-2021", split="train", streaming=True)

    output_file = "arxiv.train.raw"
    count = 0
    max_samples = 100000
    
    print(f"Writing {max_samples} abstracts to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            # Depending on the dataset structure, it might be 'abstract' or 'summary'
            abstract = item.get('abstract', item.get('summary', ''))
            if abstract:
                # Clean a bit (remove newlines inside abstract)
                clean_abstract = abstract.replace("\n", " ").strip()
                f.write(clean_abstract + "\n")
                count += 1
            
            if count >= max_samples:
                break
            if count % 10000 == 0:
                print(f"Progress: {count}/{max_samples}")

    print(f"Finished! {output_file} is ready.")

if __name__ == "__main__":
    prepare_arxiv()
