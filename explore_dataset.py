from datasets import load_dataset
from pprint import pprint

dataset = load_dataset("hotpot_qa", "fullwiki", split="validation", trust_remote_code=True)

# Take the top n samples
samples = dataset.select(range(5))

for sample in samples:
    pprint(sample)

