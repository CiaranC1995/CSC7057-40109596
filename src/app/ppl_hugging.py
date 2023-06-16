from evaluate import load

# Reading in text to analyse from file
with open(r'C:\Users\ccase\Desktop\Dissertation\Examples of Text\Human\Human_HarryPotter.txt') as f:
    text_to_analyse = f.read()

perplexity = load("perplexity", module_type="metric")
results = perplexity.compute(predictions=text_to_analyse, model_id='gpt2')

print(results)

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
from tqdm import tqdm

model_id = "gpt2"
tokenizer_gpt = GPT2TokenizerFast.from_pretrained('gpt2')
model_gpt = GPT2LMHeadModel.from_pretrained('gpt2')


def calculate_ppl(input_text, model, tokenizer):
    device = "cpu"

    inputs = tokenizer.encode(input_text, add_special_tokens=True, return_tensors='pt')

    max_length = model.config.n_positions
    stride = 512
    seq_len = inputs.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = inputs[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_ppl = torch.exp(torch.stack(nlls).mean())
    return avg_ppl


print(calculate_ppl(text_to_analyse, model_gpt, tokenizer_gpt))