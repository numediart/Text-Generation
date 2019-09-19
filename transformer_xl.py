import torch
import argparse
from pytorch_pretrained_bert import TransfoXLTokenizer, TransfoXLLMHeadModel
import logging

# Adapted from https://github.com/kimiyoung/transformer-xl/issues/49#issuecomment-472212730
# Copyright UMONS - 2019

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='transfo-xl-wt103',
                        help='pretrained model name or path to local checkpoint')
    parser.add_argument("--text", type=str, default="Traditional Chinese literary",
                        help='The sentence used to initiate generation.')
    parser.add_argument("--tokens_to_generate", type=int, default="40",
                        help='Number of tokens to generate after the end of the sentence.')
    parser.add_argument("--select_from_k", type=int, default="40",
                        help='From how many top tokens at each iteration, random selection will be made')
    args = parser.parse_args()

    transformer_xl_generation(args.model_name_or_path, args.text, args.tokens_to_generate, args.select_from_k)


def format_text(tokens):
    line = ''
    for token in tokens:
        if token == '<eos>':
            line += '\n'
        else:
            line += token
            line += ' '

    # simple rules of detokenization
    line = line.replace(' @-@ ', '-')
    line = line.replace(' @,@ ', ',')
    line = line.replace(' @.@ ', '.')
    line = line.replace(' . ', '. ')
    line = line.replace(' , ', ', ')
    line = line.replace(' : ', ': ')
    line = line.replace(' ; ', '; ')
    line = line.replace(" 's ", "'s ")
    line = line.replace(' ( ', ' (')
    line = line.replace(' ) ', ') ')

    return line


def transformer_xl_generation(model_name_or_path, text, tokens_to_generate, select_from_k):
    tokenizer = TransfoXLTokenizer.from_pretrained(model_name_or_path)
    sentences = text.split("\n")
    lines = [s.strip().split() + ['<eos>'] for s in sentences]

    idx = 0
    context = []
    while idx < len(lines):
        context += lines[idx]
        idx += 1

    ctx_tensor = torch.tensor([tokenizer.convert_tokens_to_ids(context)])

    model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')
    model.eval()

    ctx_tensor = ctx_tensor.to('cuda')
    model.to('cuda')

    unk_id = tokenizer.convert_tokens_to_ids(['<unk>'])[0]

    with torch.no_grad():
        tensor = ctx_tensor
        generation = []
        for i in range(tokens_to_generate):
            if i == 0:
                log_prob, mems = model(tensor)
            else:
                log_prob, mems = model(tensor, mems=mems)

            prob = torch.exp(log_prob[0, -1, :])
            prob[unk_id].data.fill_(0.)

            # sample from the top-k tokens
            top_prob, top_index = torch.topk(prob, select_from_k)
            token = torch.multinomial(top_prob, 1)
            token = top_index[token]

            tensor = token.detach().view(1, 1)

            symbol = tokenizer.get_sym(token.item())

            generation.append(symbol)

    print(format_text(generation))


if __name__ == '__main__':
    main()
