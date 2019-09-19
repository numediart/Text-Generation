# Copyright UMONS – 2019
import torch
import argparse
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='openai-gpt',
                        help='pretrained model name or path to local checkpoint')
    parser.add_argument("--text", type=str, default="This is the story of a little dog named Boo.",
                        help='The sentence used to initiate generation.')
    parser.add_argument("--tokens_to_generate", type=int, default="30",
                        help='Number of tokens to generate after the end of the sentence.')
    args = parser.parse_args()

    openai_generation(args.model_name_or_path, args.text, args.tokens_to_generate)


def openai_generation(model_name_or_path, text, tokens_to_generate):
    tokenizer = OpenAIGPTTokenizer.from_pretrained(model_name_or_path)
    lm_model = OpenAIGPTLMHeadModel.from_pretrained(model_name_or_path)
    lm_model.eval()

    #  Prepare tokenized input
    tokenized_text = tokenizer.tokenize(text)
    while tokens_to_generate > 0:
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])

        # Predict all tokens
        with torch.no_grad():
            predictions = lm_model(tokens_tensor)

        # Get the last predicted token
        predicted_index = torch.argmax(predictions[0, -1, :]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
        tokenized_text.append(predicted_token)
        tokens_to_generate -= 1

    print("".join(tokenized_text).replace("</w>", " "))


if __name__ == '__main__':
    main()
