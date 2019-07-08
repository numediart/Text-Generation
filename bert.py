import torch
import argparse
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='bert-base-uncased',
                        help='pretrained model name or path to local checkpoint')
    parser.add_argument("--text", type=str, default="This is the story of a little dog named Boo.",
                        help='The sentence to use for word suggestion')
    parser.add_argument("--mask", type=str, default="dog",
                        help='The word to mask for suggestion')
    args = parser.parse_args()

    bert_generation(args.model_name_or_path, args.text, args.mask)


def bert_generation(model_name_or_path, text, mask):

    tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
    model = BertForMaskedLM.from_pretrained(model_name_or_path)

    tokenized_text = tokenizer.tokenize(text)

    # Mask a token that we will try to predict back with `BertForMaskedLM`
    try:
        masked_index = tokenized_text.index(mask)

    except ValueError:
        print("Error : Masked word doesn't appear in sentence.")
        return -1

    tokenized_text[masked_index] = '[MASK]'

    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    segments_ids = [0] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    model.eval()

    # Predict all tokens
    predictions = model(tokens_tensor, segments_tensors)
    predicted_index = torch.argmax(predictions[0, masked_index]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])

    print("Original:", text)
    print("Masked:", " ".join(tokenized_text))

    print("Predicted token:", predicted_token)
    print("Other options:")
    # just curious about what the next few options look like.
    for i in range(10):
        predictions[0, masked_index, predicted_index] = -11100000
        predicted_index = torch.argmax(predictions[0, masked_index]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
        print(predicted_token)


if __name__ == '__main__':
    main()
