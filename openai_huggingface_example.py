#!/usr/bin/env python3
# Adapted from https://github.com/huggingface/pytorch-transformers/tree/master/examples#language-generation
# Copyright 2019 - UMONS

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You should have received a copy of the Apache License Version 2.0 along with this program.
# If not, see
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
from tqdm import trange

import torch
import torch.nn.functional as F
import numpy as np
import random as rd

from pytorch_pretrained_bert import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def top_k_logits(logits, k):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)


def sample_sequence(model, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0,
                    device='cuda', sample=True):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
    prev = context
    output = context
    past = None
    with torch.no_grad():
        for i in trange(length):
            logits = model(prev)
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                word = torch.multinomial(log_probs, num_samples=1)
                prev = torch.cat((prev, word), dim=1)

            else:
                _, word = torch.topk(log_probs, k=1, dim=-1)
                prev = torch.cat((prev, word), dim=1)
            output = torch.cat((output, word), dim=1)
    return output


def run_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='openai-gpt',
                        help='pretrained model name or path to local checkpoint')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--length", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')
    args = parser.parse_args()

    if args.batch_size == -1:
        args.batch_size = 1
    assert args.nsamples % args.batch_size == 0

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = OpenAIGPTTokenizer.from_pretrained(args.model_name_or_path)
    model = OpenAIGPTLMHeadModel.from_pretrained(args.model_name_or_path)
    model.to(device)
    model.eval()

    if args.length == -1:
        args.length = model.config.n_ctx // 2
    elif args.length > model.config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % model.config.n_ctx)

    while True:
        context_tokens = []
        if not args.unconditional:
            raw_text = input("Model prompt >>> ")
            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input("Model prompt >>> ")
            context_tokens = enc.encode(raw_text)
            generated = 0
            for _ in range(args.nsamples // args.batch_size):
                out = sample_sequence(
                    model=model, length=args.length,
                    context=context_tokens,
                    start_token=None,
                    batch_size=args.batch_size,
                    temperature=args.temperature, top_k=args.top_k, device=device
                )
                out = out[:, len(context_tokens):].tolist()
                for i in range(args.batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                    print(text)
            print("=" * 80)
        else:
            generated = 0
            start_token = rd.choice(list(enc.encoder.keys()))
            print("Using start word :" + start_token)
            for _ in range(args.nsamples // args.batch_size):
                out = sample_sequence(
                    model=model, length=args.length,
                    context=None,
                    start_token=enc.encoder[start_token],
                    batch_size=args.batch_size,
                    temperature=args.temperature, top_k=args.top_k, device=device
                )
                out = out[:, 1:].tolist()
                for i in range(args.batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                    print(text)
            print("=" * 80)


if __name__ == '__main__':
    run_model()
