import pickle
from typing import Dict, List

import gradio as gr
import torch

from models.TextRCNN import Model, Config

UNK, PAD = "<UNK>", "<PAD>"

dataset = "NewDataset"
model_name = "TextRCNN"
model = None
vocab = None
categories = [
    # "财经", "房产", "股票", "教育", "科技", "社会", "时政", "体育", "游戏", "娱乐"
    "Attractive",
    "One-dimensional",
    "Must-be",
    "Indifference",
]
id_to_category = {i: c for i, c in enumerate(categories)}


def tokenize(
    sentence: str,
    vocab: Dict[str, float],
    use_word: bool = False,
    padding_size: int = 32,
) -> List[int]:
    sentence = sentence.strip()

    if use_word:
        tokens = sentence.split(" ")
    else:
        tokens = [c for c in sentence]

    if len(tokens) < padding_size:
        tokens += [PAD for _ in range(padding_size - len(tokens))]
    else:
        tokens = tokens[:padding_size]

    return [vocab.get(t, vocab.get(UNK)) for t in tokens]


def classify(sentence: str) -> Dict[str, float]:
    config = Config(
        dataset=dataset,
        embedding="random",
    )

    global model, vocab
    if vocab is None:
        with open(config.vocab_path, "rb") as f:
            vocab = pickle.load(f)
        config.n_vocab = len(vocab)

    if model is None:
        model = Model(config)
        model.load_state_dict(
            torch.load(f"{dataset}/saved_dict/{model_name}.ckpt")
        )
        model.eval()

    tokens = tokenize(sentence, vocab, use_word=False)
    tokens = torch.as_tensor(tokens)[None]

    logits = model((tokens, None)).squeeze()
    probs = logits.softmax(0).tolist()

    res = {}
    for i, p in enumerate(probs):
        res[categories[i]] = p

    return res


def main():
    demo = gr.Interface(
        fn=classify,
        inputs="text",
        outputs="label",
        allow_flagging=False,
        examples=[
        ]
    )

    demo.launch(share=False)


if __name__ == "__main__":
    main()
