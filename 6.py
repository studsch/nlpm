import argparse
import re
import time
from collections import Counter

import nltk
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return nltk.tokenize.word_tokenize(text)


def text_to_indices(text, word_to_idx):
    tokens = tokenize(text)
    indices = [word_to_idx.get(token, word_to_idx["<UNK>"]) for token in tokens]
    return indices


class TranscriptDataset(Dataset):
    def __init__(self, pairs, word_to_idx, max_len=50):
        self.pairs = pairs
        self.word_to_idx = word_to_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        input_text, target_text = self.pairs[index]
        input_indices = (
            [self.word_to_idx["<SOS>"]]
            + text_to_indices(input_text, self.word_to_idx)
            + [self.word_to_idx["<EOS>"]]
        )
        target_indices = (
            [self.word_to_idx["<SOS>"]]
            + text_to_indices(target_text, self.word_to_idx)
            + [self.word_to_idx["<EOS>"]]
        )

        input_padded = input_indices + [self.word_to_idx["<PAD>"]] * (
            self.max_len - len(input_indices)
        )
        target_padded = target_indices + [self.word_to_idx["<PAD>"]] * (
            self.max_len - len(target_indices)
        )

        return torch.tensor(input_padded[: self.max_len]), torch.tensor(
            target_padded[: self.max_len]
        )


def create_encoder(vocab_size, embed_size, hidden_size):
    embedding = torch.nn.Embedding(vocab_size, embed_size)
    lstm = torch.nn.LSTM(embed_size, hidden_size, batch_first=True)
    return embedding, lstm


def encoder_forward(embedding, lstm, x):
    embedded = embedding(x)
    _, (hidden, cell) = lstm(embedded)
    return hidden, cell


def create_decoder(vocab_size, embed_size, hidden_size):
    embedding = torch.nn.Embedding(vocab_size, embed_size)
    lstm = torch.nn.LSTM(embed_size, hidden_size, batch_first=True)
    fc = torch.nn.Linear(hidden_size, vocab_size)
    return embedding, lstm, fc


def decoder_forward(embedding, lstm, fc, x, hidden, cell):
    embedded = embedding(x)
    output, (hidden, cell) = lstm(embedded, (hidden, cell))
    output = fc(output)
    return output, hidden, cell


def seq2seq_forward(
    encoder_embedding,
    encoder_lstm,
    decoder_embedding,
    decoder_lstm,
    decoder_fc,
    src,
    trg,
):
    hidden, cell = encoder_forward(encoder_embedding, encoder_lstm, src)
    batch_size = trg.size(0)
    trg_len = trg.size(1)
    vocab_size = decoder_fc.out_features
    outputs = torch.zeros(batch_size, trg_len, vocab_size).to(src.device)

    input = trg[:, 0].unsqueeze(1)
    for t in range(1, trg_len):
        output, hidden, cell = decoder_forward(
            decoder_embedding, decoder_lstm, decoder_fc, input, hidden, cell
        )
        outputs[:, t] = output.squeeze(1)
        input = trg[:, t].unsqueeze(1)
    return outputs


def train(pairs, epochs, device):
    all_words = []
    for i, t in pairs:
        all_words.extend(tokenize(i))
        all_words.extend(tokenize(t))
    word_counts = Counter(all_words)
    vocab = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"] + [
        word for word, count in word_counts.items() if count > 1
    ]
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    vocab_size = len(word_to_idx)

    dataset = TranscriptDataset(pairs, word_to_idx)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    embed_size = 256
    hidden_size = 512

    encoder_embedding, encoder_lstm = create_encoder(
        vocab_size, embed_size, hidden_size
    )
    decoder_embedding, decoder_lstm, decoder_fc = create_decoder(
        vocab_size, embed_size, hidden_size
    )

    encoder_embedding.to(device)
    encoder_lstm.to(device)
    decoder_embedding.to(device)
    decoder_lstm.to(device)
    decoder_fc.to(device)

    params = (
        list(encoder_embedding.parameters())
        + list(encoder_lstm.parameters())
        + list(decoder_embedding.parameters())
        + list(decoder_lstm.parameters())
        + list(decoder_fc.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=word_to_idx["<PAD>"])

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        for src, trg in dataloader:
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            output = seq2seq_forward(
                encoder_embedding,
                encoder_lstm,
                decoder_embedding,
                decoder_lstm,
                decoder_fc,
                src,
                trg,
            )
            output = output[:, 1:].reshape(-1, output.shape[2])
            trg = trg[:, 1:].reshape(-1)
            loss = criterion(output, trg)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        elapsed_timed = time.time() - start_time
        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}, Time: {elapsed_timed:.2f}s"
        )

    model_state = {
        "encoder_embedding": encoder_embedding.state_dict(),
        "encoder_lstm": encoder_lstm.state_dict(),
        "decoder_embedding": decoder_embedding.state_dict(),
        "decoder_lstm": decoder_lstm.state_dict(),
        "decoder_fc": decoder_fc.state_dict(),
        "word_to_idx": word_to_idx,
        "idx_to_word": idx_to_word,
        "embed_size": embed_size,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
    }
    torch.save(model_state, "chatbot_model.pth")


def generate_response(
    encoder_embedding,
    encoder_lstm,
    decoder_embedding,
    decoder_lstm,
    decoder_fc,
    sentence,
    word_to_idx,
    idx_to_word,
    max_len,
    device,
):
    with torch.no_grad():
        indices = (
            [word_to_idx["<SOS>"]]
            + text_to_indices(sentence.lower(), word_to_idx)
            + [word_to_idx["<EOS>"]]
        )
        src = torch.tensor(
            [indices + [word_to_idx["<PAD>"]] * (max_len - len(indices))]
        )[:max_len].to(device)
        hidden, cell = encoder_forward(encoder_embedding, encoder_lstm, src)

        input = torch.tensor([[word_to_idx["<SOS>"]]]).to(device)
        generated = []
        for _ in range(max_len):
            output, hidden, cell = decoder_forward(
                decoder_embedding, decoder_lstm, decoder_fc, input, hidden, cell
            )
            pred = output.argmax(2).item()
            if pred == word_to_idx["<EOS>"]:
                break
            generated.append(idx_to_word.get(pred, "<UNK>"))
            input = torch.tensor([[pred]]).to(device)
        return " ".join(generated)


def main():
    parser = argparse.ArgumentParser(description="Chat-bot")
    parser.add_argument("--train", action="store_true", help="train model")
    parser.add_argument("--epochs", type=int, default=10, help="epochs for train model")
    args = parser.parse_args()
    nltk.download("punkt", quiet=True)
    device = torch.device(
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )

    df = pd.read_csv("data/The-Office-Lines-V4.csv")
    transcripts = df["line"].tolist()
    pairs_len = min(5000, len(transcripts) - 1)

    pairs = []
    for i in range(pairs_len):
        input_text = transcripts[i].strip().lower()
        target_text = transcripts[i + 1].strip().lower()
        pairs.append((input_text, target_text))

    if args.train:
        train(pairs, args.epochs, device)

    model_state = torch.load("chatbot_model.pth", map_location=device)
    word_to_idx = model_state["word_to_idx"]
    idx_to_word = model_state["idx_to_word"]
    embed_size = model_state["embed_size"]
    hidden_size = model_state["hidden_size"]
    vocab_size = model_state["vocab_size"]

    encoder_embedding, encoder_lstm = create_encoder(
        vocab_size, embed_size, hidden_size
    )
    decoder_embedding, decoder_lstm, decoder_fc = create_decoder(
        vocab_size, embed_size, hidden_size
    )

    encoder_embedding.load_state_dict(model_state["encoder_embedding"])
    encoder_lstm.load_state_dict(model_state["encoder_lstm"])
    decoder_embedding.load_state_dict(model_state["decoder_embedding"])
    decoder_lstm.load_state_dict(model_state["decoder_lstm"])
    decoder_fc.load_state_dict(model_state["decoder_fc"])

    encoder_embedding.to(device)
    encoder_lstm.to(device)
    decoder_embedding.to(device)
    decoder_lstm.to(device)
    decoder_fc.to(device)

    print("'exit' - завершить чат-бота")
    while True:
        user_input = input("Введите сообщение чат-боту. 'exit' - чтобы выйти: > ")
        if user_input.lower() == "exit":
            break
        response = generate_response(
            encoder_embedding,
            encoder_lstm,
            decoder_embedding,
            decoder_lstm,
            decoder_fc,
            user_input,
            word_to_idx,
            idx_to_word,
            50,
            device,
        )
        print(f"Пользовательский ввод: {user_input}")
        print(f"{'Чат-бот':>21}: {response}")


if __name__ == "__main__":
    main()
