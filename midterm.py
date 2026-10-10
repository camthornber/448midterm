# Libraries
from pathlib import Path 
from collections import Counter 
import copy 
import json 
import random 
import re 
import time 
import pandas as pd 
import torch
from torch import nn 
from torch.utils.data import DataLoader, TensorDataset 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import ( 
    accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

data_path = "TXD-22 A Large-Scale Benchmark Dataset.csv" 
results_dir = Path("results")

#classification labels 
sources = ["ChatGPT", "Claude", "Gemini"]
label_map ={"ChatGPT":0, "Claude":1, "Gemini":2} 

#data, model, training settings 
seed = 448
vocab_limit = 20000 
prompt_limit = 64 
response_limit = 512
embedding_size = 64 
hidden_size = 64 
filters = 64 
kernel_size = 3 
batch_size = 64 
learning_rate = 0.001
epochs = 10
max_prompts = None
pad_id = 0 
unknown_id = 1 
seperator_id = 2 

#seed set 
def set_seed(seed): 
    random.seed(seed)
    torch.manual_seed(seed)

#data cleaning 
def load_data(path):
    data = pd.read_csv(path, encoding="latin1")
    encoding = "latin1"
    req_columns = ["Question", "Answer", "Source"]
    data = data[req_columns].copy()
    data = data.dropna(subset=req_columns)
    #remove spaces, empty prompts and responses 
    for column in req_columns:
        data[column] = data[column].astype(str).str.strip() 
    data = data[
        (data["Question"] != "") &
        (data["Answer"] != "")
    ]
    #select only chatgpt, claude, gemini 
    data = data[data["Source"].isin(sources)]
    #find prompts across all 3 sources
    data = data.drop_duplicates(subset=req_columns)
    data = data.drop_duplicates(subset=["Question", "Source"]) 
    prompt_counts = data.groupby("Question")["Source"].nunique() 
    shared_prompts = prompt_counts[prompt_counts == len(sources)].index.tolist()
    shared_prompts = sorted(shared_prompts)
    data = data[data["Question"].isin(shared_prompts)].copy() 
    data["label"] = data["Source"].map(label_map)
    data = data.reset_index(drop = True)

    return data, encoding

#datasplit  70/15/15
def split_data(data):
    prompts = sorted(data["Question"].unique())

    train_prompts, remaining = train_test_split(
        prompts, 
        test_size = 0.30,
        random_state = seed,
    )

    validation_prompts, test_prompts = train_test_split(
        remaining, 
        test_size = 0.50, 
        random_state = seed,
    )

    train = data[data["Question"].isin(train_prompts)].copy() 
    valid = data[data["Question"].isin(validation_prompts)].copy() 
    test = data[data["Question"].isin(test_prompts)].copy() 
    return train, valid, test

#tokenization: seperating the words and punctuation marks 
def tokenize(text):
    text = text.lower() 
    tokens = re.findall(r"\w+|[^\w\s]", text) #a regex: \w+ matches word characters, ^\w\s grabs punctuation
    return tokens 

def build_vocab(datset):
    counts = Counter() 
    for question in datset["Question"].unique(): 
        counts.update(tokenize(question))
    for answer in datset["Answer"]:
        counts.update(tokenize(answer))
    vocab = {
        "<PAD>": pad_id, 
        "<UNK>": unknown_id, 
        "<SEP>": seperator_id
    }
    for token, count in counts.most_common(vocab_limit):
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab 

def token_id(tokens, vocab): 
    ids = [] 
    for token in tokens: 
        token_id = vocab.get(token, unknown_id)
        ids.append(token_id)
    return ids 

#encoding for rq2: input only, output only, input + output 
def make_loader(data, vocab, mode, shuffle): 
    if mode == "input": 
        max_length = prompt_limit
    elif mode == "output":
        max_length = response_limit
    else:
        max_length = prompt_limit + 1 + response_limit
    sequences = []
    lengths = []
    truncated_count = 0

    for row in data.itertuples(index=False):
        prompt_tokens = tokenize(row.Question)
        response_tokens = tokenize(row.Answer)
        if mode == "input":
            tokens = prompt_tokens[:prompt_limit]
            ids = token_id(tokens, vocab)
            truncated = len(prompt_tokens) > prompt_limit
        elif mode == "output": 
            tokens = response_tokens[:response_limit]
            ids = token_id(tokens, vocab)
            truncated = len(response_tokens) > response_limit
        else: 
            prompt_ids = token_id( prompt_tokens[:prompt_limit], vocab)
            response_ids = token_id(response_tokens[:response_limit], vocab)
            ids = prompt_ids + [seperator_id] + response_ids
            truncated = (len(prompt_tokens) > prompt_limit or len(response_tokens) > response_limit)
        if len(ids) == 0:
            ids = [unknown_id]
        lengths.append(len(ids))
        padding_needed = max_length - len(ids)
        ids = ids + [pad_id] * padding_needed
        sequences.append(ids)
        truncated_count += int(truncated)

    x = torch.tensor(sequences, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)
    y = torch.tensor(data["label"].tolist(), dtype=torch.long)
    dataset = TensorDataset(x, lengths, y)
    loader = DataLoader( 
        dataset,
        batch_size=batch_size,
        shuffle = shuffle, 
    )
    truncated_fraction = truncated_count / len(data)
    return loader, truncated_fraction 

#MODELS 

# CNN model
#init establishes the layers, forward passes the dadta through them
class CNNClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_id)
        self.convolution = nn.Conv1d(in_channels=embedding_size, out_channels=filters, kernel_size=kernel_size, padding=kernel_size // 2)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(filters, len(sources))
    def forward(self,x,lengths): 
        embedded=self.embedding(x)
        embedded=embedded.transpose(1,2)
        features = self.convolution(embedded)
        features = self.activation(features)
        positions = torch.arange(x.size(1), device = x.device)
        padding_mask = (positions.unsqueeze(0) >= lengths.unsqueeze(1))
        features = features.masked_fill(padding_mask.unsqueeze(1), float("-inf"))
        pooled = features.max(dim=2).values 
        scores = self.classifier(pooled) 
        return scores

# RNN (LSTM) model 
#which processes the tokens in order, with the hidden state being the last real token.
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size): 
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_id)
        self.lstm = nn.LSTM(input_size = embedding_size, hidden_size = hidden_size, batch_first = True)
        self.classifier = nn.Linear(hidden_size, len(sources))
    def forward(self, x, lengths):
        embedded = self.embedding(x)
        outputs, _ = self.lstm(embedded)
        batch_indices = torch.arange(x.size(0), device=x.device)
        last_hidden=outputs[batch_indices, lengths - 1]
        scores = self.classifier(last_hidden)
        return scores

#eval: calculates loss and accuracy, collects predictions 
def evaluate_model(model, loader, loss_fun, device): 
    model.eval() 
    total_loss = 0.0 
    true_labels = []
    predictions = []
    with torch.no_grad(): 
        for x, lengths, y in loader: 
            x = x.to(device) 
            lengths = lengths.to(device)
            y = y.to(device) 
            scores = model(x, lengths)
            loss = loss_fun(scores,y)
            total_loss += loss.item() * y.size(0)
            predicted_labels = scores.argmax(dim=1)
            predictions.extend(predicted_labels.cpu().tolist())
            true_labels.extend(y.cpu().tolist())
    average_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(true_labels, predictions)
    return average_loss, accuracy, true_labels, predictions 

#training
def train_model(model, train_loader, valid_loader, device):
    loss_fun = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float("inf")
    best_weights = None 
    best_epoch = 0 
    history = []
    for epoch in range(1, epochs + 1):
        model.train() 
        total_loss = 0.0 
        correct = 0 
        for x, lengths, y in train_loader:
            x = x.to(device)
            lengths = lengths.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            scores = model(x,lengths)
            loss = loss_fun(scores,y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            predicted_labels = scores.argmax(dim=1)
            correct += (predicted_labels == y).sum().item()
        train_loss = total_loss / len(train_loader.dataset)
        train_accuracy = correct / len(train_loader.dataset)
        valid_loss, valid_accuracy, _, _ = evaluate_model(model, valid_loader, loss_fun, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy, "valid_loss": valid_loss, "valid_accuracy": valid_accuracy})
        if valid_loss < best_loss: 
            best_loss = valid_loss 
            best_epoch = epoch 
            best_weights = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_weights)
    return pd.DataFrame(history), best_epoch 

#results 
def save_run(model, history, test, true_labels, predictions, run_dir): 
    run_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(run_dir / "training_history.csv", index=False)
    cpu_weights = {} 
    for name,value in model.state_dict().items():
        cpu_weights[name] = value.detach().cpu() 
    torch.save(cpu_weights, run_dir / "model.pt")
    labels = list(range(len(sources)))
    report = classification_report(true_labels, predictions, labels = labels, target_names = sources, output_dict = True, zero_division = 0)
    report_table = pd.DataFrame(report).transpose()
    report_table.to_csv(run_dir / "classification_report.csv")
    prediction_table = test[["Question", "Source", "label"]].copy() 
    prediction_table["predicted_label"] = predictions 
    predicted_sources = [] 
    for label in predictions: 
        predicted_sources.append(sources[label])   
    prediction_table["predicted_source"] = predicted_sources 
    prediction_table.to_csv(run_dir / "test_predictions.csv", index = False)
    matrix = confusion_matrix(true_labels, predictions, labels = labels)
    matrix_table = pd.DataFrame(matrix, index = sources, columns = sources)
    matrix_table.to_csv(run_dir / "confusion_matrix.csv")
    display = ConfusionMatrixDisplay(confusion_matrix = matrix, display_labels = sources) 
    display.plot(cmap="Blues", colorbar=False)
    plt.savefig(run_dir / "confusion_matrix.png", dpi=150)
    plt.close() 
    plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="Training")
    plt.plot(history["epoch"], history["valid_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.legend() 
    plt.savefig(run_dir / "loss_curve.png", dpi = 150)
    plt.close()

    
#main

def main():
#prepare the Rq1 and Rq2 experiments 
    set_seed(seed)
    device = torch.device("cpu")
    results_dir.mkdir(parents = True, exist_ok = True)
    data, encoding = load_data(data_path)
    train, valid, test = split_data(data)
    vocab = build_vocab(train)
    with open(results_dir / "vocabulary.json", "w", encoding = "utf-8") as file:
        json.dump(vocab, file, ensure_ascii=False, indent=2)
    split_tables = [] 
    for split_name, frame in [("train", train), ("validation", valid), ("test",test)]:
        frame = frame.copy()
        frame["split"] = split_name 
        split_tables.append(frame)

    prepared_Data = pd.concat(split_tables, ignore_index=True)
    prepared_Data.to_csv(results_dir / "prepared_data.csv", index=False)

#run all six experiments 
    results = []

    for mode in ["output", "input", "combined"]: #output is the main for RQ1. Then, it cycles for RQ2.
        train_loader, train_truncated = make_loader(train, vocab, mode, shuffle=True)
        validation_loader, validation_truncated = make_loader(valid, vocab, mode, shuffle=False)
        test_loader, test_truncated = make_loader(test, vocab, mode, shuffle=False)

        for model_name in ["CNN", "LSTM"]:
            set_seed(seed)
            print(f"\nRunning {model_name}, text mode={mode}")
            if model_name == "CNN":
                model = CNNClassifier(len(vocab))
            else:
                model = LSTMClassifier(len(vocab))
            model = model.to(device)
            start = time.perf_counter()

            history, best_epoch = train_model(model, train_loader, validation_loader, device)
            training_seconds = time.perf_counter() - start
            test_loss, test_accuracy, true_labels, predictions = evaluate_model(model, test_loader, nn.CrossEntropyLoss(), device)
            macro_f1 = f1_score(true_labels, predictions, labels=list(range(len(sources))), average="macro", zero_division=0)
            run_dir = results_dir / f"{model_name}_{mode}"

            save_run(model, history, test, true_labels, predictions, run_dir)
            results.append({"model": model_name, "text_mode": mode, "test_accuracy": test_accuracy, "macro_f1": macro_f1, "test_loss": test_loss, "best_epoch": best_epoch, "training_seconds": training_seconds, "train_truncated_fraction": train_truncated, "validation_truncated_fraction": validation_truncated, "test_truncated_fraction": test_truncated})
            results_table = pd.DataFrame(results)

            results_table.to_csv(results_dir / "results_summary.csv", index=False)

            print(f"Test accuracy={test_accuracy:.3f}, "f"macro-F1={macro_f1:.3f}")

    print("\nFinal results:")
    print(pd.DataFrame(results).to_string(index=False))

    print(f"\nSaved outputs in {results_dir}")


if __name__ == "__main__":
    main()