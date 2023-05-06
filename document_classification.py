from PIL import Image, ImageDraw, ImageFont
import requests
import pytesseract
import numpy as np
import zipfile
import pandas as pd
from datasets import Dataset
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D
import os
from transformers import AdamW
from tqdm.notebook import tqdm
import torch
from transformers import LayoutLMv2FeatureExtractor, LayoutLMv2Tokenizer, LayoutLMv2Processor, LayoutLMv2ForSequenceClassification


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


image = Image.open(
    "./Data Base/PAN Card/PAN-Card.tiff")
image = image.convert("RGB")

ocr_df = pytesseract.image_to_data(image, output_type='data.frame')
ocr_df = ocr_df.dropna().reset_index(drop=True)
float_cols = ocr_df.select_dtypes('float').columns
ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
words = ' '.join([word for word in ocr_df.text if str(word) != 'nan'])


feature_extractor = LayoutLMv2FeatureExtractor()
tokenizer = LayoutLMv2Tokenizer.from_pretrained(
    "microsoft/layoutlmv2-base-uncased")
processor = LayoutLMv2Processor(feature_extractor, tokenizer)

encoded_inputs = processor(image, return_tensors="pt")

processor.tokenizer.decode(encoded_inputs.input_ids.squeeze().tolist())

dataset_path = "./Data Base"
labels = [label for label in os.listdir(dataset_path)]
id2label = {v: k for v, k in enumerate(labels)}
label2id = {k: v for v, k in enumerate(labels)}

print(label2id)

images = []
labels = []

for label_folder, _, file_name in os.walk(dataset_path):
    if label_folder != dataset_path:
        label = label_folder[12:]
        print(label)
        for _, _, image_names in os.walk(label_folder):
            relative_image_names = []
            for image_file in image_names:
                relative_image_names.append(
                    dataset_path + "/" + label + "/"+image_file)
            images.extend(relative_image_names)
            labels.extend([label] * len(relative_image_names))


data = pd.DataFrame.from_dict({"image_path:": images, "label": labels})
dataset = Dataset.from_pandas(data)

features = Features({
    'image': Array3D(dtype="int64", shape=(3, 224, 224)),
    'input_ids': Sequence(feature=Value(dtype='int64')),
    'attention_mask': Sequence(Value(dtype='int64')),
    'token_type_ids': Sequence(Value(dtype='int64')),
    'bbox': Array2D(dtype="int64", shape=(512, 4)),
    'labels': ClassLabel(num_classes=len(labels), names=labels),
})


def preprocess_data(datasets):
    images = [Image.open(path).convert("RGB")
              for path in datasets["image_path:"]]
    encoded_inputs = processor(
        images, padding="max_length", truncation=True)

    encoded_inputs["labels"] = [label2id[label] for label in datasets["label"]]
    return encoded_inputs


encoded_datasets = dataset.map(
    preprocess_data, remove_columns=dataset.column_names, features=features, batched=True, batch_size=2)

encoded_datasets.set_format(type="torch", device="cpu")
dataloader = torch.utils.data.DataLoader(encoded_datasets, batch_size=4)
batch = next(iter(dataloader))

processor.tokenizer.decode(batch["input_ids"][0].tolist())

test = id2label[batch['labels'][0].item()]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LayoutLMv2ForSequenceClassification.from_pretrained("microsoft/layoutlmv2-base-uncased",
                                                            num_labels=len(labels))
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

global_step = 0
num_train_epochs = 10
t_total = len(dataloader) * num_train_epochs  # total number of training steps

# put the model in training mode
model.train()
for epoch in range(num_train_epochs):
    print("Epoch:", epoch)
    running_loss = 0.0
    correct = 0
    for batch in tqdm(dataloader):
        # forward pass
        outputs = model(**batch)
        loss = outputs.loss

        running_loss += loss.item()
        predictions = outputs.logits.argmax(-1)
        correct += (predictions == batch['labels']).float().sum()

        # backward pass to get the gradients
        loss.backward()

        # update
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1

    print("Loss:", running_loss / batch["input_ids"].shape[0])
    accuracy = 100 * correct / len(data)
    print("Training accuracy:", accuracy.item())


def document_classify(image2):
    image2 = Image.open(image2)
    image2 = image2.convert("RGB")

    encoded_inputs = processor(image2, return_tensors="pt", truncation=True)
    for k, v in encoded_inputs.items():
        encoded_inputs[k] = v.to(model.device)

    outputs = model(**encoded_inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return id2label[predicted_class_idx]
