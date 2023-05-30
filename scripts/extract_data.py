import numpy as np
import torch
import torch.nn as nn
import pickle as pkl
import math

import librosa
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    is_apex_available,
    trainer_utils,
)

device = torch.device('cpu', 0)

loc = './finetuned_model/' #"./finetuned_model/" #'facebook/wav2vec2-large-100k-voxpopuli'
model = Wav2Vec2ForCTC.from_pretrained(loc, cache_dir = "tmp").to(device)
processor =  Wav2Vec2FeatureExtractor.from_pretrained(loc, cache_dir = "tmp") #Wav2Vec2Processor.from_pretrained(loc, cache_dir = "tmp")

logits = {}
embeddings = {}
pllrs = {}
for line in open("data/lp-test-multitranscriber/wav.txt").readlines():
    file_path = line.strip()
    id = file_path.split("/")[-1].replace(".flac","")
    print(id)
    speech, _ = librosa.load(file_path, sr=16000)
    input_values = processor(speech, sampling_rate=16000, return_tensors="pt", padding="longest").input_values
    print(input_values.size()[1]/16000)
    input_values_2 = None
    if (input_values.size()[1]/16000 > 300):
         input_values_2 = input_values[:, math.ceil(input_values.size()[1]/2):]
         input_values = input_values[:, :math.ceil(input_values.size()[1]/2)]
    with torch.no_grad():
        outputs = model.wav2vec2.feature_extractor(input_values).transpose(1, 2) # model(input_values.to(device)).last_hidden_state
        print(outputs.size())
        if input_values_2 != None:
            outputs_2 = model.wav2vec2.feature_extractor(input_values).transpose(1, 2) #(input_values_2.to(device)).last_hidden_state
            outputs = torch.cat((outputs,outputs_2), 1)
        #logits[id] = outputs.logits.numpy()[0,:,:]
        print(outputs.size())
        embeddings[file_path]  = outputs.to("cpu").numpy()
        #calc PLLR
        #pllrs[id] = np.log(np.exp(logits[id])/(1-np.exp(logits[id])))
        #print(logits[id].size(), embeddings[id].size())

#pkl.dump(logits, open("data/wave2vec2_output_features_adult_only.pkl", "wb"))

pkl.dump(embeddings, open("data/wave2vec2_finetuned_cnn_embeddings.pkl", "wb"))
