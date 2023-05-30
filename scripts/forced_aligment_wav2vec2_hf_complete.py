# -*- coding: utf-8 -*-


import os
from grid import IntervalTier
from grid import TextGrid
from dataclasses import dataclass
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
#import torchaudio
import pickle
import librosa
import sys, math

torch.random.manual_seed(0)
device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

sampling_rate = 16_000

ref_dir = "data/lp-test-multitranscriber/"
def convert_to_id(text):
    t = text.split("/")
    id = t[-2]+t[-1].replace(".flac","")
    if id.startswith("ses"):
        id = t[-3]+id
    return id

def load_ref(file):
    with open(file,"rt",encoding="utf-8") as f:
        text=f.read().split('\n')
    ref = {}
    for t in text:
        ref[t.split(" ")[0]] = " ".join(t.split(" ")[1:])
    return ref

link = './finetuned_model'
#replace this with Yaroslav's. HF. submisssion
processor = Wav2Vec2Processor.from_pretrained(link, cache_dir = "tmp")
model = Wav2Vec2ForCTC.from_pretrained(link, cache_dir = "tmp")

#ref, hyp= pickle.load(open('../asr_test_data', 'rb'))

SPEECH_FILE = "/teamwork/t40511_asr/p/teflon/SpeechData/KI_SpeechData_IntellProject/wavs/0001.wav"

labels = processor.tokenizer.get_vocab()
refs = ["text1", "text2", "text3", "text4"]
ref = load_ref(ref_dir+sys.argv[1])

for line in open("data/lp-test-multitranscriber/wav.txt").readlines():
    file_path = line.strip()
    id = file_path.split("/")[-1].replace(".flac","")
    print(file_path)
    SPEECH_FILE = file_path
    transcript = ref[convert_to_id(file_path)].upper().replace("-", "")
    #remove - and other marks starting with .
    transcript = " ".join(word for word in transcript.split() if not word.startswith("."))
    transcript = transcript.replace(" ","|")
    with torch.inference_mode():
        waveform, _ = librosa.load(SPEECH_FILE, sr=16000)
        waveform = processor(waveform, sampling_rate=16000, return_tensors="pt", padding="longest").input_values
        input_values_2 = None
        if (waveform.size()[1]/16000 > 300):
            input_values_2 = waveform[:, math.ceil(waveform.size()[1]/2):]
            waveform = waveform[:, :math.ceil(waveform.size()[1]/2)]
        emissions = model(waveform.to(device)).logits
        if input_values_2 != None:
            outputs_2 = model(input_values_2.to(device)).logits
            emissions = torch.cat((emissions,outputs_2), 1)
        emissions = torch.log_softmax(emissions, dim=-1)

    emission = emissions[0].cpu().detach()

    dictionary = {c: i for i, c in enumerate(labels)}
    print(dictionary, transcript)
    tokens = [dictionary[c] for c in transcript]
    print(list(zip(transcript, tokens)))


    def get_trellis(emission, tokens, blank_id=0):
        num_frame = emission.size(0)
        num_tokens = len(tokens)
        trellis = torch.full((num_frame + 1, num_tokens + 1), -float("inf"))
        trellis[:, 0] = 0
        for t in range(num_frame):
            trellis[t + 1, 1:] = torch.maximum(
                trellis[t, 1:] + emission[t, blank_id],
                trellis[t, :-1] + emission[t, tokens],
            )
        return trellis


    trellis = get_trellis(emission, tokens)

    @dataclass
    class Point:
        token_index: int
        time_index: int
        score: float


    def backtrack(trellis, emission, tokens, blank_id=0):
        j = trellis.size(1) - 1
        t_start = torch.argmax(trellis[:, j]).item()

        path = []
        for t in range(t_start, 0, -1):
            stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
            changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

            prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()

            path.append(Point(j - 1, t - 1, prob))

            if changed > stayed:
                j -= 1
                if j == 0:
                    break
        else:
            raise ValueError("Failed to align")
        return path[::-1]


    path = backtrack(trellis, emission, tokens)
    print(path)

    # Merge the labels
    @dataclass
    class Segment:
        label: str
        start: int
        end: int
        score: float

        def __repr__(self):
            return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

        @property
        def length(self):
            return self.end - self.start


    def merge_repeats(path):
        i1, i2 = 0, 0
        segments = []
        while i1 < len(path):
            while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                i2 += 1
            score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
            segments.append(
                Segment(
                    transcript[path[i1].token_index],
                    path[i1].time_index,
                    path[i2 - 1].time_index + 1,
                    score,
                )
            )
            i1 = i2
        return segments


    segments = merge_repeats(path)

    # Merge words
    def merge_words(segments, separator="|"):
        words = []
        i1, i2 = 0, 0
        while i1 < len(segments):
            if i2 >= len(segments) or segments[i2].label == separator:
                if i1 != i2:
                    segs = segments[i1:i2]
                    word = "".join([seg.label for seg in segs])
                    score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                    words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
                i1 = i2 + 1
                i2 = i1
            else:
                i2 += 1
        return words


    word_segments = merge_words(segments)
    for word in word_segments:
        print(word)

    ratio = (waveform.size(1) / (trellis.size(0) - 1)) / sampling_rate
    for w in segments:
      print(f"Char:{w.label}\tStarts:{(w.start*ratio):.3f}\tEnds:{(w.end*ratio):.3f}\tScore:{ (w.score):.3f}")

    ratio = (waveform.size(1) / (trellis.size(0) - 1)) /sampling_rate
    for w in word_segments:
      print(f"Word: {w.label}\tStarts:{(w.start*ratio):.8f}\tEnds:{(w.end*ratio):.6f}\tScore:{ (w.score):.3f}")

    '''import IPython

    def display_segment(i, save_seg=False):
        ratio = waveform.size(1) / (trellis.size(0) - 1)
        word = word_segments[i]
        x0 = int(ratio * word.start)
        x1 = int(ratio * word.end)
        filename = f"_assets/{i}_{word.label}.wav"
        if save_seg:
          torchaudio.save(filename, waveform[:, x0:x1], sampling_rate)
        print(f"{word.label} ({word.score:.2f}): {x0/sampling_rate:.3f} - {x1/sampling_rate:.3f} sec")
        return IPython.display.Audio(filename)
    '''
    # Generate the audio for each segment
    # print(transcript)
    # IPython.display.Audio(SPEECH_FILE)

    # display_segment(0)

    # display_segment(1)

    # display_segment(2)




    tier = IntervalTier('words')
    tier_char = IntervalTier('chars')
    txtgrid = TextGrid()

    # output words
    for w in word_segments:
      word = w.label
      start_time = (w.start*ratio)
      end_time = (w.end*ratio) 
      duration = end_time - start_time
      tier.add(start_time, end_time, word)

    # output chars
    for w in segments:
      word = w.label
      start_time = (w.start*ratio)
      end_time = (w.end*ratio) 
      duration = end_time - start_time
      tier_char.add(start_time, end_time, word)


    txtgrid.append(tier)
    txtgrid.append(tier_char)
    txtgrid.write('data/align/'+sys.argv[1]+'/'+convert_to_id(file_path)+'.TextGrid')
