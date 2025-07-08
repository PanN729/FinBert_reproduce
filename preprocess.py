# preprocess.py
import pandas as pd

def load_phrasebank(path='C://Users//PanN//OneDrive//Desktop//FinBERT-Reproduce//data//Sentences_50Agree.txt'):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if '@' in line:
                parts = line.strip().split('@')
                text = parts[0].strip()
                label_str = parts[1].strip()
                if label_str == 'negative':
                    label = 0
                elif label_str == 'neutral':
                    label = 1
                elif label_str == 'positive':
                    label = 2
                else:
                    continue
                data.append((text, label))
    df = pd.DataFrame(data, columns=['text', 'label'])
    return df

