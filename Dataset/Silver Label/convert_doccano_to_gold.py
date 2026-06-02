import json
import pandas as pd
import os

# Ganti dengan nama file hasil export Doccano kamu
INPUT_FILE = 'admin.jsonl'  
OUTPUT_FILE = 'gold_test.csv'

if not os.path.exists(INPUT_FILE):
    print(f"❌ Error: File {INPUT_FILE} tidak ditemukan!")
    exit()

rows = []
sentence_id = 0

print(f"🔄 Memulai konversi {INPUT_FILE}...")

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        data = json.loads(line.strip())
        text = data['text']
        labels = data.get('label', [])  # [[start, end, 'LABEL'], ...]
        
        # Tokenisasi whitespace
        tokens = text.split()
        
        # Mapping char position ke token index
        char_to_token = {}
        char_idx = 0
        for i, token in enumerate(tokens):
            start = text.find(token, char_idx)
            if start == -1:
                start = char_idx
            end = start + len(token)
            for c in range(start, end):
                char_to_token[c] = i
            char_idx = end
        
        # Default semua O
        token_labels = ['O'] * len(tokens)
        
        # Apply labels dari Doccano
        for start, end, label in labels:
            token_indices = set()
            for c in range(start, end):
                if c in char_to_token:
                    token_indices.add(char_to_token[c])
            
            token_indices = sorted(list(token_indices))
            for i, idx in enumerate(token_indices):
                if i == 0:
                    token_labels[idx] = f'B-{label}'
                else:
                    token_labels[idx] = f'I-{label}'
        
        # Add to rows
        for token, label in zip(tokens, token_labels):
            rows.append({
                'sentence_id': sentence_id,
                'token': token,
                'label': label
            })
        
        sentence_id += 1

# Save
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ {OUTPUT_FILE} berhasil dibuat!")
print(f"   Total sentences: {sentence_id}")
print(f"   Total tokens: {len(df)}")
print(f"\nLabel distribution:")
print(df['label'].value_counts())
