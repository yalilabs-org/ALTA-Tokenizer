# ALTA Tokenizer - Quick Start Guide

## 🚀 Getting Started in 5 Minutes

This guide will help you train and use the ALTA tokenizer quickly.

---

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `tokenizers>=0.13.0` - Hugging Face tokenizers library (core)
- `regex` - For pattern matching
- Other utilities

---

## Step 2: Prepare Your Data

The tokenizer supports two data formats:

### Option A: Plain Text File (.txt)
```
Muraho! Amakuru?
Nuko Semugeshi akunda uwo mutwe w'abatwa.
Ejo hazaza abashyitsi.
```

### Option B: JSONL File (.jsonl)
```json
{"text": "Muraho! Amakuru?"}
{"text": "Nuko Semugeshi akunda uwo mutwe w'abatwa."}
{"text": "Ejo hazaza abashyitsi."}
```

---

## Step 3: Train the Tokenizer

### Basic Training

```bash
python train_hf_tokenizer.py \
    --data_path sample_data.txt \
    --output_dir ./my_tokenizer
```

### With Custom Settings

```bash
python train_hf_tokenizer.py \
    --data_path ./data/ \
    --output_dir ./my_tokenizer \
    --vocab_size 65536 \
    --batch_size 1000 \
    --test
```

**Parameters:**
- `--data_path`: Path to your training data (file or directory)
- `--output_dir`: Where to save the trained tokenizer
- `--vocab_size`: Vocabulary size (default: 65536)
- `--batch_size`: Streaming batch size (default: 1000)
- `--test`: Run tests after training

**Training Output:**
```
==================================================================
ALTA Tokenizer Training - Hugging Face Backend
==================================================================
Configuration:
  - Algorithm: Byte-Level BPE with byte_fallback=True
  - Vocabulary Size: 65,536
  - Data Path: sample_data.txt
  - Output Directory: ./my_tokenizer
  - CPU Cores: 8
  - Batch Size: 1000
==================================================================

[1/3] Creating tokenizer with Kinyarwanda-specific configuration...
[2/3] Streaming training data...
Found 1 file(s) to process
Processing: sample_data.txt
[3/3] Training tokenizer (this may take a while)...
[4/4] Saving tokenizer...
✓ Tokenizer saved to: ./my_tokenizer/tokenizer.json
✓ Config saved to: ./my_tokenizer/tokenizer_config.json

==================================================================
Training completed successfully!
==================================================================
```

---

## Step 4: Use the Tokenizer

### Method 1: Using ALTATokenizer Class

```python
from alta_tokenizer import ALTATokenizer

# Load tokenizer
tokenizer = ALTATokenizer.from_pretrained("./my_tokenizer/tokenizer.json")

# Encode text
text = "Nuko Semugeshi akunda uwo mutwe w'abatwa."
token_ids = tokenizer.encode(text)
print(token_ids)  # [245, 123, 456, ...]

# Decode back
decoded = tokenizer.decode(token_ids)
print(decoded)  # "Nuko Semugeshi akunda uwo mutwe w'abatwa."

# Batch processing
texts = ["Muraho!", "Amakuru?", "Ejo hazaza."]
encoded_batch = tokenizer.encode_batch(texts)
decoded_batch = tokenizer.decode_batch(encoded_batch)
```

### Method 2: Direct Hugging Face API

```python
from tokenizers import Tokenizer

# Load tokenizer
tokenizer = Tokenizer.from_file("./my_tokenizer/tokenizer.json")

# Encode
encoding = tokenizer.encode("Muraho! Amakuru?")
print(encoding.tokens)  # ['Mur', 'aho', '!', ...]
print(encoding.ids)     # [245, 123, 456, ...]

# Decode
text = tokenizer.decode(encoding.ids)
print(text)  # "Muraho! Amakuru?"
```

---

## Step 5: Test with Examples

Run the provided example script:

```bash
python example_usage.py
```

This demonstrates:
1. Basic encoding/decoding
2. Apostrophe handling (Kinyarwanda feature)
3. Special tokens usage
4. Batch processing
5. Compression analysis
6. Vocabulary inspection

---

## Common Use Cases

### 1. Training Language Models

```python
from alta_tokenizer import ALTATokenizer
from transformers import GPT2Config, GPT2LMHeadModel

# Load tokenizer
tokenizer = ALTATokenizer.from_pretrained("./my_tokenizer/tokenizer.json")

# Create model with matching vocab size
config = GPT2Config(vocab_size=tokenizer.vocab_size)
model = GPT2LMHeadModel(config)

# Tokenize training data
text = "Your training text here"
input_ids = tokenizer.encode(text)
```

### 2. Text Classification

```python
from alta_tokenizer import ALTATokenizer

tokenizer = ALTATokenizer.from_pretrained("./my_tokenizer/tokenizer.json")

# Enable padding and truncation
tokenizer.enable_padding(pad_id=0, pad_token="<|pad|>")
tokenizer.enable_truncation(max_length=512)

# Prepare batch
texts = ["Text 1", "Text 2", "Text 3"]
input_ids = tokenizer.encode_batch(texts)
# Now ready for your classifier!
```

### 3. Chat/Instruction Tuning

```python
from alta_tokenizer import ALTATokenizer

tokenizer = ALTATokenizer.from_pretrained("./my_tokenizer/tokenizer.json")

# Format conversation
conversation = """<|bos|><|im_start|>system
Uri umufasha w'ikinyarwanda.<|im_end|>
<|im_start|>user
Muraho! Wampfashije iki?<|im_end|>
<|im_start|>assistant
Muraho! Ndashobora kugufasha mu bintu byinshi.<|im_end|><|eos|>"""

# Encode
input_ids = tokenizer.encode(conversation)
```

---

## Performance Tips

### 1. Large Datasets
For datasets larger than available RAM, the training script automatically streams data:

```bash
# This handles 100GB+ datasets efficiently
python train_hf_tokenizer.py \
    --data_path /path/to/large/dataset/ \
    --output_dir ./tokenizer \
    --batch_size 5000  # Increase for better throughput
```

### 2. Multi-core Utilization
The script automatically uses all CPU cores. You'll see output like:
```
CPU Cores: 16
Running 16 processes
```

### 3. Memory Management
- Batch size controls memory usage
- Larger batches = faster training (if RAM allows)
- Default 1000 works well for most systems

---

## Troubleshooting

### Issue: "FileNotFoundError: Tokenizer file not found"
**Solution:** Check your path:
```python
from pathlib import Path
print(Path("./my_tokenizer/tokenizer.json").exists())
```

### Issue: "Out of Memory"
**Solution:** Reduce batch size:
```bash
python train_hf_tokenizer.py \
    --data_path data.txt \
    --output_dir ./tokenizer \
    --batch_size 500  # Reduced from 1000
```

### Issue: "Training is slow"
**Solutions:**
1. Ensure you're using all CPU cores (check output)
2. Use SSD instead of HDD for data
3. Increase batch size if you have RAM
4. For very small datasets, training might be slow due to overhead

### Issue: "Apostrophes are being split"
**Check:** The training script includes custom regex for Kinyarwanda. If you modified it, ensure the pattern is:
```python
r"\p{L}+(?:'\p{L}+)*"
```

---

## Next Steps

1. **Read the full documentation**: See `README_HF.md` for detailed information
2. **Test apostrophe handling**: Use the example script to verify
3. **Integrate with your project**: Use `alta_tokenizer` package
4. **Fine-tune for your domain**: Train on your specific corpus

---

## File Structure After Training

```
my_tokenizer/
├── tokenizer.json          # Main tokenizer file (load this)
└── tokenizer_config.json   # Configuration metadata
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Train tokenizer | `python train_hf_tokenizer.py --data_path data.txt --output_dir ./tokenizer` |
| Test tokenizer | `python train_hf_tokenizer.py --data_path data.txt --output_dir ./tokenizer --test` |
| Run examples | `python example_usage.py` |
| Load in Python | `ALTATokenizer.from_pretrained("./tokenizer/tokenizer.json")` |

---

## Support

- **Documentation**: `README_HF.md`
- **Issues**: GitHub Issues
- **Email**: niyibizischadrack@gmail.com

---

**Ready to go! Train your first tokenizer now! 🚀**

```bash
python train_hf_tokenizer.py \
    --data_path sample_data.txt \
    --output_dir ./my_first_tokenizer \
    --test
```
