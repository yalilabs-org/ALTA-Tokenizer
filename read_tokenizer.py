import pickle
from kin_tokenizer.tokenizer import KinTokenizer
from kin_tokenizer.utils import create_sequences, preprocess_text


def read(path):
    with open(path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer



if __name__ == "__main__":

    tokenizer = KinTokenizer() # instantiating the tokenizer
    
    print(f"vocab size: {tokenizer.vocab_size}")

    text = "Kuwa mbere yagiye I Rubavu gusura masenge nsaga bose bagiye gusarura imyaka yukwa 9." * 5
    text = "<|PAD|> <|BOS|> <|EOS|> <|SEP|> <|MASK|> <|UNK|> <|CLS|>"

    # text = preprocess_text(text)
    print(text)
    encoded_text = tokenizer.encode(text)
    decoded_text = tokenizer.decode(encoded_text)

    # sequences = create_sequences(encoded_text, 10)

    # print(f"original text: {text}")
    print(f"Encoded text: {encoded_text}")
    print(f"Original length: {len(text)}\tEncoded length: {len(encoded_text)}")
    
    # print(f"Compression size: {len(text) / len(encoded_text):.2f}x")
    print(f"Decoded text: {decoded_text}\nlength: {len(decoded_text)}")


    # print(f"The first token: {min(tokenizer.vocab.keys())}: ({tokenizer.vocab[min(tokenizer.vocab.keys())]})")
    # print("\nSpecial tokens\n====================================\n")
    # count = 0
    # for key in reversed(sorted(list(tokenizer.vocab.keys()))):
    #     if count > 15:
    #         break
    #     print(key, ": ", tokenizer.vocab[key])
    #     count += 1
    



