
from kin_tokenizer.utils import create_dataset
import numpy as np
import re as re2
import regex as re


def remove_urls(text):
    """
    Removes URLs from a given text string.

    Args:
        text (str): The text to remove URLs from.

    Returns:
        str: The text with URLs removed.
    """
    # This is the simpler, more common regex
    url_pattern = re2.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text) 

if __name__ == "__main__":
    # pass
    # text_file_path = "alta_dataset.txt"
    # nbr_processes = 8
    # sequence_length = 512
    # step_size = 3
    # destination_dir ="C:\\Users\\Schadrack\\Desktop\\Test"

    # create_dataset(text_file_path=text_file_path, nbr_processes=nbr_processes, sequence_length=sequence_length, destination_dir=destination_dir, step_size=step_size)

    # destination_dir ="C:\\Users\\Schadrack\\Desktop\\Test\\sequences.npy"
    # seq = np.load(destination_dir, mmap_mode='r', allow_pickle=False)

    # # print(seq[:3])
    # count = 0
    # for s in seq:
    #     count += 1
    #     if 2000< count  < 2040:
    #         print(f"Length: {len(s)} \tMax token: {max(s)}\tToken at 200: {s[200]}\tMin token: {min(s)}")

    sample_text = """
    Here is my professional profile: https://www.linkedin.com/in/johndoe.
    Check out this amazing search engine: http://google.com.
    Sometimes people just write www.example.com without the http part.
    This text should remain.
    Another link with query parameters: https://api.service.com/v1/data?id=123&type=A
    """
    print("\n", sample_text)
    print("\n\n=========================\n\n")
    print(remove_urls(sample_text))

    text = "Ejo nagiye                 gusura*ubwo uzagruka *ryari"
    print("\n\n", text, "\n\n")

    text = re.sub(r'[^\S\r\n]+', ' ', text) # Where there is more than one space within the line followed by non-space, replace it with one space( wagiye    gusura -> wagiye gusura)
    print(text)

    text = re.sub(r'\*', ' ', text)

    print(text)

