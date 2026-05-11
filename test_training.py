import os
import sys
import time
import datetime
from kin_tokenizer.utils import train_kin_tokenizer
from helpers import send_mails



if __name__ == "__main__":

    args = sys.argv
    receiver_email = "niyibizischadrack05@gmail.com"
    DATA_ROOT = os.path.expanduser("~") # get home directory
    SAMPLE_TEXT_PATH = "alta_dataset.txt"
    with open(SAMPLE_TEXT_PATH, encoding="UTF-8", errors="ignore") as f:
        text = f.read()

    if len(args) > 1:
        if  args[1].lower() in ("train", "retrain"):
            training_start_time = datetime.datetime.now(datetime.timezone.utc)
            start_train = time.time()
            print("==================== STARTED ================================")

            # Training the tokenizer
            prompt = input(f"Do you really want to {args[1]}(Y/N): ")
            if prompt == "N":
                sys.exit(0)
            elif prompt not in ("N", "Y"):
                print("Wrong choice. Accepted Y/N")
                sys.exit(1)

            start_time = time.time()
            if args[1].lower() == "retrain" and len(args) < 3:
                print("You should provide the path containing tokenizer")
                sys.exit(1)
            
            if args[1].lower() == "retrain" and not os.path.exists(os.path.join(args[2], "kin_tokenizer.pkl")):
                print(f'The file {os.path.join(args[2], "kin_tokenizer.pkl")} does not exist!')
                sys.exit(1)

            if args[1].lower() == "retrain":
                tokenizer = train_kin_tokenizer(text, vocab_size=260, save=True, tokenizer_path=args[2], retrain=True)
            else:
                tokenizer = train_kin_tokenizer(text, vocab_size=260, save=True, tokenizer_path=None, retrain=False)

            total_time = time.time() - start_time
            training_end_time = datetime.datetime.now(datetime.timezone.utc)

            months, remaining = int(total_time // (3600 * 24 * 30)), total_time % (3600 * 24 * 30)
            days, remaining = int(remaining // (3600 * 24)), remaining % (3600 * 24)
            hours, remaining = int(remaining // 3600), remaining % 3600
            minutes, seconds = int(remaining // 60), int(remaining % 60)

            if months > 0:
                months = f"{months} months "
            else:
                months = ""
            
            if days > 0:
                days = f"{days} days "
            else:
                days = ""
            
            if hours > 0:
                hours = f"{hours} hours "
            else:
                hours = ""
            
            if minutes > 0:
                minutes = f"{minutes} minutes "
            else:
                minutes = ""
            
            if seconds > 0:
                seconds = f"{seconds} seconds"
            else:
                seconds = ""
            
            total_time = f"{months}{days}{hours}{minutes}{seconds}".strip()

            subject = f"{args[1].title()}ing ALTA Tokenizer Completed!"
            body = f"""<h1> {args[1].title()}ing ALTA Tokenizer </h1>
                    <h4>{args[1].title()}ing started on {training_start_time.strftime('%d-%m-%Y %H:%M:%S')} UTC </h4>
                    <h4>{args[1].title()}ing end on on {training_end_time.strftime('%d-%m-%Y %H:%M:%S')} UTC </h4>
                    <h4>Took: {total_time}</h4>
            """
            send_mails(receiver_email, subject, body)
    
    else:
        print("You should provide command line argument:")
        print("\ttrain: python test_training.py train/retrain -> for training tokenizer")






