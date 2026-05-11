import os
import sys
import argparse
import time
import datetime
from kin_tokenizer.utils import train_kin_tokenizer
from helpers import send_mails, initialize_wandb



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script for utilizing ALTA models",
                                        formatter_class=argparse.RawTextHelpFormatter)

    # Adding arguments
    parser.add_argument("-a", "--action", help="""Action to be performed: train, retrain""")
    parser.add_argument("-d", "--save_checkpoint_path", help="""The path to save tokenizer checkpoint""")
    parser.add_argument("-p", "--project_name", type=str, default=None, help="Project name to be used on WANDB")
    parser.add_argument("-w", "--wandb_run", type=str, default=None, help="The run name for WANDB")

    args = parser.parse_args()
    receiver_email = "niyibizischadrack05@gmail.com"
    DATA_ROOT = os.path.expanduser("~") # get home directory
    SAMPLE_TEXT_PATH = os.path.join(DATA_ROOT, "data", "alta_dataset.txt")
    with open(SAMPLE_TEXT_PATH, encoding="UTF-8", errors="ignore") as f:
        text = f.read()

    if  args.action.lower() in ("train", "retrain"):
        training_start_time = datetime.datetime.now(datetime.timezone.utc)
        start_train = time.time()
        print("==================== STARTED ================================")

        # Training the tokenizer
        prompt = input(f"Do you really want to {args.action}(Y/N): ")
        if prompt == "N":
            sys.exit(0)
        elif prompt not in ("N", "Y"):
            print("Wrong choice. Accepted Y/N")
            sys.exit(1)

        start_time = time.time()
        if args.action.lower() == "retrain":
            print("You should provide the path containing tokenizer")
            sys.exit(1)
        
        if args.action.lower() == "retrain" and not os.path.exists(os.path.join(args.save_checkpoint_path, "kin_tokenizer.pkl")):
            print(f'The file {os.path.join(args.save_checkpoint_path, "kin_tokenizer.pkl")} does not exist!')
            sys.exit(1)

        if args.project_name and args.wandb_run:
            initialize_wandb(project_name=args.project_name, run_name=args.wandb_run, resume=False)

        if args.action.lower() == "retrain":
            tokenizer = train_kin_tokenizer(text, vocab_size=50251, save=True, tokenizer_path=args.save_checkpoint_path, retrain=True)
        else:
            tokenizer = train_kin_tokenizer(text, vocab_size=50251, save=True, tokenizer_path=DATA_ROOT, retrain=False)

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

        subject = f"{args.action.title()}ing ALTA Tokenizer Completed!"
        body = f"""<h1> {args.action.title()}ing ALTA Tokenizer </h1>
                <h4>{args.action.title()}ing started on {training_start_time.strftime('%d-%m-%Y %H:%M:%S')} UTC </h4>
                <h4>{args.action.title()}ing end on on {training_end_time.strftime('%d-%m-%Y %H:%M:%S')} UTC </h4>
                <h4>Took: {total_time}</h4>
        """
        send_mails(receiver_email, subject, body)






