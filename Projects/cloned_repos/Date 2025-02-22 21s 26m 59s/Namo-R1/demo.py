from namo.api.vl import VLInfer
import os
from termcolor import colored
import torch


def chat():
    model = VLInfer(
        model_type="namo", device="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    crt_input = ["images/cats.jpg", None]

    while True:
        img_or_txt = input(colored("\nUser (txt/img_path): ", "cyan")).strip()

        if os.path.exists(img_or_txt.split(" ")[0]):
            crt_input[0] = img_or_txt
            print(colored("System: Image updated.", "green"))
            continue
        else:
            crt_input[1] = img_or_txt

        if crt_input[0] and crt_input[1]:
            print(colored("Assistant:", "green"), end=" ")
            model.generate(images=crt_input[0], prompt=crt_input[1], verbose=False)
            crt_input[0] = None
        elif not crt_input[0] and crt_input[1]:
            # pure text
            print(colored("Assistant:", "green"), end=" ")
            model.generate(images=None, prompt=crt_input[1], verbose=False)
        else:
            print(
                colored("System: Please provide either an image or text input.", "red")
            )


if __name__ == "__main__":
    chat()
