import re

def pre_process(text):
    # lowercase
    text = text.lower()
    #remove tags
    text = re.sub("</?.*?>", " <> ", text)
    # remove special characters and digits

    text = re.sub("(\\d|\\W)+", " ", text)
    return text
