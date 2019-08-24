import pickle

stop_file_path = "resources/stopwords.txt"

with open(stop_file_path, 'r', encoding="utf-8") as f:
    stopwords = f.readlines()
    stop_set = set(m.strip() for m in stopwords)

with open("./resources/stopwords_set.pkl", "wb") as f:
    pickle.dump(stop_set, f, pickle.HIGHEST_PROTOCOL)
