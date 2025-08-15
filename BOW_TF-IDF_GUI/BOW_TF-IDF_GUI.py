import random
import timeit
import tkinter as tk
from tkinter import scrolledtext, Toplevel
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import os
import re
from nltk.corpus import stopwords


input_folder_full = "C:/Users/nofar/PycharmProjects/AI_project/Rabbi Letters"
all_content_letters_whole = []
for filename in os.listdir(input_folder_full):
    if filename.endswith('.txt'):
        file_path = os.path.join(input_folder_full, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
            all_content_letters_whole.append(file_content)


def preprocess_text(text):
    # Remove non-Hebrew characters and punctuation
    cleaned_text = re.sub(r'[^א-ת\s]', '', text)
    # Remove multiple spaces and newlines
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    # Convert to lowercase
    cleaned_text = cleaned_text.lower()
    return cleaned_text

def bow(word_to_find):
    input_folder = "C:/Users/nofar/PycharmProjects/AI_project/Rabbi Letters"
    all_content_letters = []
    stop_words = set(stopwords.words('hebrew'))
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()
                cleaned_content = preprocess_text(file_content)
                cleaned_content = ' '.join([word for word in cleaned_content.split() if word not in stop_words])
                all_content_letters.append(cleaned_content)

    vectorized = CountVectorizer()
    vectorized.fit(all_content_letters)
    bow_model = vectorized.transform(all_content_letters)
    word_index = vectorized.vocabulary_.get(word_to_find, -1)

    if word_index != -1:
        word_freq_in_documents = bow_model[:, word_index].toarray().flatten()
        max_freq_documents_indices = []
        for i, freq in enumerate(word_freq_in_documents):
            if freq == word_freq_in_documents.max():
                max_freq_documents_indices.append(i)
        selected_document_index = random.choice(max_freq_documents_indices)
        output_window = Toplevel()
        output_window.title("Bag Of Word Output")
        output_window.geometry("400x300")
        output_text = scrolledtext.ScrolledText(output_window, wrap=tk.WORD, width=40, height=20)
        output_text.pack()
        output_text.insert(tk.END, 'with Bag Of Word\n')
        output_text.insert(tk.END, all_content_letters_whole[selected_document_index])

    else:
        output_window = Toplevel()
        output_window.title("Bag Of Word Output")
        output_window.geometry("400x300")
        output_text = scrolledtext.ScrolledText(output_window, wrap=tk.WORD, width=40, height=20)
        output_text.pack()
        output_text.insert(tk.END, f"The word '{word_to_find}' is not found in the vocabulary.")


def tfidf(word_to_find):
    input_folder = "C:/Users/nofar/PycharmProjects/AI_project/Rabbi Letters"
    all_content_letters = []
    stop_words = set(stopwords.words('hebrew'))

    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()
                cleaned_content = preprocess_text(file_content)
                cleaned_content = ' '.join([word for word in cleaned_content.split() if word not in stop_words])
                all_content_letters.append(cleaned_content)

    tf_idf_class = TfidfVectorizer()
    tfidf_matrix = tf_idf_class.fit_transform(all_content_letters)
    tf_idf_array = tfidf_matrix.toarray()
    word_in_files = tf_idf_class.get_feature_names_out()

    if word_to_find in word_in_files:
        word_index = np.where(word_in_files == word_to_find)[0]
        highest_score_index = np.argmax(tf_idf_array[:, word_index])
        highest_score_document = all_content_letters_whole[highest_score_index]

        output_window = Toplevel()
        output_window.title("TF-IDF Output")
        output_window.geometry("400x300")
        output_text = scrolledtext.ScrolledText(output_window, wrap=tk.WORD, width=40, height=20)
        output_text.pack()
        output_text.insert(tk.END, 'with TF-IDF\n')
        output_text.insert(tk.END, highest_score_document)

    else:
        output_window = Toplevel()
        output_window.title("TF-IDF Output")
        output_window.geometry("400x300")
        output_text = scrolledtext.ScrolledText(output_window, wrap=tk.WORD, width=40, height=20)
        output_text.pack()
        output_text.insert(tk.END, "There is no letters talk about this topic")


def create_gui():
    window = tk.Tk()
    window.title("Text Analysis")
    window.geometry("400x100")

    input_label = tk.Label(window, text="Enter your word in Hebrew:")
    input_label.pack()

    input_entry = tk.Entry(window)
    input_entry.pack()

    process_button = tk.Button(window, text="Process", command=lambda: process_word(input_entry.get()))
    process_button.pack()

    window.mainloop()


def process_word(input_word):
    bow(input_word)
    tfidf(input_word)


# runtime of bow
bow_time = timeit.timeit("bow('נסיכה')", setup="from __main__ import bow", number=5)

# runtime of tfidf
tfidf_time = timeit.timeit("tfidf('נסיכה')", setup="from __main__ import tfidf", number=5)

print("The runtime of the BOW function ", bow_time)
print("The runtime of the TF-IDF function ", tfidf_time)


create_gui()