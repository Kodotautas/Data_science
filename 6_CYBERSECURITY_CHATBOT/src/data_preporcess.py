import os
import pandas as pd

# convert df to tuple of questions and answers with '\n' as delimiter
def df_to_tuple(df):
    questions = df['Question'].tolist()
    answers = df['Answer'].tolist()
    faq = ''
    for i in range(len(questions)):
        faq += questions[i] + '\n' + answers[i] + '\n'
    return faq

def preprocess_text(text):
    # Split the text into individual sentences and conversations
    # (a conversation is a group of consecutive sentences that are exchanged between two or more people)
    sentences = text.split("\n")
    conversations = []
    current_conversation = []
    for sentence in sentences:
        if sentence == "":
            # End of a conversation
            if current_conversation:
                conversations.append(current_conversation)
            current_conversation = []
        else:
            # Add the sentence to the current conversation
            current_conversation.append(sentence)
    
    # Create the training data
    train_data = []
    for conversation in conversations:
        for i in range(len(conversation) - 1):
            # The input is a sequence of words from the current sentence and the previous sentence
            input_sequence = conversation[i] + " " + conversation[i-1]
            # The target is the next sentence
            target_sequence = conversation[i+1]
            # Add the input-target pair to the training data
            train_data.append((input_sequence, target_sequence))
    
    return train_data


# Create the vocabulary by extracting the unique words from the training data
def create_vocabulary(train_data):
    vocabulary = set()
    for input_sequence, target_sequence in train_data:
        for word in input_sequence.split(" "):
            vocabulary.add(word)
        for word in target_sequence.split(" "):
            vocabulary.add(word)
    print("Vocabulary size:", len(vocabulary))