import streamlit as st
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5Tokenizer
import re
import random



# Load pre-trained T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Function to generate MCQs
def generate_mcqs(model, tokenizer, tokens):
    outputs = model.generate(tokens, max_length=150)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Parse and format MCQs from decoded_output
    mcqs = []
    questions = re.split(r'Question |Answer:', decoded_output)[1:]

    for i in range(0, len(questions), 2):
        question = questions[i]
        answer = questions[i+1].strip()

        # Generate options
        options = [answer]
        for _ in range(3):
            option = random.choice(questions[(i+1)%len(questions)])
            options.append(option.strip())

        # Shuffle options
        random.shuffle(options)

        # Find correct answer index
        correct_answer_index = options.index(answer)

        # Format MCQ
        mcq = {
            "question": question,
            "options": options,
            "correct_answer": chr(65 + correct_answer_index)  # A, B, C, D
        }

        mcqs.append(mcq)

    return mcqs

# Function to generate article
def generate_article(model, tokenizer, tokens):
    outputs = model.generate(tokens, max_length=300)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output

# Streamlit app
st.title("AI Chat History Analyzer")

# Input chat history
st.header("Enter Chat History")
chat_history = st.text_area("Chat History", height=200)

if st.button("Generate MCQs"):
    if chat_history:
        # Preprocess chat history
        input_text = " ".join(chat_history.splitlines())
        tokens = tokenizer.encode(input_text, return_tensors='pt')

        # Generate MCQs
        mcqs = generate_mcqs(model, tokenizer, tokens)

        # Display MCQs
        st.header("Generated MCQs")
        for i, mcq in enumerate(mcqs):
            st.write(f"Question {i+1}: {mcq['question']}")
            for j, option in enumerate(mcq['options']):
                st.write(f"{chr(65 + j)}. {option}")
            st.write(f"Correct Answer: {mcq['correct_answer']}")
            st.write("---")
    else:
        st.warning("Please enter a chat history.")

if st.button("Generate Article"):
    if chat_history:
        # Preprocess chat history
        input_text = " ".join(chat_history.splitlines())
        tokens = tokenizer.encode(input_text, return_tensors='pt')

        # Generate article
        article = generate_article(model, tokenizer, tokens)

        # Display article
        st.header("Generated Article")
        st.write(article)
    else:
        st.warning("Please enter a chat history.")
