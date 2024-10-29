import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')  # Adjust the path if necessary

# Function to generate structured answer
def generate_structured_answer(question, context):
    input_text = f"question: {question} context: {context}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

    outputs = model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    structured_answer = "\n".join(["- " + sentence.strip() for sentence in answer.split(".") if sentence])
    return structured_answer

# Function to generate article
def generate_article(prompt):
    input_text = f"write an article with sections (introduction, body, conclusion) on: {prompt}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

    outputs = model.generate(input_ids, max_length=1000, num_beams=4, early_stopping=True)
    article = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return article

# Function to generate questions from passage
def generate_questions_from_passage(passage):
    input_text = f"generate question from: {passage}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

    outputs = model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return question

# Streamlit app
st.title("AI Text Generation App")

# Sidebar for navigation
option = st.sidebar.selectbox(
    'Select a task',
    ('Generate Structured Answer', 'Generate Article', 'Generate Questions from Passage')
)

if option == 'Generate Structured Answer':
    st.subheader("Generate Structured Answer")
    question = st.text_input("Enter your question:")
    context = st.text_area("Enter the context:")
    if st.button("Generate Answer"):
        answer = generate_structured_answer(question, context)
        st.write("### Answer:")
        st.write(answer)

elif option == 'Generate Article':
    st.subheader("Generate Article")
    prompt = st.text_input("Enter the topic for the article:")
    if st.button("Generate Article"):
        article = generate_article(prompt)
        st.write("### Article:")
        st.write(article)

elif option == 'Generate Questions from Passage':
    st.subheader("Generate Questions from Passage")
    passage = st.text_area("Enter the passage:")
    if st.button("Generate Questions"):
        questions = generate_questions_from_passage(passage)
        st.write("### Generated Questions:")
        st.write(questions)
