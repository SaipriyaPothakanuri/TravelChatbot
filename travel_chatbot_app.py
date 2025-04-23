import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch

# Sample Q&A knowledge base for Tourism in Spain
qa_pairs = [
    {"question": "What are the top tourist destinations in Spain?", 
     "answer": "Spain offers diverse attractions including Barcelona, Madrid, Seville, Valencia, Granada, and Ibiza."},
    {"question": "Do I need a visa to visit Spain?", 
     "answer": "If you're from outside the EU/EEA/Schengen area, you may need a short-stay Schengen visa."},
    {"question": "What is the best time to visit Spain?", 
     "answer": "Spring (April to June) and Fall (September to November) are ideal due to pleasant weather and fewer crowds."},
    {"question": "Are there guided tours available in Spain?", 
     "answer": "Yes, many cities offer guided walking tours, food tours, cultural experiences, and day trips."},
    {"question": "How can I travel within Spain?", 
     "answer": "Spain has a well-connected transport network including high-speed trains (AVE), buses, and domestic flights."},
    {"question": "What cultural events should I attend?", 
     "answer": "Popular events include La Tomatina, Running of the Bulls, Semana Santa, and Flamenco shows."},
    {"question": "What should I try from Spanish cuisine?", 
     "answer": "Don't miss out on tapas, paella, churros, tortilla española, and sangria!"}
]

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()
questions = [pair["question"] for pair in qa_pairs]
question_embeddings = model.encode(questions, convert_to_tensor=True)

# Streamlit UI
st.title("Spain Tourism Chatbot")
st.info("""  
Welcome to the Spain Tourism Chatbot!  
Ask me anything about visiting Spain, such as:

- What are the top tourist destinations?
- Do I need a visa to visit Spain?
- What’s the best time to travel?
- Are guided tours available?
- How do I travel between cities?
- What local events can I experience?
- What food should I try?

We can customize this chatbot with your travel agency’s specific offerings and locations!
""")

with st.form(key="chat_form"):
    user_input = st.text_input("You:", "", placeholder="Type your travel question here...")
    submit_button = st.form_submit_button(label="Ask")

if submit_button and user_input.strip():
    query_embedding = model.encode(user_input, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, question_embeddings)[0]
    best_match_idx = torch.argmax(cos_scores).item()
    best_answer = qa_pairs[best_match_idx]["answer"]
    st.markdown(f"**Chatbot response:** {best_answer}")

elif not submit_button:
    st.markdown("👋 Hello! I'm your travel assistant for exploring Spain. Ask me a question!")
