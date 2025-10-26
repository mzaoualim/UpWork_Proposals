import streamlit as st
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import pandas as pd
from query_engine import llm_to_mongo

# Load secrets
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")

@st.cache_resource
def get_mongo_client():
    return MongoClient(MONGO_URI)

def run_pipeline(pipeline):
    client = get_mongo_client()
    db = client[MONGO_DB]
    collection = db["transactions"]
    return list(collection.aggregate(pipeline))

st.set_page_config(page_title="TinyLlama MongoDB Chat", layout="wide")
st.title("🦙 TinyLlama × MongoDB Chat Interface")

st.markdown("Type a question in plain English (e.g. *Average house price in Sydney in 2024*)")

query = st.text_area("Your query")

if st.button("Run"):
    with st.spinner("Interpreting your query..."):
        pipeline = llm_to_mongo(query)
    st.subheader("🔧 Generated MongoDB Pipeline")
    st.code(pipeline, language="python")

    with st.spinner("Running query on MongoDB..."):
        try:
            results = run_pipeline(pipeline)
            if results:
                df = pd.DataFrame(results)
                st.dataframe(df)
            else:
                st.warning("No results found.")
        except Exception as e:
            st.error(f"MongoDB error: {e}")
