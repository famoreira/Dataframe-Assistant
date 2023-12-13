import streamlit as st
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

# Function to load data from CSV
def load_data_from_csv(url):
    return pd.read_csv(url)

# Function to extract text from the dataframe
def extract_text_from_dataframe(df):
    return " ".join(df.astype(str).sum())

# Function to split text into chunks
def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    return text_splitter.split_text(text=text)

# Function to embed text chunks and create a vector store
def create_vector_store(chunks):
    embedding = OpenAIEmbeddings()
    return FAISS.from_texts(chunks, embedding=embedding)

# Function to run a specific agent command
def run_agent_command(agent, command):
    return agent.run(command)

# Main application
def main():
    load_dotenv()
    st.header("Dataframe Assistant")

    # Load data from CSV file
    url_data = "URL"
    df = load_data_from_csv(url_data)
    df_text = extract_text_from_dataframe(df)
    df_chunks = split_text_into_chunks(df_text)
    vectorstore = create_vector_store(df_chunks)

    # Create the agent
    llm = OpenAI(model_name="gpt-3.5-turbo")
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    agent = create_pandas_dataframe_agent(llm, df, chain_type="stuff", verbose=True)

    # Pre-screened sentences or commands
    preset_commands = [        
    	"Query 1",
        "Query 2",
        # Add more pre-screened sentences here if needed
    ]

    #st.subheader("Commands")
    command_choice = st.selectbox("Choose a query:", ["Custom"] + preset_commands)

    # Custom command input
    if command_choice == "Custom":
        custom_command = st.text_input("Enter a query:")
        if custom_command:
            response = run_agent_command(agent, custom_command)
            st.write(response)
    else:
        # Running a pre-screened command
        if st.button("Run Command"):
            response = run_agent_command(agent, command_choice)
            st.write(response)

if __name__ == "__main__":
    main()
