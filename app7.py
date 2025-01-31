import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import faiss
import numpy as np
import bs4

# Initialize Streamlit page configuration
st.set_page_config(page_title="Simple Q&A App", page_icon=":guardsman:", layout="wide")

# Add a title to the Streamlit app
st.title("Simple Q&A App with LangChain")

# Step 1: Get the Groq API key from the user
groq_api_key = st.text_input("Enter your Groq API Key:")

# Step 2: Get the web URL from the user
web_url = st.text_input("Enter the URL of the article or content:")

# If both Groq API key and URL are provided, proceed
if groq_api_key and web_url:
    # Add 'https://' to the URL if it's not provided
    if not web_url.lower().startswith(('http://', 'https://')):
        web_url = 'https://' + web_url
    
    # Load the page content from the URL
    loader = WebBaseLoader(
        web_paths=(web_url,),
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_="reflist"))
    )
    docs = loader.load()

    # Debug: Print loaded documents to check content
    st.write(f"Loaded documents: {len(docs)}")
    for doc in docs[:3]:  # Print first 3 documents for debugging
        st.write(doc.page_content[:500])  # Display first 500 characters of each document

    # Check if the content is loaded properly
    if len(docs) > 0:
        document_content = docs[0].page_content  # Access the content of the first document
        st.write(f"Document content preview: {document_content[:500]}")  # Show the first 500 characters

    # Split the content into chunks (adjusting chunk size for smaller documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)  # Adjusted chunk size
    splits = text_splitter.split_documents(docs)

    # Debug: Print out the number of splits
    st.write(f"Number of splits: {len(splits)}")
    for split in splits[:3]:  # Display first 3 splits for debugging
        st.write(split.page_content[:500])  # Show first 500 characters of the splits

    # If no splits were generated, display an error
    if len(splits) == 0:
        st.error("No splits generated. The document might be too short to split into meaningful chunks.")
    else:
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Generate embeddings for the documents (this returns a list of embeddings)
        try:
            embeddings_list = embeddings.embed_documents([doc.page_content for doc in splits])
            
            # Debug: Check if embeddings are being generated
            if not embeddings_list:
                st.error("Embeddings list is empty. Please check your embeddings generation.")
            else:
                # Debug: Check the first embedding
                st.write(f"First embedding: {embeddings_list[0]}")
                st.write(f"Total number of embeddings: {len(embeddings_list)}")

                # Convert embeddings list to numpy array (2D array)
                embeddings_array = np.array(embeddings_list).astype("float32")
                st.write(f"Embeddings array shape: {embeddings_array.shape}")
                
                # Check that the array has the correct shape (2D)
                if len(embeddings_array.shape) == 2:  # 2D array: (num_docs, embedding_dim)
                    st.success("Embeddings are in the correct format!")
                else:
                    st.error("Embeddings are not in the correct 2D format!")
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")

    # Now create a FAISS index from the embeddings
    if 'embeddings_array' in locals() and len(embeddings_array.shape) == 2:
        # Initialize FAISS index (L2 distance)
        index = faiss.IndexFlatL2(embeddings_array.shape[1])  # FAISS expects the embedding dimension here
        index.add(embeddings_array)  # Add the embeddings to the FAISS index

        # Create a docstore (a dictionary to map document index to original documents)
        docstore = {i: splits[i].page_content for i in range(len(splits))}
        
        # Create a mapping from index to docstore ID (simply use the index as the ID here)
        index_to_docstore_id = {i: i for i in range(len(splits))}

        # Wrap FAISS in LangChain vectorstore
        vectorstore = FAISS(index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id, embedding_function=embeddings)

        retriever = vectorstore.as_retriever()

        # Initialize the Groq model
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
        
        # Create a simple retrieval chain for Q&A
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Step 3: Initialize chat history in session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Step 4: Display previous chat history
        if len(st.session_state.chat_history) > 0:
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                st.markdown(f"**Q{i+1}:** {question}")
                st.markdown(f"**A{i+1}:** {answer}")

        # Step 5: User input box for asking a question
        user_input = st.text_input("Ask a question:")

        # If a question is asked, process the input and generate an answer
        if user_input:
            # Retrieve the answer using the RAG chain
            response = rag_chain.invoke({"input": user_input, "chat_history": st.session_state.chat_history})
            answer = response["answer"]

            # Add the question and answer to the chat history
            st.session_state.chat_history.append((user_input, answer))

            # Display the new answer
            st.markdown(f"**Answer:** {answer}")
else:
    if not groq_api_key:
        st.warning("Please enter your Groq API key to get started.")
    if not web_url:
        st.warning("Please enter the URL of the content to load.")
