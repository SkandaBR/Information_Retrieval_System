import streamlit as st
from src.helper import get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain, list_available_models

def user_input(user_question):
    if st.session_state.conversation is None:
        st.error("Please upload and process some documents first!")
        return
        
    # Get the response from the conversation object using the question
    response = st.session_state.conversation({'question': user_question})
    
    # Update chat history in the session state
    st.session_state.chatHistory = response['chat_history']

    # Display chat history with alternating labels for user and AI replies
    for i, message in enumerate(st.session_state.chatHistory):
        if i % 2 == 0:
            st.write("User: ", message.content)
        else:
            st.write("Reply: ", message.content)


def main():
    st.set_page_config("Information Retrieval")
    st.header("Information-Retrieval-System 🧠")

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = []
    if "user_question" not in st.session_state:
        st.session_state.user_question = ""

    with st.sidebar:
        st.title("Menu:")
        
        if st.button("List Available Models"):
            list_available_models()
            
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", 
            accept_multiple_files=True
        )

        if st.button("Submit & Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file!")
                return
                
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectore_store = get_vector_store(text_chunks)
                st.session_state.conversation = get_conversational_chain(vectore_store)
                st.success("Done")

    user_question = st.text_input(
        "Ask a question about the uploaded documents:", 
        key="user_question",
        placeholder="Type your question here..."
    )

    # Process the user's question if it exists
    if user_question:
        user_input(user_question)


if __name__ == "__main__":
    main()
