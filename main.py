import streamlit as st
from langchain.text_splitter  import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

def main():
	st.title("Pakistan Constitution RAG")
	st.subheader("Ask me questions about Pakistan Constitution.")

	loader = PyPDFLoader("data/data.pdf")
	documents = loader.load()
	
	text_splitter = CharacterTextSplitter(
		separator="\n",
		chunk_size=1000,
		chunk_overlap=200,
		length_function=len
	)

	texts = text_splitter.split_documents(documents)

	st.write(texts)

if __name__ == '__main__':
	main()