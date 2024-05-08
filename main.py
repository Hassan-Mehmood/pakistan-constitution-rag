import streamlit as st
from langchain.text_splitter  import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

def main():
	st.title("Pakistan Constitution RAG")
	st.subheader("Ask me questions about Pakistan Constitution.")

	with st.spinner('Loading documents...'):
		loader = PyPDFLoader("data/data.pdf")
		documents = loader.load()
	
	text_splitter = CharacterTextSplitter(
		separator="\n",
		chunk_size=1000,
		chunk_overlap=200,
		length_function=len
	)

	with st.spinner('Splitting text chunks...'):
		texts = text_splitter.split_documents(documents)

	with st.spinner('Creating Embedding object...'):
		embeddings = HuggingFaceInstructEmbeddings(
			model_name="hkunlp/instructor-xl",
			model_kwargs={"device": "cuda"},
		)

	with st.spinner('Embedding documents...'):
		search = embeddings.embed_documents(texts)

	st.subheader("Search:")


if __name__ == '__main__':
	main()