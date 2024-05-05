from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter  import CharacterTextSplitter
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader

def main():

	loader = PyPDFLoader("data/data.pdf")
	documents = loader.load()

	st.title("Pakistan Constitution RAG")
	st.subheader("Ask me questions about Pakistan Constitution.")

	text_splitter = CharacterTextSplitter(
		separator="\n",
		chunk_size=1000,
		chunk_overlap=200,
		length_function=len
	)

	texts = text_splitter.split_documents(documents)

	st.write(texts)




	# repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

	# prompt = ChatPromptTemplate.from_messages([
	# 	("system", "You are an assistant that will answer questions provided by the user."),
	# 	])

	# llm = HuggingFaceEndpoint(
	# 	repo_id=repo_id, temperature=0.5
	# 	)
	
	# chain = prompt | llm

	# print(chain.invoke({
	# 	"dish": "chicken curry",
	# }))

	
if __name__ == '__main__':
	main()