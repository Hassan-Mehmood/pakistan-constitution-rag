def main():
	from langchain_community.llms import HuggingFaceEndpoint
	from langchain_core.prompts import ChatPromptTemplate
	import streamlit as st

	from langchain_community.document_loaders import PyPDFLoader

	loader = PyPDFLoader("data/data.pdf")
	pages = loader.load_and_split()

	st.title("Pakistan Constitution RAG")
	st.subheader("Ask me questions about Pakistan Constitution.")

	st.write(pages)



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