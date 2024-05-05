def main():
	from langchain_community.llms import HuggingFaceEndpoint
	from langchain.chains import LLMChain
	from langchain_core.prompts import ChatPromptTemplate
	
	repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

	prompt = ChatPromptTemplate.from_messages([
		("system", "You are a chef who makes delicious food. Provide a detailed instruction on how to cook a dish named {dish}."),
		])

	llm = HuggingFaceEndpoint(
		repo_id=repo_id, temperature=0.5
		)
	
	chain = prompt | llm

	print(chain.invoke({
		"dish": "chicken curry",
	}))

	
if __name__ == '__main__':
	main()