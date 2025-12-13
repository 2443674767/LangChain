from langchain_ollama import OllamaEmbeddings  # 引入新的 OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import RetrievalQA

from langchain_core.prompts import PromptTemplate

# 创建 Ollama 嵌入模型实例
ollama_embedding = OllamaEmbeddings(model="deepseek-r1:32b")  # 使用合适的模型名

# 示例文档
documents = [
    "Ollama is a framework for large language models.",
    "LangChain integrates with different LLMs for various NLP tasks.",
    "FAISS is a library for efficient similarity search and clustering of dense vectors."
]

# 使用 Ollama 嵌入模型生成文档的嵌入
document_embeddings = ollama_embedding.embed_documents(documents)

# 创建 FAISS 向量存储
faiss_store = FAISS.from_texts(documents, ollama_embedding)

# 创建查询模板
query_prompt = PromptTemplate(
    input_variables=["query"],
    template="Find the most relevant document for the following query: {query}"
)

# 设置用于检索的检索链
retriever = faiss_store.as_retriever()

# 创建问答链
qa_chain = RetrievalQA.from_chain_type(
    llm=ollama_embedding,
    retriever=retriever,
    return_source_documents=True
)

# 测试查询
query = "What is Ollama?"
response = qa_chain.run(query)

print(response)
