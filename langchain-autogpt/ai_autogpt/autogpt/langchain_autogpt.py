from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
from langchain.embeddings import OpenAIEmbeddings
import faiss
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain_experimental.autonomous_agents import AutoGPT
from langchain.chat_models import ChatOpenAI


class LangchainAutogpt:
    def __init__(self):
        # 构造 AutoGPT 的工具集
        self.search = SerpAPIWrapper()
        tools = [
            Tool(
                name="search",
                func=self.search.run,
                description="useful for when you need to answer questions about current events. You should ask targeted questions",
            ),
            WriteFileTool(),
            ReadFileTool(),
        ]
        # OpenAI Embedding 模型
        self.embeddings_model = OpenAIEmbeddings()
        # OpenAI Embedding 向量维数
        embedding_size = 1536
        # 使用 Faiss 的 IndexFlatL2 索引
        index = faiss.IndexFlatL2(embedding_size)
        # 实例化 Faiss 向量数据库
        self.vectorstore = FAISS(self.embeddings_model.embed_query, index, InMemoryDocstore({}), {})
        self.agent = AutoGPT.from_llm_and_tools(
            ai_name="Jarvis",
            ai_role="Assistant",
            tools=tools,
            llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
            memory=self.vectorstore.as_retriever(),  # 实例化 Faiss 的 VectorStoreRetriever
        )

    def invoke(self,
                    input_text: str):
        self.agent.run([input_text])