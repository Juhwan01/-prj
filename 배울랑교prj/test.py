from dotenv import load_dotenv

load_dotenv()

from rag.pdf import PDFRetrievalChain

pdf = PDFRetrievalChain(["test1.pdf"]).create_chain()
pdf_retriever = pdf.retriever
pdf_chain = pdf.chain
from typing import TypedDict



# GraphState 상태를 저장하는 용도로 사용합니다.
class GraphState(TypedDict):
    question: str  # 질문
    context: str  # 문서의 검색 결과
    answer: str  # 답변
    relevance: str  # 답변의 문서에 대한 관련성
    text: str # 본문
    
from langchain_upstage import UpstageGroundednessCheck
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from rag.utils import *
from langchain import hub

# 업스테이지 문서 관련성 체크 기능을 설정합니다. https://upstage.ai
upstage_ground_checker = UpstageGroundednessCheck()

qchain=qChain()
# 문서에서 검색하여 관련성 있는 문서를 찾습니다.
def retrieve_document(state: GraphState) -> GraphState:
    # 문서에서 검색하여 관련성 있는 문서를 찾습니다.
    retrieved_docs = pdf_retriever.invoke(state["question"])

    # 검색된 문서를 형식화합니다.
    retrieved_docs = format_docs(retrieved_docs)

    # 검색된 문서를 context 키에 저장합니다.
    return GraphState(context=retrieved_docs)





# LLM을 사용하여 답변을 생성합니다.
def llm_answer(state: GraphState) -> GraphState:
    question = state["question"]
    context = state["context"]
    
     # Question rewriting model
    model = ChatOpenAI(temperature=0, model="gpt-4o-2024-08-06")
    prompt = hub.pull("lings/answer-generator")
    chain = prompt | model | StrOutputParser()
    response = chain.invoke(
        {"question": question, "examContext": context}
    )
    print(state["question"])
    print(response)
    return GraphState(answer=response)


# LLM을 사용하여 답변 생성 가능성 체크
def docChecker(state: GraphState) -> GraphState:
    question = state["question"]
    context = state["context"]
    
    # 체인을 호출하여 답변을 생성합니다.
    response = pdf_chain.invoke({"question": question, "context": context})
    return GraphState(relevance=response)


def qMaker(state):
    response = qchain.invoke(
        {"num_questions": 1,"context_str": state["text"]}
    )
    return GraphState(question=response)
    

def rewrite(state):
    question = state["question"]
    answer = state["answer"]
    context = state["context"]
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a professional prompt rewriter. Your task is to generate the question in order to get additional information that is now shown in the context."
                "Your generated question will be searched on the web to find relevant information.",
            ),
            (
                "human",
                "Rewrite the question to get additional information to get the answer."
                "\n\nHere is the initial question:\n ------- \n{question}\n ------- \n"
                "\n\nHere is the initial context:\n ------- \n{context}\n ------- \n"
                "\n\nHere is the initial answer to the question:\n ------- \n{answer}\n ------- \n"
                "\n\nFormulate an improved question in Korean:",
            ),
        ]
    )

    # Question rewriting model
    model = ChatOpenAI(temperature=0, model="gpt-4o-2024-08-06")

    chain = prompt | model | StrOutputParser()
    response = chain.invoke(
        {"question": question, "answer": answer, "context": context}
    )
    return GraphState(question=response)


# def search_on_web(state: GraphState) -> GraphState:
#     # 문서에서 검색하여 관련성 있는 문서를 찾습니다.
#     search_tool = TavilySearchResults(max_results=5)
#     search_result = search_tool.invoke({"query": state["question"]})

#     # 검색된 문서를 형식화합니다.
#     search_result = format_searched_docs(search_result)
#     # 검색된 문서를 context 키에 저장합니다.
#     return GraphState(
#         context=search_result,
#     )


def relevance_check(state: GraphState) -> GraphState:
    # 관련성 체크를 실행합니다. 결과: grounded, notGrounded, notSure
    response = upstage_ground_checker.run(
        {"context": state["context"], "answer": state["answer"]}
    )
    return GraphState(
        relevance=response, question=state["question"], answer=state["answer"]
    )


def is_relevant(state: GraphState) -> GraphState:
    return state["relevance"]


from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# langgraph.graph에서 StateGraph와 END를 가져옵니다.
workflow = StateGraph(GraphState)

# 노드들을 정의합니다.
workflow.add_node("retrieve", retrieve_document)  # 에이전트 노드를 추가합니다.
workflow.add_node("llm_answer", llm_answer)  # 정보 검색 노드를 추가합니다.
workflow.add_node(
    "relevance_check", relevance_check
)  # 답변의 문서에 대한 관련성 체크 노드를 추가합니다.\
workflow.add_node("doccheck",docChecker) # 답변 생성 가능성 검사
workflow.add_node("rewrite", rewrite)  # 질문을 재작성하는 노드를 추가합니다.
# workflow.add_node("search_on_web", search_on_web)  # 웹 검색 노드를 추가합니다.
# workflow.add_node("requestion", resetQ)
workflow.add_node("qmaker",qMaker)





# 각 노드들을 연결합니다.
workflow.add_edge("retrieve", "doccheck")  # 검색 -> 답변
workflow.add_edge("llm_answer", "relevance_check")  # 답변 -> 관련성 체크
workflow.add_edge("rewrite", "retrieve")  # 재작성 -> 관련성 체크
# workflow.add_edge("search_on_web", "llm_answer")  # 웹 검색 -> 답변
workflow.add_edge("qmaker", "retrieve")  # 웹 검색 -> 답변

# 조건부 엣지를 추가합니다.
workflow.add_conditional_edges(
    "doccheck",  # 관련성 체크 노드에서 나온 결과를 is_relevant 함수에 전달합니다.
    is_relevant,
    {
        "yes": "llm_answer",  # 생성 가능하다면
        "no": "retrieve",  # 생성 불가능 하다면
    },
)
# 조건부 엣지를 추가합니다.
workflow.add_conditional_edges(
    "relevance_check",  # 관련성 체크 노드에서 나온 결과를 is_relevant 함수에 전달합니다.
    is_relevant,
    {
        "grounded": END,  # 관련성이 있으면 종료합니다.
        "notGrounded": "rewrite",  # 관련성이 없으면 다시 답변을 생성합니다.
        "notSure": "rewrite",  # 관련성 체크 결과가 모호하다면 다시 답변을 생성합니다.
    },
)


workflow.set_entry_point("qmaker")

memory = MemorySaver()

app = workflow.compile(checkpointer=memory)

from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    recursion_limit=100, configurable={"thread_id": "CORRECTIVE-SEARCH-RAG"}
)
inputs = GraphState(
    text=load_doc("test1.pdf")
)
app.invoke(inputs,config=config)