from dotenv import load_dotenv

load_dotenv()

from rag.pdf import PDFRetrievalChain

pdf = PDFRetrievalChain(["배울랑교prj/testfile.pdf"]).create_chain()
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
    generated_question: str  # 생성된 문제
    generated_answer: str  # 생성된 답변
    verification: str  # 검증 결과
    
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
    
    # 체인을 호출하여 답변을 생성합니다.
    response = pdf_chain.invoke({"question": question, "context": context})
    print(response)
    return GraphState(answer=response)

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
    model = ChatOpenAI(temperature=0, model="gpt-4")

    chain = prompt | model | StrOutputParser()
    response = chain.invoke(
        {"question": question, "answer": answer, "context": context}
    )
    return GraphState(question=response)


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

# 새로운 함수: PDF 내용을 기반으로 문제 생성
def generate_question(state: GraphState) -> GraphState:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a professional question generator. Your task is to create a challenging question based on the given text.",
            ),
            (
                "human",
                "Generate a question based on the following text:\n\n{text}\n\nQuestion:",
            ),
        ]
    )
    model = ChatOpenAI(temperature=0.7)
    chain = prompt | model | StrOutputParser()
    generated_question = chain.invoke({"text": state["text"]})
    return GraphState(generated_question=generated_question, text=state["text"])

# 수정된 함수: 생성된 문제에 대한 간결한 답변 생성
def generate_answer(state: GraphState) -> GraphState:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert in answering questions concisely. Your task is to provide a brief and accurate answer to the given question based on the provided text. Keep your answer under 50 words.",
            ),
            (
                "human",
                "Text: {text}\n\nQuestion: {question}\n\nProvide a brief answer:",
            ),
        ]
    )
    model = ChatOpenAI(temperature=0)
    chain = prompt | model | StrOutputParser()
    generated_answer = chain.invoke({"text": state["text"], "question": state["generated_question"]})
    return GraphState(generated_answer=generated_answer, generated_question=state["generated_question"], text=state["text"])

# 새로운 함수: 생성된 답변 검증
def verify_answer(state: GraphState) -> GraphState:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a fact-checker. Your task is to verify the accuracy of the given answer based on the provided text. Provide a concise verification, focusing only on the accuracy of the answer.",
            ),
            (
                "human",
                "Text: {text}\n\nQuestion: {question}\n\nAnswer: {answer}\n\nProvide a brief verification of the answer's accuracy:",
            ),
        ]
    )
    model = ChatOpenAI(temperature=0)
    chain = prompt | model | StrOutputParser()
    verification = chain.invoke({"text": state["text"], "question": state["generated_question"], "answer": state["generated_answer"]})
    return GraphState(verification=verification, generated_question=state["generated_question"], generated_answer=state["generated_answer"], text=state["text"])

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# langgraph.graph에서 StateGraph와 END를 가져옵니다.
workflow = StateGraph(GraphState)

# 노드들을 정의합니다.
workflow.add_node("retrieve", retrieve_document)
workflow.add_node("llm_answer", llm_answer)
workflow.add_node("relevance_check", relevance_check)
workflow.add_node("rewrite", rewrite)
workflow.add_node("qmaker", qMaker)
workflow.add_node("generate_question", generate_question)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("verify_answer", verify_answer)

# 각 노드들을 연결합니다.
workflow.add_edge("retrieve", "llm_answer")
workflow.add_edge("llm_answer", "relevance_check")
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("qmaker", "retrieve")
workflow.add_edge("generate_question", "generate_answer")
workflow.add_edge("generate_answer", "verify_answer")
workflow.add_edge("verify_answer", END)

# 조건부 엣지를 추가합니다.
workflow.add_conditional_edges(
    "relevance_check",
    is_relevant,
    {
        "grounded": "generate_question",
        "notGrounded": "rewrite",
        "notSure": "rewrite",
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
    text=load_doc("배울랑교prj/testfile.pdf")
)
result = app.invoke(inputs, config=config)

print("질문 : " + result.get("generated_question", "질문을 생성하지 못했습니다."))
print("\n답변 : " + result.get("generated_answer", "답변을 생성하지 못했습니다."))
print("\n검증 : " + result.get("verification", "검증을 수행하지 못했습니다."))