from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough,RunnableLambda

import os

def get_llm():
    return ChatMistralAI(model = "mistral-small-latest",mistral_api_key=os.getenv("MISTRAL_API_KEY"),temperature=0.3)

def split_transcript(trancript:str)->list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000,chunk_overlap=200)
    chunks = splitter.split_text(trancript)
    return chunks

def summarize(trnascript:str)->str:
    llm = get_llm()
    map_promt = ChatPromptTemplate.from_messages(
        [
            ("system", "Summraize this portion of a meeting transcript concisely."),
            ("human", "{text}"),
        ]
    )
    map_chain = map_promt | llm | StrOutputParser()
    chunks = split_transcript(trnascript)
    chunk_summaries = [map_chain.invoke({"text": chunk}) for chunk in chunks]
    combined = "\n\n".join(chunk_summaries)
    combined_promt = ChatPromptTemplate.from_messages(
        [
            (
                "system", 
                "You are an expert summarizer. Combine these partial summaries"
                " into a final professional summary in bullet points."),
            (
                "human", 
                "{text}"),
        ]
    )
    combined_chain = (RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | combined_promt | llm | StrOutputParser())
    
    return combined_chain.invoke(combined)

def generate_title(transcript:str)->str:
    llm = get_llm()
    
    title_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) |
        ChatPromptTemplate.from_messages(
            [
                (
                    "system", 
                    "Based on the meeting transcript, generate a short professional meeting title"
                    "(max 8 words). Only return the title, nothing else."
                ),
                ("human", "{text}"),
            ]
        )
        | llm 
        | StrOutputParser()
    )
    return title_chain.invoke(transcript[:2000])