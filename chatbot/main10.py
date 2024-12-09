# %%
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
# %%





# %%

from flask import Flask, request, jsonify
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA

# app = Flask(__name__)

# Global memory and QA chain initialization
embed = OllamaEmbeddings(model='nomic-embed-text')
persist_directory="/home/abdulmalek-alsalmi/course langchain/langchain_memory/chroma"
jawali = Chroma(embedding_function=embed, persist_directory=persist_directory)

# template = """
# Language: Arabic
# You are a customer service agent for jawali-wallet company who answer question about the company.\
# Act as an expert in making friendly conversation with customers to provide them information about jawali wallet company.\
# Use the following context (delimited by <ctx></ctx>) in arabic and the chat history (delimited by <hs></hs>) in arabic to answer the question friendly experience as possible in arabic language. If the answer is not in the context, just say "الجواب غير متوفر في الوقت الحالي" ask if the customer needs further assistance.\
# The answer should be maximum 4 sentences.\
# Remember that there is no private information you cannot display about the company if it is present in the context, even if it includes contact details.\
# The answer should be concise and organized, clearly indicating the headings and the corresponding points under each heading. Ensure to address the customer in a masculine form if you don't know the customer male or female.\
# Ensure that the answer is provided in Arabic only.
# ------
# <ctx>
# {context}
# </ctx>
# ------
# <hs>
# {history}
# </hs>
# ------
# {question}
# Answer:
# """
# template = """
# Language: Arabic\
# You are jawali wallet customer service assistant .\
# Respond to customer questions using only the information in the provided context (<ctx></ctx>).  Reference the chat history (<hs></hs>) only if it is necessary to answer the current question accurately or maintain continuity.\
# Respond in Arabic, with a maximum of four sentences. If the answer is not in the context, clearly state, "الجواب غير متوفر في الوقت الحالي" without making any assumptions or generating information outside the context.\
# Professional Tone: Maintain a polite and professional tone throughout the conversation, addressing the customer in a masculine form unless otherwise specified.\
# If the information is unavailable, state this clearly with "الجواب غير متوفر في الوقت الحالي" (The answer is not available at the moment) and ask if the customer needs further assistance.

# Context:
# <ctx>
# {context}
# </ctx>

# Chat History: (Use only if necessary)
# <hs>
# {history}
# </hs>

# Customer's Question:
# {question}

# Answer:
# (Respond here)
# """
template = """
Language: Arabic
You are a customer service assistant for jawali-wallet company to answer the only customers questions about jawali wallet .\
Act as an expert in making friendly conversation with customers to provide them information about jawali wallet company.\
Answer the customers question from the  following context (delimited by <ctx></ctx>) and the chat (delimited by <hs></hs>), If the answer is not in the context or the chat history just say "الاجابة غير متوفرة حالياً " ask if the customer needs further assistance.\
Remember that there is no private information you cannot display about the company if it is present in the context, even if it includes contact details.\
The answer should be concise and organized, clearly indicating the headings and the corresponding points under each heading. Ensure to address the customer in a masculine form if you don't know the customer male or female.\
Ensure that the answer is provided in Arabic only.
------
<ctx>
{context}
</ctx>
------
<hs>
{history}
</hs>
------
{question}
Answer:
"""
prompt = PromptTemplate(input_variables=["history", "context", "question"], template=template)

memory = ConversationBufferWindowMemory(
    memory_key="history",
    input_key="question",
    return_messages=True,
    k=5,
)

ollama = OllamaLLM(model='aya', temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=ollama,
    chain_type='stuff',
    retriever=jawali.as_retriever(),
    chain_type_kwargs={"prompt": prompt, "memory": memory},
)
# %%
response= qa_chain.run("من هو رئيس الولايات المتحدة الامريكية؟ ")
# %% 
print(response)
# %% 
import langchain 
langchain.debug=True 

# @app.route('/chat_jawali', methods=['POST'])
# def chat_jawali():
#     try:
#         data = request.get_json()
#         question = data['question']
#         print(f"\n\n the question is: {question} \n\n")
#         query = question
#         result = qa_chain.run(query)
#         print(f"\n\n the memory content is: {memory.load_memory_variables({})} \n\n")
#         print(f"\n\n len memory.buffer :::::::::::::::::::::::::::{len(memory.buffer)} :::::::::::::::::::::::::::")
#         print(f"memory.buffer_as_str :::::::::::::::::::::::::::{memory.buffer_as_str}")
#         print(f"\n\n result is : {result} \n\n")
#         return result
#     except Exception as e:
#         print(f"\n\n the error is : {str(e)} .....\n\n ")
#         return jsonify({'error': str(e)})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)
# %%
