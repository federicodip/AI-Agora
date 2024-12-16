import os
import shutil
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import gradio as gr
import PyPDF2

# Load the text from the PDF file
with open("transcript.pdf", "rb") as file:
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

#print(text)

# Split the text into smaller chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=350,
    chunk_overlap=40,
    separators=["\n\n", "\n", ".", ",", "CAPUT"]
)
splits = splitter.split_text(text)
print(f"Split into {len(splits)} chunks.")

# Create the vector database
embedding = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
persist_directory = './docs/chroma'

if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)

vectordb = Chroma.from_texts(
    texts=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

print(vectordb._collection.count())

# Define the language model and retrieval chain
llm = ChatOpenAI(model_name='gpt-4', temperature=0.15)

template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. 
{context}
Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

# Function to generate a response from the RAG model with sources
def chatbot_response_with_sources(question):
    result = qa_chain({"query": question})
    answer = result.get("result", "No response available.")
    source_docs = result.get("source_documents", [])
    sources = "\n\n".join([f"- {doc.page_content}" for doc in source_docs])
    return f"Answer:\n{answer}\n\nSources:\n{sources}"

# Set up the Gradio interface
iface = gr.Interface(
    fn=chatbot_response_with_sources,  # Use the updated function
    inputs="text",                     # Input type: text
    outputs="text",                    # Output type: text
    title="AI Agora: Fostering Democratic Knowledge via Augmented Retrieval",
    description="Ask our Assistant a question about Congressional Sessions. Get both the answer and the sources."
)

# Launch the Gradio app
iface.launch()
