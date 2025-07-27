#import the required python libraries


import os
import gradio as gr

#PyPDF2 is used for reading PDF files
from PyPDF2 import PdfReader
from langchain_community.llms import Cohere
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize environment variables (using HF secrets)
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")
os.environ["COHERE_API_KEY"] = COHERE_API_KEY

# Initializing components
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=['\n', '\n\n', ' ', '']
)
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectorstore = None

def process_pdf(file):
    global vectorstore
    try:
        pdf_reader = PdfReader(file)
        pdf_text = ""
        for page in pdf_reader.pages:
            if page.extract_text():
                pdf_text += page.extract_text()
        
        chunks = text_splitter.split_text(pdf_text)
        vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
        return "PDF processed successfully! You can now ask questions."
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def generate_answer(question):
    global vectorstore
    if not vectorstore:
        return "Please upload and process a PDF first!"
    
    prompt_template = """Answer the question precisely using the context. If the answer isn't in the context, say "answer not available in context".
    Context: {context}
    Question: {question}
    Answer:"""
    
    prompt = PromptTemplate.from_template(prompt_template)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    
    try:
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | Cohere(model="command", temperature=0.1)
            | StrOutputParser()
        )
        return rag_chain.invoke(question)
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## 📄 Resume Q&A with RAG")
    with gr.Row():
        pdf_input = gr.File(label="Upload PDF", type="filepath")  # Fixed parameter
        process_btn = gr.Button("Process PDF")
    status = gr.Textbox(label="Status")
    question = gr.Textbox(label="Ask about the resume")
    answer = gr.Textbox(label="Answer", interactive=False)
    
    process_btn.click(
        fn=process_pdf,
        inputs=pdf_input,
        outputs=status
    )
    question.submit(
        fn=generate_answer,
        inputs=question,
        outputs=answer
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
