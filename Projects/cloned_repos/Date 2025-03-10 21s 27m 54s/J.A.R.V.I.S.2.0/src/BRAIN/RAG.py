

# !pip3 install  "langchain>=0.1.0" "langchain-community>=0.0.13" "langchain-core>=0.1.17" \
# "langchain-ollama>=0.0.1" "pdfminer.six>=20221105" "markdown>=3.5.2" "docling>=2.0.0" \
# "beautifulsoup4>=4.12.0" "unstructured>=0.12.0" "chromadb>=0.4.22" "faiss-cpu>=1.7.4" # Required imports

import os
import tempfile
import shutil
from pathlib import Path



# Docling imports

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption, SimplePipeline



# LangChain imports
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory




def get_document_format(file_path) -> InputFormat:
    """Determine the document format based on file extension"""

    try:

        file_path = str(file_path)

        extension = os.path.splitext(file_path)[1].lower()

        format_map = {
            '.pdf': InputFormat.PDF,
            '.docx': InputFormat.DOCX,
            '.doc': InputFormat.DOCX,
            '.pptx': InputFormat.PPTX,
            '.html': InputFormat.HTML,
            '.htm': InputFormat.HTML
        }

        return format_map.get(extension, None)

    except:

        return "Error in get_document_format: {str(e)}"


def convert_document_to_markdown(doc_path , md_path) -> str:
    """Convert document to markdown using simplified pipeline"""

    try:
        # Convert to absolute path string
        input_path = os.path.abspath(str(doc_path))
        print(f"Converting document: {doc_path}")
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy input file to temp directory
            temp_input = os.path.join(temp_dir, os.path.basename(input_path))
            shutil.copy2(input_path, temp_input)

            # Configure pipeline options

            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True  # Disable OCR temporarily
            pipeline_options.do_table_structure = True
            #pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
            # Create converter with minimal options

            converter = DocumentConverter(
                allowed_formats=[
                    InputFormat.PDF,
                    InputFormat.DOCX,
                    InputFormat.HTML,
                    InputFormat.PPTX,
                ],

                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                    ),

                    InputFormat.DOCX: WordFormatOption(
                        pipeline_cls=SimplePipeline
                    )
                }
            )

            # Convert document

            print("Starting conversion...")
            conv_result = converter.convert(temp_input)
            if not conv_result or not conv_result.document:
                raise ValueError(f"Failed to convert document: {doc_path}")

            # Export to markdown
            print("Exporting to markdown...")
            md = conv_result.document.export_to_markdown()
            md_path = os.path.abspath(md_path)
            # Write markdown file
            print(f"Writing markdown to: {md_path}")
            
            with open(md_path, "w", encoding="utf-8") as fp:
                fp.write(md)
            return md_path
    except:
        print(f"Error converting document: {doc_path}")
        return None 

def setup_qa_chain(markdown_path: Path, embeddings_model_name:str = "nomic-embed-text:latest", model_name: str = "granite3.1-dense:2b"):

    """Set up the QA chain for document processing"""

    # Load and split the document

    loader = UnstructuredMarkdownLoader(str(markdown_path))
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_documents(documents)



    # Create embeddings and vector store
    embeddings = OllamaEmbeddings(model=embeddings_model_name)
    vectorstore = FAISS.from_documents(texts, embeddings)
    # Initialize LLM
    llm = OllamaLLM(
        model=model_name,
        temperature=0
    )

    # Set up conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )

    # Create the chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),         
        memory=memory,
        return_source_documents=False
    )
    return qa_chain



def ask_question(qa_chain, question: str):
    """Ask a question and display the answer"""

    result = qa_chain.invoke({"question": question})
    print(f"**Question:** {question}\n\n**Answer:** {result.get('answer', 'No answer found.')}")


def chat_with_rag(subject:str) -> None:
    """"For Depper and insgithfull discussions using (RAG)"""
    #/Users/Shared/Code/testing/DATA/RAWKNOWLEDGEBASE
    #/Users/Shared/Code/testing/DATA/KNOWLEDGEBASE
    subject = subject.lower().strip().replace(" ","_")
    doc_path = Path(f"./DATA/RAWKNOWLEDGEBASE/{subject}_data.pdf")  # Replace with your document path
    md_path = Path(f"./DATA/KNOWLEDGEBASE/{subject}_data_converted.md")
    
    # Check if markdown file exists, if not, create it
    if not md_path.exists():
        if not doc_path.exists():
            print("No document to process")
            return None 
        else:
            # Check format and process only if needed
            doc_format = get_document_format(doc_path)
            if not doc_format:
                print(f"Unsupported document format: {doc_path.suffix}")
                return None
            md_path = convert_document_to_markdown(doc_path , md_path)
    
    
    qa_chain = setup_qa_chain(md_path)
    
    print("\nChat with the RAG model. Type 'exit' to quit.\n")
    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        ask_question(qa_chain, question)
    return None 

if __name__ == "__main__":
    chat_with_rag()
