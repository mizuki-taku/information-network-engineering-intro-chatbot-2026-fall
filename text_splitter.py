from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(documents, chunk_size=800, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。"]
    )
    return text_splitter.split_documents(documents)
