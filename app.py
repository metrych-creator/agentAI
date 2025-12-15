import os
from rag import answer_query_with_rag

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    answer_query_with_rag("query", pdf_path='data/my_pdf.pdf')