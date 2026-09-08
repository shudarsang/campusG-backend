from app.rag.retriever import retriever

question = "What is the eligibility for MCA?"

documents = retriever.retrieve(question)

print("=" * 100)

print(f"Retrieved {len(documents)} documents\n")

for i, doc in enumerate(documents, start=1):

    print(f"Result {i}")

    print(doc.metadata)

    print(doc.page_content)

    print("=" * 100)