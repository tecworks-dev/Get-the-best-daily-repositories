from src.endpoint.models import VectorStoreQueryRequest
from src.endpoint.vectorQuery import query_vectorstore
from src.llms.llmQuery import llm_query


def rag_query(data: VectorStoreQueryRequest, collectionInfo):
    try:
        results = query_vectorstore(data, data.is_local)
        data.prompt = f"The following is the data that the user has provided via their custom data collection: " + \
            f"\n\n{results}" + \
            f"\n\nCollection/Store Name: {collectionInfo.name}" + \
            f"\n\nCollection/Store Files: {collectionInfo.files}" + \
            f"\n\nCollection/Store Description: {collectionInfo.description}"

        llm_response = llm_query(data, data.api_key)
        return llm_response
    except Exception as e:
        print(e)
        raise e
