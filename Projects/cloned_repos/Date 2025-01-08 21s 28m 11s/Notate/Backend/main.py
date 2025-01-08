from src.authentication.api_key_authorization import api_key_auth
from src.authentication.token import verify_token
from src.data.database.checkAPIKey import check_api_key
from src.data.dataFetch.youtube import youtube_transcript
from src.endpoint.deleteStore import delete_vectorstore_collection
from src.endpoint.models import EmbeddingRequest, QueryRequest, VectorStoreQueryRequest, DeleteCollectionRequest, YoutubeTranscriptRequest, WebCrawlRequest
from src.endpoint.embed import embed
from src.endpoint.vectorQuery import query_vectorstore
from src.endpoint.devApiCall import rag_call, llm_call, vector_call
from src.endpoint.transcribe import transcribe_audio
from src.endpoint.webcrawl import webcrawl

from fastapi import FastAPI, Depends, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import asyncio
import os
import signal
import sys
import psutil
import threading
import uvicorn
import json

app = FastAPI()
embedding_task = None
embedding_event = None
crawl_task = None
crawl_event = None

origins = ["http://localhost", "http://127.0.0.1"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/webcrawl")
async def webcrawl_endpoint(data: WebCrawlRequest, user_id: str = Depends(verify_token)):
    if user_id is None:
        return {"status": "error", "message": "Unauthorized"}

    global crawl_task, crawl_event
    if crawl_task is not None:
        return {"status": "error", "message": "A crawl process is already running"}

    crawl_event = asyncio.Event()

    async def event_generator():
        global crawl_task, crawl_event
        try:
            for result in webcrawl(data, crawl_event):
                if crawl_event.is_set():
                    yield f"data: {{'type': 'cancelled', 'message': 'Crawl process cancelled'}}\n\n"
                    break
                yield f"{result}\n\n"
                await asyncio.sleep(0.1)
        except Exception as e:
            error_data = {
                "status": "error",
                "data": {
                    "message": str(e)
                }
            }
            yield f"data: {json.dumps(error_data)}\n\n"
        finally:
            crawl_task = None
            crawl_event = None

    response = StreamingResponse(event_generator(), media_type="text/event-stream")
    crawl_task = asyncio.create_task(event_generator().__anext__())
    return response


@app.post("/transcribe")
async def transcribe_audio_endpoint(audio_file: UploadFile = File(...), model_name: str = "base", user_id: str = Depends(verify_token)):
    if user_id is None:
        return {"status": "error", "message": "Unauthorized"}
    return await transcribe_audio(audio_file, model_name)


@app.post("/embed")
async def add_embedding(data: EmbeddingRequest, user_id: str = Depends(verify_token)):
    if user_id is None:
        return {"status": "error", "message": "Unauthorized"}
    print("Metadata:", data.metadata)
    global embedding_task, embedding_event

    if embedding_task is not None:
        return {"status": "error", "message": "An embedding process is already running"}

    embedding_event = asyncio.Event()

    async def event_generator():
        global embedding_task, embedding_event
        try:
            for result in embed(data):
                if embedding_event.is_set():
                    yield f"data: {{'type': 'cancelled', 'message': 'Embedding process cancelled'}}\n\n"
                    break

                if result["status"] == "progress":
                    progress_data = result["data"]
                    yield f"data: {{'type': 'progress', 'chunk': {progress_data['chunk']}, 'totalChunks': {progress_data['total_chunks']}, 'percent_complete': '{progress_data['percent_complete']}', 'est_remaining_time': '{progress_data['est_remaining_time']}'}}\n\n"
                else:
                    yield f"data: {{'type': '{result['status']}', 'message': '{result['message']}'}}\n\n"
                await asyncio.sleep(0.1)
        except Exception as e:
            yield f"data: {{'type': 'error', 'message': '{str(e)}'}}\n\n"
        finally:
            embedding_task = None
            embedding_event = None

    response = StreamingResponse(
        event_generator(), media_type="text/event-stream")
    embedding_task = asyncio.create_task(event_generator().__anext__())

    return response


@app.post("/youtube-ingest")
async def youtube_ingest(data: YoutubeTranscriptRequest, user_id: str = Depends(verify_token)):
    if user_id is None:
        return {"status": "error", "message": "Unauthorized"}

    async def event_generator():
        try:
            for result in youtube_transcript(data):
                if result["status"] == "progress":
                    progress_data = result["data"]
                    yield f"data: {{'type': 'progress', 'chunk': {progress_data['chunk']}, 'totalChunks': {progress_data['total_chunks']}, 'percent_complete': '{progress_data['percent_complete']}', 'message': '{progress_data['message']}'}}\n\n"
                else:
                    yield f"data: {{'type': '{result['status']}', 'message': '{result['message']}'}}\n\n"
                await asyncio.sleep(0.1)
        except Exception as e:
            yield f"data: {{'type': 'error', 'message': '{str(e)}'}}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/cancel-embed")
async def cancel_embedding(user_id: str = Depends(verify_token)):
    if user_id is None:
        return {"status": "error", "message": "Unauthorized"}
    global embedding_task, embedding_event
    if embedding_event:
        embedding_event.set()
        return {"status": "success", "message": "Embedding process cancelled"}
    return {"status": "error", "message": "No embedding process running"}


@app.post("/restart-server")
async def restart_server(user_id: str = Depends(verify_token)):
    if user_id is None:
        return {"status": "error", "message": "Unauthorized"}

    def restart():
        pid = os.getpid()
        parent = psutil.Process(pid)
        # Kill all child processes
        for child in parent.children(recursive=True):
            child.kill()
        # Kill the current process
        os.kill(pid, signal.SIGTERM)
        # Start a new instances
        python = sys.executable
        os.execl(python, python, *sys.argv)

    threading.Thread(target=restart).start()
    return {"status": "success", "message": "Server restart initiated"}


@app.post("/vector-query")
async def vector_query(data: VectorStoreQueryRequest, user_id: str = Depends(verify_token)):
    if user_id is None:
        return {"status": "error", "message": "Unauthorized"}
    try:
        result = query_vectorstore(data, data.is_local)
        return result
    except Exception as e:
        print(f"Error querying vectorstore: {str(e)}")
        return {"status": "error", "message": str(e)}


@app.post("/delete-collection")
async def delete_collection(data: DeleteCollectionRequest, user_id: str = Depends(verify_token)):
    if user_id is None:
        return {"status": "error", "message": "Unauthorized"}
    print("Authorized")
    return delete_vectorstore_collection(data)


@app.post("/api/vector")
async def api_vector(query_request: QueryRequest, user_id: str = Depends(api_key_auth)):
    if user_id is None:
        return {"status": "error", "message": "Unauthorized"}
    """ check to see if the userId has API key in SQLite """
    if not query_request.collection_name:
        print("No collection name provided")
        return {"status": "error", "message": "No collection name provided"}
    if check_api_key(int(user_id)) == False:
        print("Unauthorized")
        return {"status": "error", "message": "Unauthorized"}
    print("Authorized")
    return vector_call(query_request, user_id)


@app.post("/api/llm")
async def api_llm(query_request: QueryRequest, user_id: str = Depends(api_key_auth)):
    if user_id is None:
        return {"status": "error", "message": "Unauthorized"}
    """ check to see if the userId has API key in SQLite """
    if not query_request.model:
        print("No model provided")
        return {"status": "error", "message": "No model provided"}
    if check_api_key(int(user_id)) == False:
        print("Unauthorized")
        return {"status": "error", "message": "Unauthorized"}
    print("Authorized")
    return llm_call(query_request, user_id)


@app.post("/api/rag")
async def api_rag(query_request: QueryRequest, user_id: str = Depends(api_key_auth)):
    if user_id is None:
        return {"status": "error", "message": "Unauthorized"}
    """ check to see if the userId has API key in SQLite """
    if not query_request.model:
        print("No model provided")
        return {"status": "error", "message": "No model provided"}
    if not query_request.collection_name:
        print("No collection name provided")
        return {"status": "error", "message": "No collection name provided"}
    if check_api_key(int(user_id)) == False:
        print("Unauthorized")
        return {"status": "error", "message": "Unauthorized"}
    print("Authorized")
    return rag_call(query_request, user_id)


@app.post("/cancel-crawl")
async def cancel_crawl(user_id: str = Depends(verify_token)):
    if user_id is None:
        return {"status": "error", "message": "Unauthorized"}
    global crawl_task, crawl_event
    if crawl_event:
        crawl_event.set()
        return {"status": "success", "message": "Crawl process cancelled"}
    return {"status": "error", "message": "No crawl process running"}

if __name__ == "__main__":
    print("Starting server...")
    uvicorn.run(app, host="127.0.0.1", port=47372)
