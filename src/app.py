import os
import uuid
from contextlib import asynccontextmanager

from elasticsearch import AsyncElasticsearch
from fastapi import FastAPI, Depends, HTTPException


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.es_client = AsyncElasticsearch(
        hosts=[os.getenv('ELASTIC_HOST')],
        api_key=os.getenv('ELASTIC_API_KEY'),
    )
    # NOTE: We delete our index everytime we change the file
    await app.state.es_client.indices.delete(index="test", ignore=[404])
    await app.state.es_client.indices.create(index='test',
                                             mappings={
                                                 "properties": {
                                                     "content": {
                                                         "type": "text"
                                                     },
                                                     "content_vector": {
                                                         "type": "dense_vector",
                                                         "dims": 3,
                                                         "index": "true",
                                                         "similarity": "cosine",
                                                     }
                                                 }
                                             }
                                             )

    await app.state.es_client.info()

    yield

    await app.state.es_client.close()


def get_elastic(app: FastAPI = Depends(lambda: app)):
    return app.state.es_client


app = FastAPI(lifespan=lifespan)


@app.post("/add_documents")
async def add_documents(index: str, docs: list[dict], es: AsyncElasticsearch = Depends(get_elastic)):
    try:
        operations = []
        for d in docs:
            operations.append({"index": {"_index": index, "_id": str(uuid.uuid4())}})
            operations.append(d)
        await es.bulk(operations=operations, index=index)
        return {'status': 'ok'}
    except HTTPException as e:
        return {"status": str(e)}


@app.post("/search")
async def search(index: str,
                 query:dict,
                 es: AsyncElasticsearch = Depends(get_elastic)):
    search_results = await es.search(
        index=index,
        size=2,
        retriever={
            # Reciprocal Rank Fusion
            "rrf": {
                "retrievers": [
                    {"standard": {"query": {"match": {"summary": query['query_string']}}}},
                    {
                        "knn": {
                            "field": "content_vector",
                            "query_vector": query['query_vector'],
                            "k": 5,
                            "num_candidates": 10,
                        }
                    },
                ]
            }
        },
    )
    return search_results


@app.get("/")
@app.get("/index")
def alive():
    return {"status": "alive"}
