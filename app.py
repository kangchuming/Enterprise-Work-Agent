from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Union


app = FastAPI()

# 内存数据库（假装有数据库）
fake_db = {
    1: {"name": "iPhone", "price": 5999.0},
    2: {"name": "MacBook", "price": 12999.0},
}


# 定义数据格式
class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None # 可选，默认为 None

# 定义请求体格式
class ChatRequest(BaseModel):
    input: str
    thread_id: str = "default"

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/chat")
async def chat(req: ChatRequest):
    return {
        "input": req.input,
        "thread_id": req.thread_id,
        "status": "received"
    }

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}

@app.get("/items")
def read_items(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}

@app.post("/items/")
def create_item(item: Item):
    return {
        "item_name": item.name,
        "item_price": item.price,
        "item_offer": item.is_offer
    }

@app.get("/items/{item_id}")
def read_item(item_id: int):
    if item_id == 0:
        # 主动抛异常，返回 404
        raise HTTPException(status_code=404, detail="Item not found")
    return {"item_id": item_id}

# GET 查询单个
@app.get("/items/{item_id}")
def read_item(item_id: int):
    if item_id not in fake_db:
        raise HTTPException(status_code=404, detail="Item not found")
    return fake_db[item_id]

# GET 查询列表
@app.get("/items/")
def read_items(skip: int = 0, limit: int = 10):
    items = list(fake_db.values())[skip : skip + limit]
    return {"items": items, "total": len(fake_db)}

# POST 创建
@app.post("/items/")
def create_item(item: Item):
    new_id = max(fake_db.keys()) + 1
    fake_db[new_id] = item.dict()
    return {"id": new_id, **item.dict()}

# 健康检查
@app.get("/health")
def health():
    return {"status": "ok"}