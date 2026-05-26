from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Union
from sqlalchemy import create_engine, text, ForeignKey
from sqlalchemy.orm import Session, DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Integer, Float, Boolean, DateTime, Text
from datetime import datetime


app = FastAPI()
# 内存数据库（假装有数据库）
fake_db = {
    1: {"name": "iPhone", "price": 5999.0},
    2: {"name": "MacBook", "price": 12999.0},
}

# 创建引擎 —— SQLite 只需要一个文件路径
engine = create_engine("sqlite:///test.db", echo=True)

# 1. 声明基类 —— 所有 Model 都继承它
class Base(DeclarativeBase):
    pass

# 2. 定义表
class User(Base):
    __tablename__ = "users" # 表名

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(50))
    age: Mapped[int | None] = mapped_column(Integer, default=None)

# 练习建表
class Person(Base):
    __tablename__ = "persons"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(50))
    age: Mapped[int | None] = mapped_column(Integer, default=None)
    height: Mapped[float | None] = mapped_column(Float, default=None)
    hometome: Mapped[str | None] = mapped_column(String(50), default=None)
    hobby: Mapped[str | None] = mapped_column(String(50), default=None) 

class Product(Base):
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    price: Mapped[float] = mapped_column(Float)
    in_stock: Mapped[bool] = mapped_column(Boolean, default=True)
    description: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

class Post(Base):
    __tablename__ = "posts"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))

    # 反向关系 —— 让你能 post.author.name 访问作者
    author: Mapped["User"] = relationship("User", back_populates="posts")

# 在 User 类里补上反向
User.posts: Mapped[list["Post"]] = relationship("Post", back_populates="author")
# 重新建表
Base.metadata.create_all(engine)

# # 练习 增
# with Session(engine) as session:
#     p1 = Person(id=1995, name='kcm', age=30, height=167.0, hometome='chongqing', hobby='sleep')
#     p2 = Person(id=1996, name='bl', age =9, height=158.0, hometome='sc', hobby='fish')

#     session.add(p1)
#     session.add(p2)
#     session.commit()

#     print(f"kcm的id: {p1.id}")
# ===== 增 (Create) =====
with Session(engine) as session:
    user1 = User(name="张三", age = 25)
    user2 = User(name="李四")

    session.add(user1)
    session.add(user2)
    session.commit()          # ⚠️ 一定要 commit，否则不写入数据库


    print(f"张三的 id: {user1.id}")  # commit 后自动回填主键

with Session(engine) as session:
    p1 = session.get(Person, 1995)
    print(f"id: {p1.id}, name: {p1.name}, age: {p1.age}, height: {p1.height}")

with Session(engine) as session:
    user = session.get(User, 1)
    post = Post(title="SQLAlchemy 入门", author=user)
    session.add(post)
    session.commit()

    #反过来查
    print(post.author.name)
    print(user.posts[0].title)

# ===== 查 (Read) =====
with Session(engine) as session:
    # 查询单个
    user = session.get(User, 1) # 按主键查
    print(user.name)

    # 查所有
    all_users = session.query(User).all()
    for u in all_users:
        print(u.name, u.age)

    # 条件查询
    adults = session.query(User).where(User.age >= 18).all()

    # 查第一个
    zhang = session.query(User).where(User.name == "张三").first()

# ===== 改 (Update) =====
with Session(engine) as session:
    user = session.get(User, 1)
    user.age = 26 # 直接改属性
    session.commit() # 提交
    # flush 会自动检测变更，生成 UPDATE SQL

# ===== 删 (Delete) =====
with Session(engine) as session:
    user = session.get(User, 2)
    session.delete(user)
    session.commit()


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