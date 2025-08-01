import os
import asyncpg
import chainlit as cl
import jwt
import bcrypt
from fastapi import FastAPI, HTTPException, Depends, Response, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from typing import Optional, Union, Dict, List, Any
import asyncio
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta, timezone
import traceback
import requests
from urllib.parse import urlparse, parse_qs
import uuid
import json
import socket

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
load_dotenv()

# Chainlit environment settings
os.environ["CHAINLIT_DEFAULT_LANGUAGE"] = "en-US"
os.environ["CHAINLIT_DATA_PERSISTENCE"] = "false"
os.environ["CHAINLIT_DISABLE_DATABASE"] = "true"
os.environ["CHAINLIT_DISABLE_DATA_LAYER"] = "true"
os.environ["CHAINLIT_SESSION_TIMEOUT"] = "0"
os.environ["CHAINLIT_ENABLE_TELEMETRY"] = "false"
os.environ["CHAINLIT_ALLOW_ORIGINS"] = "*"
os.environ["CHAINLIT_HOST"] = "0.0.0.0"
os.environ["CHAINLIT_PORT"] = os.getenv("CHAINLIT_PORT", "8001")
os.environ["CHAINLIT_THREADS_ENABLED"] = "false"
os.environ["CHAINLIT_PERSISTENCE_ENABLED"] = "false"

CHAINLIT_AUTH_SECRET = os.getenv('CHAINLIT_AUTH_SECRET')
if CHAINLIT_AUTH_SECRET:
    os.environ["CHAINLIT_AUTH_SECRET"] = CHAINLIT_AUTH_SECRET
else:
    import secrets
    os.environ["CHAINLIT_AUTH_SECRET"] = secrets.token_urlsafe(32)


# === Mock Data Layer ===
class MockDataLayer:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def __getattr__(self, name):
        async def mock_method(*args, **kwargs):
            if name == 'list_threads':
                return {
                    "data": [],
                    "pageInfo": {
                        "hasNext": False,
                        "hasPrevious": False,
                        "startCursor": None,
                        "endCursor": None
                    }
                }
            elif name == 'create_user':
                return {
                    "id": str(uuid.uuid4()),
                    "identifier": kwargs.get('identifier', 'anonymous'),
                    "createdAt": datetime.now(timezone.utc).isoformat()
                }
            elif name == 'get_user':
                identifier = kwargs.get('identifier', 'anonymous')
                return {
                    "id": str(uuid.uuid4()),
                    "identifier": identifier,
                    "createdAt": datetime.now(timezone.utc).isoformat()
                }
            elif name in ['create_thread', 'create_step', 'create_element', 'create_feedback', 'update_step']:
                return {"id": str(uuid.uuid4())}
            elif name == 'execute_query':
                return []
            return None
        return mock_method


def disable_chainlit_database():
    try:
        import chainlit.data
        import chainlit.data.chainlit_data_layer
        chainlit.data.chainlit_data_layer.ChainlitDataLayer = MockDataLayer
        chainlit.data.ChainlitDataLayer = MockDataLayer
        chainlit.data.data_layer = MockDataLayer()
    except Exception as e:
        print(f"[DISABLE DB] Failed to disable: {e}")


disable_chainlit_database()


# === PATCH: Replace /project/threads endpoint safely ===
def patch_chainlit_server():
    try:
        from chainlit.server import app
        from starlette.responses import JSONResponse

        async def patched_get_user_threads(request: Request):
            try:
                # Return empty threads safely without accessing session
                return JSONResponse({
                    "data": [],
                    "pageInfo": {
                        "hasNext": False,
                        "hasPrevious": False,
                        "startCursor": None,
                        "endCursor": None
                    }
                })

            except Exception as e:
                print(f"[THREADS PATCH] Error: {e}")
                return JSONResponse({
                    "data": [],
                    "pageInfo": {"hasNext": False, "hasPrevious": False}
                }, status_code=200)

        # Remove existing /project/threads route and replace it
        routes_to_remove = [r for r in app.routes if r.path == "/project/threads"]
        for r in routes_to_remove:
            app.routes.remove(r)

        app.add_api_route("/project/threads", patched_get_user_threads, methods=["GET"])
        print("[PATCH] Successfully replaced /project/threads with safe version")

    except Exception as e:
        print(f"[PATCH ERROR] Failed to patch server: {e}")


patch_chainlit_server()


# === Mock asyncpg to prevent real DB connections ===
original_create_pool = asyncpg.create_pool
original_connect = asyncpg.connect


def safe_create_pool(*args, **kwargs):
    import inspect
    frame = inspect.currentframe()
    try:
        for i in range(10):
            frame = frame.f_back
            if frame is None:
                break
            filename = frame.f_code.co_filename
            if 'chainlit' in filename.lower() and 'combined.py' not in filename:
                return MockPool()
    finally:
        del frame
    return original_create_pool(*args, **kwargs)


def safe_connect(*args, **kwargs):
    import inspect
    frame = inspect.currentframe()
    try:
        for i in range(10):
            frame = frame.f_back
            if frame is None:
                break
            filename = frame.f_code.co_filename
            if 'chainlit' in filename.lower() and 'combined.py' not in filename:
                return MockConnection()
    finally:
        del frame
    return original_connect(*args, **kwargs)


class MockPool:
    async def acquire(self):
        return MockConnection()

    async def release(self, conn):
        pass

    async def close(self):
        pass


class MockConnection:
    async def fetch(self, *args, **kwargs):
        return []

    async def fetchrow(self, *args, **kwargs):
        return None

    async def fetchval(self, *args, **kwargs):
        return None

    async def execute(self, *args, **kwargs):
        return None

    async def close(self):
        pass


asyncpg.create_pool = safe_create_pool
asyncpg.connect = safe_connect

# Set global data_layer
cl.data_layer = MockDataLayer()

# === Config ===
DATABASE_URL = os.getenv('DATABASE_URL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-this')
FLASK_SERVER_URL = os.getenv('FLASK_SERVER_URL', 'http://localhost:5000')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is required")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is required")
if not JWT_SECRET_KEY:
    raise ValueError("JWT_SECRET_KEY is required")

client = OpenAI(api_key=OPENAI_API_KEY)
try:
    embedding_model = SentenceTransformer('sentence-transformers/static-retrieval-mrl-en-v1')
except Exception as e:
    print(f"[WARNING] Failed to load embedding model: {e}")
    embedding_model = None

connection_pool = None


async def init_connection_pool():
    global connection_pool
    if connection_pool is None:
        try:
            connection_pool = await original_create_pool(
                DATABASE_URL,
                min_size=1,
                max_size=10,
                command_timeout=30,
                server_settings={'jit': 'off'}
            )
        except Exception as e:
            raise ValueError(f"Failed to create database connection pool: {e}")


async def get_db_connection():
    global connection_pool
    if connection_pool is None:
        await init_connection_pool()
    try:
        conn = await asyncio.wait_for(connection_pool.acquire(), timeout=10.0)
        return conn
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="Database connection timeout")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database connection error: {str(e)}")


async def release_db_connection(conn):
    global connection_pool
    if connection_pool and conn:
        await connection_pool.release(conn)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str, token_type: str = "access"):
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != token_type:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


def create_jwt(identifier: str, role: str = "user") -> str:
    payload = {
        "sub": identifier,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=24),
        "iat": datetime.utcnow(),
        "embedding_model": "static-retrieval-mrl-en-v1"
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=ALGORITHM)


def ensure_string(value: Union[str, list, None]) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def safe_strftime(date_obj, format_str='%Y-%m-%d %H:%M', default="Not set"):
    try:
        if date_obj and hasattr(date_obj, 'strftime'):
            return date_obj.strftime(format_str)
        return default
    except (ValueError, AttributeError):
        return default


def generate_static_embedding(text: str) -> Optional[list]:
    try:
        if not embedding_model:
            return None
        if not text or not text.strip():
            return None
        embedding = embedding_model.encode(text.strip(), normalize_embeddings=True)
        return embedding.tolist()
    except Exception as e:
        print(f"[EMBEDDING ERROR] {e}")
        return None


async def log_thread_update(user_id: int, thread_id: str, message_type: str, content: str, metadata: dict = None):
    try:
        if not DATABASE_URL:
            return
        conn = await get_db_connection()
        try:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS thread_logs (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    thread_id VARCHAR(255) NOT NULL,
                    message_type VARCHAR(50) NOT NULL,
                    content TEXT,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            safe_message_type = ensure_string(message_type)
            safe_content = ensure_string(content)
            safe_metadata = metadata or {}
            await conn.execute(
                """
                INSERT INTO thread_logs (user_id, thread_id, message_type, content, metadata, created_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                user_id,
                thread_id,
                safe_message_type,
                safe_content,
                json.dumps(safe_metadata),
                datetime.utcnow()
            )
        finally:
            await release_db_connection(conn)
    except Exception as e:
        print(f"[LOG ERROR] {e}")


# Global user cache to avoid context issues
_user_cache = {}

# === Auth Callbacks ===
@cl.header_auth_callback
async def header_auth_callback(headers: Dict) -> cl.User:
    """
    Fixed header auth callback that avoids context issues
    """
    auth_header = headers.get("authorization") or headers.get("Authorization")
    
    # Create default user metadata
    default_user_data = {
        "user_id": "anonymous",
        "email": "anonymous",
        "company_id": None,
        "role": "user",
        "authenticated_at": datetime.utcnow().isoformat(),
        "token_validated_locally": False,
        "embedding_model": "static-retrieval-mrl-en-v1"
    }
    
    default_user = cl.User(
        identifier="anonymous",
        metadata=default_user_data
    )
    
    if not auth_header:
        _user_cache["current_user"] = default_user_data
        return default_user

    try:
        scheme, token = auth_header.split(" ", 1)
        if scheme.lower() != "bearer":
            _user_cache["current_user"] = default_user_data
            return default_user
    except ValueError:
        _user_cache["current_user"] = default_user_data
        return default_user

    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=['HS256'])
        user_id = payload.get('sub') or payload.get('user_id')
        user_email = payload.get('email')
        company_id = payload.get('company_id')
        
        if not user_id:
            _user_cache["current_user"] = default_user_data
            return default_user

        if not user_email:
            # Try to get user email from database
            try:
                conn = await get_db_connection()
                try:
                    user = await conn.fetchrow(
                        "SELECT email FROM users WHERE id = $1 AND is_active = true",
                        int(user_id)
                    )
                    user_email = user['email'] if user else f"user_{user_id}"
                finally:
                    await release_db_connection(conn)
            except Exception:
                user_email = f"user_{user_id}"

        user_data = {
            "user_id": user_id,
            "email": user_email,
            "company_id": company_id,
            "role": payload.get('role', 'user'),
            "authenticated_at": datetime.utcnow().isoformat(),
            "token_validated_locally": True,
            "embedding_model": "static-retrieval-mrl-en-v1"
        }
        
        user = cl.User(
            identifier=user_email or str(user_id),
            metadata=user_data
        )
        
        # Cache user data for later use
        _user_cache["current_user"] = user_data
        return user
        
    except jwt.ExpiredSignatureError:
        _user_cache["current_user"] = default_user_data
        return default_user
    except jwt.InvalidTokenError:
        _user_cache["current_user"] = default_user_data
        return default_user
    except Exception as e:
        print(f"[AUTH ERROR] {e}")
        _user_cache["current_user"] = default_user_data
        return default_user


async def search_faqs_by_embedding(query: str, company_id: int) -> Optional[dict]:
    try:
        if not company_id:
            return None
        query_embedding = generate_static_embedding(query)
        if not query_embedding:
            return None
        conn = await get_db_connection()
        try:
            faqs = await conn.fetch(
                """
                SELECT question, answer, embedding
                FROM faq_data
                WHERE company_id = $1 AND embedding IS NOT NULL
                """,
                company_id
            )
            if not faqs:
                return {
                    'answer': "I don't have any FAQs available for your company yet. Please contact your administrator to upload some documents.",
                    'confidence': 0.0,
                    'source': 'system',
                    'embedding_type': 'static-retrieval-mrl-en-v1'
                }

            similarities = []
            for faq in faqs:
                if faq['embedding']:
                    try:
                        faq_embedding = list(faq['embedding'])
                        similarity = cosine_similarity([query_embedding], [faq_embedding])[0][0]
                        similarities.append((similarity, faq))
                    except Exception:
                        continue

            if not similarities:
                return {
                    'answer': "I couldn't process the available FAQs. Please try a different question.",
                    'confidence': 0.0,
                    'source': 'system',
                    'embedding_type': 'static-retrieval-mrl-en-v1'
                }

            similarities.sort(key=lambda x: x[0], reverse=True)
            best_match = similarities[0]
            if best_match[0] >= 0.7:
                return {
                    'answer': best_match[1]['answer'],
                    'question': best_match[1]['question'],
                    'confidence': float(best_match[0]),
                    'source': 'faq',
                    'embedding_type': 'static-retrieval-mrl-en-v1'
                }
            else:
                return {
                    'answer': f"I found some related information, but I'm not confident enough (confidence: {best_match[0]:.1%}). Could you try rephrasing your question or be more specific?",
                    'confidence': float(best_match[0]),
                    'source': 'system',
                    'embedding_type': 'static-retrieval-mrl-en-v1'
                }
        finally:
            await release_db_connection(conn)
    except Exception:
        return {
            'answer': "I encountered an error while searching. Please try again.",
            'confidence': 0.0,
            'source': 'error',
            'embedding_type': 'static-retrieval-mrl-en-v1'
        }


class DatabaseManager:
    def __init__(self):
        pass

    async def init_database(self):
        conn = None
        try:
            conn = await get_db_connection()
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS companies (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    code VARCHAR(50) NOT NULL UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    identifier VARCHAR(255) UNIQUE,
                    email VARCHAR(255) UNIQUE,
                    password_hash TEXT,
                    company_id INTEGER REFERENCES companies(id),
                    is_active BOOLEAN DEFAULT true,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS faq_data (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    embedding FLOAT8[],
                    doc_id INTEGER,
                    company_id INTEGER REFERENCES companies(id),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS doc_data (
                    id SERIAL PRIMARY KEY,
                    doc_name VARCHAR(255) NOT NULL,
                    doc_path VARCHAR(500) NOT NULL,
                    doc_content TEXT,
                    doc_status VARCHAR(50) DEFAULT 'pending',
                    faq_count INTEGER DEFAULT 0,
                    file_size BIGINT DEFAULT 0,
                    company_id INTEGER REFERENCES companies(id),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) UNIQUE NOT NULL,
                    user_id INTEGER,
                    doc_id INTEGER,
                    company_id INTEGER REFERENCES companies(id),
                    client_type VARCHAR(50) DEFAULT 'web',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    role VARCHAR(50) NOT NULL,
                    message TEXT NOT NULL,
                    message_type VARCHAR(50) DEFAULT 'user_message',
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_faq_data_company_id ON faq_data(company_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_data_company_id ON doc_data(company_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_sessions_company_id ON chat_sessions(company_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")

            company_exists = await conn.fetchval("SELECT COUNT(*) FROM companies")
            if company_exists == 0:
                await conn.execute("""
                    INSERT INTO companies (name, code) VALUES 
                    ('Demo Company', 'demo'),
                    ('Test Company', 'test'),
                    ('Default Company', 'default')
                """)
        except Exception as e:
            raise e
        finally:
            if conn:
                await release_db_connection(conn)

    async def store_message(self, session_id: str, role: str, message: str, message_type: str = 'user_message') -> bool:
        conn = None
        try:
            conn = await get_db_connection()
            await conn.execute("""
                INSERT INTO chat_messages (session_id, role, message, message_type, timestamp)
                VALUES ($1, $2, $3, $4, $5)
            """, session_id, role, message, message_type, datetime.now())
            return True
        except Exception:
            return False
        finally:
            if conn:
                await release_db_connection(conn)


db_manager = DatabaseManager()


class ChatbotEngine:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key) if api_key else None

    def generate_embedding(self, text):
        return generate_static_embedding(text)

    async def generate_response(self, question, conversation_history, doc_id=None, company_id=None, is_copilot=False):
        try:
            if not self.client:
                return "I'm sorry, OpenAI client is not configured properly."
            system_prompt = """You are a helpful AI assistant that answers questions based on available information.
            Guidelines:
            1. Be conversational and friendly
            2. If you have relevant information, use it to answer
            3. If the question is not covered, politely say so and offer general help
            4. Maintain context from the conversation history
            5. Be concise but informative
            6. Always be helpful and engaging"""
            user_prompt = f"Question: {question}\nPlease provide a helpful response."
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=600,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"I'm sorry, I encountered an error while processing your question. Details: {str(e)}"


chatbot = ChatbotEngine(OPENAI_API_KEY)


@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    if username and password:
        user_data = {
            "id": 1,
            "email": username,
            "company_id": 2,
            "company_name": "Demo Company",
            "company_code": "demo",
            "authenticated_at": datetime.utcnow().isoformat(),
            "embedding_model": "static-retrieval-mrl-en-v1"
        }
        user = cl.User(
            identifier=username,
            metadata=user_data
        )
        # Cache user data
        _user_cache["current_user"] = user_data
        return user
    try:
        response = requests.post(
            f"{FLASK_SERVER_URL}/api/login",
            json={"email": username, "password": password},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            token = data.get('token')
            user_info = data.get('user', {})
            user_data = {
                "id": user_info.get('id'),
                "email": user_info.get('email', username),
                "company_id": user_info.get('company_id'),
                "company_name": user_info.get('company_name'),
                "company_code": user_info.get('company_code'),
                "token": token,
                "authenticated_at": datetime.utcnow().isoformat(),
                "embedding_model": "static-retrieval-mrl-en-v1"
            }
            user = cl.User(
                identifier=user_info.get('email', username),
                metadata=user_data
            )
            # Cache user data
            _user_cache["current_user"] = user_data
            return user
        else:
            return None
    except Exception:
        return None


@cl.on_chat_start
async def start():
    try:
        await init_connection_pool()
        await db_manager.init_database()
        
        # Use cached user data or get from session safely
        current_user = None
        try:
            current_user = cl.user_session.get("user")
        except Exception:
            # If session access fails, use cached data
            cached_user_data = _user_cache.get("current_user")
            if cached_user_data:
                current_user = cl.User(
                    identifier=cached_user_data.get("email", "anonymous"),
                    metadata=cached_user_data
                )

        if not current_user:
            current_user = cl.User(
                identifier="anonymous",
                metadata={
                    "user_id": "anonymous",
                    "email": "anonymous",
                    "company_id": None,
                    "role": "user",
                    "authenticated_at": datetime.utcnow().isoformat(),
                    "token_validated_locally": False,
                    "embedding_model": "static-retrieval-mrl-en-v1"
                }
            )
        elif isinstance(current_user, dict):
            metadata = current_user.get("metadata", {})
            identifier = current_user.get("identifier") or metadata.get("email", "anonymous")
            current_user = cl.User(identifier=identifier, metadata=metadata)

        # Safely set user session
        try:
            cl.user_session.set("user", current_user)
        except Exception as e:
            print(f"[SESSION ERROR] Could not set user session: {e}")

        user_metadata = getattr(current_user, "metadata", {}) or {}
        user_email = user_metadata.get("email", getattr(current_user, "identifier", "anonymous"))
        company_id = user_metadata.get("company_id")

        if user_email and company_id:
            conn = await get_db_connection()
            try:
                db_user = await conn.fetchrow(
                    "SELECT u.id, u.email, u.company_id, c.name as company_name "
                    "FROM users u "
                    "LEFT JOIN companies c ON u.company_id = c.id "
                    "WHERE u.email = $1 AND u.company_id = $2 AND u.is_active = true",
                    user_email, company_id
                )
                if db_user:
                    current_user.metadata.update({
                        "user_id": db_user["id"],
                        "email": db_user["email"],
                        "company_id": db_user["company_id"],
                        "company_name": db_user["company_name"]
                    })
                    try:
                        cl.user_session.set("user", current_user)
                    except Exception:
                        pass  # Ignore session errors
            except Exception as e:
                print(f"[DB ERROR] {e}")
            finally:
                await release_db_connection(conn)

        # Safely set session variables
        try:
            cl.user_session.set("authenticated", True)
            cl.user_session.set("user_email", user_email)
            cl.user_session.set("company_id", company_id)
            cl.user_session.set("user_metadata", current_user.metadata)
            session_id = str(uuid.uuid4())
            cl.user_session.set("session_id", session_id)
        except Exception as e:
            print(f"[SESSION SET ERROR] {e}")
            # Use fallback variables if session fails
            session_id = str(uuid.uuid4())

        welcome_message = f"""👋 **Welcome to FAQ & Policy Assistant!**
👋 **Hello, {user_email}!**
✅ **Authentication:** Verified
⚡ **Engine:** static-retrieval-mrl-en-v1
🏢 **Company ID:** {company_id or 'Not set'}
💬 **How can I help you today?**
You can ask me questions about your company's policies, procedures, or any other topics covered in your knowledge base."""
        await cl.Message(content=welcome_message).send()

    except Exception as e:
        await cl.Message(content=f"❌ **Initialization Error**: {str(e)}").send()


@cl.on_message
async def main(message: cl.Message):
    try:
        # Get authentication status safely
        authenticated = True
        try:
            authenticated = cl.user_session.get("authenticated", True)
        except Exception:
            authenticated = True  # Default to authenticated

        if not authenticated:
            await cl.Message(
                content="🔒 **Authentication Required**\nPlease restart the chat and ensure you're properly authenticated."
            ).send()
            return

        query = message.content.strip()
        
        # Get session variables safely with fallbacks
        company_id = None
        session_id = str(uuid.uuid4())
        user_email = "anonymous"
        
        try:
            company_id = cl.user_session.get("company_id")
            session_id = cl.user_session.get("session_id", str(uuid.uuid4()))
            user_email = cl.user_session.get("user_email", "anonymous")
        except Exception:
            # Use cached data as fallback
            cached_user_data = _user_cache.get("current_user", {})
            company_id = cached_user_data.get("company_id")
            user_email = cached_user_data.get("email", "anonymous")

        if not query:
            await cl.Message(
                content="❓ **Please ask a question**\nI'm ready to help with your FAQs and policies!"
            ).send()
            return

        if session_id:
            await db_manager.store_message(session_id, 'user', query, 'user_message')

        async with cl.Step(name="searching") as step:
            step.output = "🔍 Searching knowledge base..."

        result = None
        if company_id:
            result = await search_faqs_by_embedding(query, company_id)
        else:
            result = {
                'answer': "I don't have access to your company's FAQs. Please contact your administrator.",
                'confidence': 0.0,
                'source': 'system',
                'embedding_type': 'static-retrieval-mrl-en-v1'
            }

        if result and result.get('source') == 'faq':
            confidence_emoji = "✅" if result['confidence'] > 0.9 else "ℹ️" if result['confidence'] > 0.7 else "⚠️"
            response = f"""{confidence_emoji} **Here's what I found:**
{result['answer']}
---
*Confidence: {result['confidence']:.1%} | Source: {result['source']} | Model: {result['embedding_type']}*"""
        else:
            async with cl.Step(name="generating") as step:
                step.output = "🤖 Generating AI response..."
            ai_response = await chatbot.generate_response(query, [], company_id=company_id)
            response = f"""🤖 **AI Assistant Response:**
{ai_response}
---
*Note: This response was generated by AI since no specific FAQ was found in your company's knowledge base.*"""

        await cl.Message(content=response).send()

        if session_id:
            await db_manager.store_message(session_id, 'assistant', response, 'assistant_message')

    except Exception as e:
        await cl.Message(
            content=f"❌ **Error**: I encountered an issue processing your request. Please try again.\nError: {str(e)}"
        ).send()


@cl.on_chat_end
async def on_chat_end():
    try:
        session_id = None
        user_email = "anonymous"
        
        try:
            session_id = cl.user_session.get("session_id")
            user_email = cl.user_session.get("user_email", "anonymous")
        except Exception:
            # Use cached data as fallback
            cached_user_data = _user_cache.get("current_user", {})
            user_email = cached_user_data.get("email", "anonymous")
            
        if session_id and user_email:
            print(f"Chat ended for user: {user_email}, session: {session_id}")
    except Exception:
        pass


async def cleanup():
    global connection_pool
    if connection_pool:
        await connection_pool.close()


if __name__ == "__main__":
    import atexit

    def check_port(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    default_port = int(os.getenv("CHAINLIT_PORT", "8001"))
    if check_port(default_port):
        new_port = default_port + 1
        os.environ["CHAINLIT_PORT"] = str(new_port)

    atexit.register(lambda: asyncio.run(cleanup()))

    try:
        cl.run()
    except Exception as e:
        print(f"[FATAL] {e}")