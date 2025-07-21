import os
import asyncpg
import chainlit as cl
import jwt
import bcrypt
from fastapi import FastAPI, HTTPException, Depends, Response, status
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
from datetime import datetime, timedelta
import traceback
import requests
from urllib.parse import urlparse, parse_qs
import uuid
import json
from chainlit.input_widget import Select

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-this')
FLASK_SERVER_URL = os.getenv('FLASK_SERVER_URL', 'http://localhost:5000')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7
DEVELOPMENT_MODE = os.getenv("DEVELOPMENT_MODE", "false").lower() == "true"

# Validate environment variables
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is required")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is required")
if not JWT_SECRET_KEY:
    raise ValueError("JWT_SECRET_KEY is required for secure authentication")

# Initialize FastAPI app
app = FastAPI(title="Unified FAQ & Policy Assistant API", version="2.0.0")
security = HTTPBearer()

# Add CORS middleware for frontend compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client (only for text processing, not embeddings)
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Initialize the static embedding model
print("🚀 Loading static embedding model for unified assistant...")
try:
    embedding_model = SentenceTransformer('sentence-transformers/static-retrieval-mrl-en-v1')
    print("✅ Static embedding model loaded successfully! (100x-400x faster on CPU)")
except Exception as e:
    print(f"❌ Error loading embedding model: {e}")
    embedding_model = None

# Configure Chainlit for iframe embedding
os.environ["CHAINLIT_ALLOW_ORIGINS"] = "*"

# Global connection pool
connection_pool = None

# Pydantic models for request validation
class LoginRequest(BaseModel):
    email: str
    password: str

class SignupRequest(BaseModel):
    email: str
    password: str
    company_code: str

class UserSignup(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: Optional[str] = "user"

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    user: dict

class LegacyLoginRequest(BaseModel):
    identifier: str
    password: str
    metadata: Optional[dict] = None

# Database connection
async def get_db_connection():
    """Get database connection"""
    global connection_pool
    if connection_pool is None:
        await init_connection_pool()
    try:
        conn = await asyncio.wait_for(connection_pool.acquire(), timeout=15.0)
        return conn
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="Database connection timeout")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database connection error: {str(e)}")

async def init_connection_pool():
    global connection_pool
    if connection_pool is None:
        try:
            connection_pool = await asyncpg.create_pool(
                DATABASE_URL,
                min_size=2,
                max_size=20,
                command_timeout=60,
                server_settings={'jit': 'off'}
            )
        except Exception as e:
            raise ValueError(f"Failed to create database connection pool: {e}")

# Hash password
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

# Verify password
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))

# Generate JWT token
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

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

def validate_jwt_token(token: str) -> Optional[cl.User]:
    try:
        decoded = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM])
        user = cl.User(
            identifier=decoded.get('sub'),
            metadata={
                "role": decoded.get('role', 'user'),
                "authenticated_at": datetime.utcnow().isoformat(),
                "token_validated_locally": True,
                "embedding_model": "static-retrieval-mrl-en-v1",
                "performance": "100x-400x faster on CPU"
            }
        )
        return user
    except jwt.ExpiredSignatureError:
        print("JWT token has expired")
        return None
    except jwt.InvalidTokenError as e:
        print(f"Invalid JWT token: {e}")
        return None

# Utility function to ensure string conversion
def ensure_string(value: Union[str, list, None]) -> str:
    """Ensure value is converted to string, handling lists and None values"""
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

# Static embedding helper function
def generate_static_embedding(text: str) -> Optional[list]:
    """Generate embedding using static-retrieval-mrl-en-v1 model (100x-400x faster on CPU)"""
    try:
        if not embedding_model:
            print("Embedding model not loaded")
            return None
        if not text or not text.strip():
            return None
        embedding = embedding_model.encode(text.strip(), normalize_embeddings=True)
        return embedding.tolist()
    except Exception as e:
        print(f"Error generating static embedding: {e}")
        return None

# Thread logging with proper error handling
async def log_thread_update(user_id: int, thread_id: str, message_type: str, content: str, metadata: dict = None):
    """Log thread updates with proper error handling"""
    try:
        if not DATABASE_URL:
            return
        
        conn = await get_db_connection()
        try:
            # Create table if it doesn't exist
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
            
            # Ensure all parameters are properly formatted
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
            if connection_pool and conn:
                await connection_pool.release(conn)
    except Exception as e:
        print(f"Error logging thread update: {e}")

# Enhanced token verification for Chainlit
def verify_token_chainlit(token: str) -> Optional[dict]:
    """Verify JWT token using local verification"""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        print("Token expired")
        return None
    except jwt.InvalidTokenError:
        print("Invalid token")
        return None

def get_token_from_chainlit_context():
    """Extract token from Chainlit context (URL parameters or environment)"""
    try:
        # Check if token is stored in session first
        if hasattr(cl, 'user_session'):
            token = cl.user_session.get("token")
            if token:
                return token
        
        # Check environment variable (for testing)
        env_token = os.getenv("CHAINLIT_TOKEN")
        if env_token:
            return env_token
            
        return None
    except Exception as e:
        print(f"Error getting token from context: {e}")
        return None

async def authenticate_with_flask(token: str) -> Optional[dict]:
    """Authenticate token with Flask backend or local verification"""
    try:
        # First try local verification
        payload = verify_token_chainlit(token)
        if payload:
            user_email = payload.get("sub")
            if user_email:
                # Get user info from database
                conn = await get_db_connection()
                try:
                    user = await conn.fetchrow(
                        """
                        SELECT u.id, u.email, u.company_id, c.name as company_name, c.code as company_code
                        FROM "User" u  -- Changed from users to "User"
                        LEFT JOIN companies c ON u.company_id = c.id
                        WHERE u.email = $1 AND u.is_active = true
                        """,
                        user_email
                    )
                    if user:
                        return {
                            "authenticated": True,
                            "user": dict(user)
                        }
                finally:
                    if connection_pool and conn:
                        await connection_pool.release(conn)
        
        # Fallback to Flask authentication if available
        if FLASK_SERVER_URL:
            try:
                headers = {"Authorization": f"Bearer {token}"}
                response = requests.get(f"{FLASK_SERVER_URL}/api/chainlit-auth", 
                                      headers=headers, timeout=10)
                
                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"Flask authentication failed: {response.status_code} - {response.text}")
            except requests.RequestException as e:
                print(f"Flask request error: {e}")
        
        return None
    except Exception as e:
        print(f"Error authenticating: {e}")
        return None

@cl.header_auth_callback
def header_auth_callback(headers: Dict) -> Optional[cl.User]:
    auth_header = headers.get("authorization") or headers.get("Authorization")
    if not auth_header:
        print("No authorization header found")
        return None
    
    try:
        scheme, token = auth_header.split(" ", 1)
        if scheme.lower() != "bearer":
            print("Invalid authorization scheme")
            return None
    except ValueError:
        print("Invalid authorization header format")
        return None
    
    return validate_jwt_token(token)

# FastAPI authentication endpoints
@app.post("/api/login")
async def login(request: LoginRequest):
    conn = None
    try:
        conn = await get_db_connection()
        user = await conn.fetchrow(
            """
            SELECT u.id, u.email, u.password_hash, u.company_id, c.name as company_name, c.code as company_code
            FROM "User" u  -- Changed from users to "User"
            LEFT JOIN companies c ON u.company_id = c.id
            WHERE u.email = $1 AND u.is_active = true
            """,
            request.email
        )
        if not user:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        if not verify_password(request.password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid email or password")

        token = create_access_token({"sub": user["id"], "email": user["email"]})

        return {
            "token": token,
            "user": {
                "id": user["id"],
                "email": user["email"],
                "company_id": user["company_id"],
                "company_name": user["company_name"],
                "company_code": user["company_code"]
            }
        }
    finally:
        if conn and connection_pool:
            await connection_pool.release(conn)

@app.post("/api/signup")
async def signup(request: SignupRequest):
    conn = None
    try:
        conn = await get_db_connection()
        company = await conn.fetchrow(
            "SELECT id, name, code FROM companies WHERE code = $1",
            request.company_code
        )
        if not company:
            raise HTTPException(status_code=400, detail="Invalid company code")

        existing_user = await conn.fetchrow(
            "SELECT id FROM \"User\" WHERE email = $1",  
            request.email
        )
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")

        hashed_password = hash_password(request.password)

        user = await conn.fetchrow(
            """
            INSERT INTO "User" (email, password_hash, company_id, is_active) 
            VALUES ($1, $2, $3, true)
            RETURNING id, email, company_id
            """,
            request.email, hashed_password, company["id"]
        )

        token = create_access_token({"sub": user["id"], "email": user["email"]})

        return {
            "token": token,
            "user": {
                "id": user["id"],
                "email": user["email"],
                "company_id": company["id"],
                "company_name": company["name"],
                "company_code": company["code"]
            }
        }
    finally:
        if conn and connection_pool:
            await connection_pool.release(conn)

@app.get("/api/verify-token")
async def verify_token_endpoint(credentials: HTTPAuthorizationCredentials = Depends(security)):
    conn = None
    try:
        token = credentials.credentials
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")

        conn = await get_db_connection()
        user = await conn.fetchrow(
            """
            SELECT u.id, u.email, u.company_id, c.name as company_name, c.code as company_code
            FROM "User" u  -- Changed from users to "User"
            LEFT JOIN companies c ON u.company_id = c.id
            WHERE u.id = $1 AND u.is_active = true
            """,
            int(user_id)
        )
        if not user:
            raise HTTPException(status_code=401, detail="User not found or inactive")

        return {
            "user": {
                "id": user["id"],
                "email": user["email"],
                "company_id": user["company_id"],
                "company_name": user["company_name"],
                "company_code": user["company_code"]
            }
        }
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    finally:
        if conn and connection_pool:
            await connection_pool.release(conn)

async def search_faqs_by_embedding(query: str, company_id: int) -> Optional[dict]:
    """Search FAQs using static embeddings and cosine similarity"""
    try:
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
                return None

            similarities = []
            for faq in faqs:
                if faq['embedding']:
                    try:
                        # Convert PostgreSQL array to Python list
                        faq_embedding = list(faq['embedding'])
                        similarity = cosine_similarity(
                            [query_embedding],
                            [faq_embedding]
                        )[0][0]
                        similarities.append((similarity, faq))
                    except Exception as e:
                        print(f"Error calculating similarity: {e}")
                        continue

            if not similarities:
                return None

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
                    'answer': "I couldn't find a specific answer to your question. Please try rephrasing or contact support for more help.",
                    'confidence': float(best_match[0]),
                    'source': 'system',
                    'embedding_type': 'static-retrieval-mrl-en-v1'
                }
        finally:
            if connection_pool and conn:
                await connection_pool.release(conn)
    except Exception as e:
        print(f"FAQ search error: {e}")
        return None

class DatabaseManager:
    def __init__(self):
        pass

    async def init_database(self):
        conn = None
        try:
            conn = await get_db_connection()
            
            # Create companies table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS companies (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    code VARCHAR(50) NOT NULL UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create users table (changed to "User")
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS "User" (
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
            
            # Create faq_data table
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
            
            # Create doc_data table
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
            
            # Create chat_sessions table
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
            
            # Create chat_messages table
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
            
            # Create indexes
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_faq_data_company_id ON faq_data(company_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_data_company_id ON doc_data(company_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_sessions_company_id ON chat_sessions(company_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON \"User\"(email)") 
            
            # Insert demo data
            company_exists = await conn.fetchval("SELECT COUNT(*) FROM companies")
            if company_exists == 0:
                await conn.execute("""
                    INSERT INTO companies (name, code) VALUES 
                    ('Demo Company', 'demo'),
                    ('Test Company', 'test')
                """)
            
        except Exception as e:
            print(f"Database initialization error: {e}")
            raise
        finally:
            if conn and connection_pool:
                await connection_pool.release(conn)

    async def store_message(self, session_id: str, role: str, message: str, message_type: str = 'user_message') -> bool:
        conn = None
        try:
            conn = await get_db_connection()
            await conn.execute("""
                INSERT INTO chat_messages (session_id, role, message, message_type, timestamp)
                VALUES ($1, $2, $3, $4, $5)
            """, session_id, role, message, message_type, datetime.now())
            return True
        except Exception as e:
            print(f"Error storing message: {e}")
            return False
        finally:
            if conn and connection_pool:
                await connection_pool.release(conn)

# Initialize components
db_manager = DatabaseManager()

class ChatbotEngine:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key) if api_key else None

    def generate_embedding(self, text):
        """Generate embedding using static-retrieval-mrl-en-v1 model"""
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

            user_prompt = f"Question: {question}\n\nPlease provide a helpful response."

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
            print(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error while processing your question. Please try again."

# Initialize chatbot
chatbot = ChatbotEngine(OPENAI_API_KEY)

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    """Handle username/password authentication with Flask backend"""
    try:
        # Send login request to Flask server
        response = requests.post(
            f"{FLASK_SERVER_URL}/api/login",
            json={"email": username, "password": password},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            token = data.get('token')
            user_info = data.get('user', {})
            
            return cl.User(
                identifier=user_info.get('email', username),
                metadata={
                    "id": user_info.get('id'),
                    "email": user_info.get('email'),
                    "company_id": user_info.get('company_id'),
                    "token": token
                }
            )
        else:
            print(f"Authentication failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Auth callback error: {e}")
        return None

@cl.on_chat_start
async def start():
    """Initialize chat session with authenticated user"""
    try:
        await init_connection_pool()
        await db_manager.init_database()
        
        # Get the authenticated user
        user = cl.user_session.get("user")
        
        if not user:
            await cl.Message(
                content="❌ **Authentication Error**: No user found. Please refresh and try again."
            ).send()
            return
        
        # Get user metadata
        user_metadata = user.metadata
        user_email = user_metadata.get('email', 'Unknown User')
        company_id = user_metadata.get('company_id')
        
        # Store user info in session
        cl.user_session.set("authenticated", True)
        cl.user_session.set("user_info", user_metadata)
        cl.user_session.set("company_id", company_id)
        
        session_id = str(uuid.uuid4())
        cl.user_session.set("session_id", session_id)
        
        welcome_message = f"""🎉 **Welcome to FAQ & Policy Assistant!**

👋 **Hello, {user_email}!**
🔐 **Authentication:** ✅ Verified
⚡ **Engine:** static-retrieval-mrl-en-v1 (Ultra-fast CPU embeddings)

💬 **How can I help you today?**"""
        
        await cl.Message(content=welcome_message).send()
        
    except Exception as e:
        print(f"Chat start error: {e}")
        await cl.Message(
            content=f"❌ **Initialization Error**: {str(e)}"
        ).send()

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    try:
        # Check if user is authenticated
        if not cl.user_session.get("authenticated"):
            await cl.Message(
                content="❌ **Authentication Required**\n\nPlease restart the chat and ensure you're properly authenticated."
            ).send()
            return

        query = message.content.strip()
        company_id = cl.user_session.get("company_id")
        session_id = cl.user_session.get("session_id")

        if not query:
            await cl.Message(
                content="❓ **Please ask a question**\n\nI'm ready to help with your FAQs and policies!"
            ).send()
            return

        # Store user message
        if session_id:
            await db_manager.store_message(session_id, 'user', query, 'user_message')

        # Search for relevant information
        result = None
        if company_id:
            result = await search_faqs_by_embedding(query, company_id)

        # Generate response
        if result:
            confidence_emoji = "🎯" if result['confidence'] > 0.9 else "✅" if result['confidence'] > 0.7 else "💭"
            response = f"""{confidence_emoji} **Here's what I found:**

{result['answer']}

---
*Confidence: {result['confidence']:.1%} | Embedding: {result['embedding_type']}*"""
        else:
            response = await chatbot.generate_response(query, [], company_id=company_id)

        await cl.Message(content=response).send()
        
        # Store assistant message
        if session_id:
            await db_manager.store_message(session_id, 'assistant', response, 'assistant_message')

    except Exception as e:
        print(f"Message handling error: {e}")
        await cl.Message(
            content=f"❌ **Error**: {str(e)}"
        ).send()

# Cleanup function
async def cleanup():
    global connection_pool
    if connection_pool:
        await connection_pool.close()

if __name__ == "__main__":
    import atexit
    import uvicorn
    
    atexit.register(lambda: asyncio.run(cleanup()))
    
    print("🤖 Unified FAQ & Policy Assistant Starting...")
    print(f"📍 FastAPI server at: http://localhost:5000")
    print(f"📍 Chainlit server at: http://localhost:8001")
    print(f"🗄️ Database: {'✅ Connected' if DATABASE_URL else '❌ Missing DATABASE_URL'}")
    print(f"🤖 OpenAI: {'✅ Configured' if OPENAI_API_KEY else '❌ Missing OPENAI_API_KEY'}")
    print(f"⚡ Embedding Model: static-retrieval-mrl-en-v1 (Ultra-fast CPU)")
    
    # Run FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=5000)