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
import socket
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# CRITICAL: Set Chainlit environment variables BEFORE importing chainlit
os.environ["CHAINLIT_DATA_PERSISTENCE"] = "false"
os.environ["CHAINLIT_DISABLE_DATA_LAYER"] = "true"
os.environ["CHAINLIT_ALLOW_ORIGINS"] = "*"
os.environ["CHAINLIT_HOST"] = "0.0.0.0"
os.environ["CHAINLIT_PORT"] = os.getenv("CHAINLIT_PORT", "8001")

# Set Chainlit JWT secret from environment
CHAINLIT_AUTH_SECRET = os.getenv('CHAINLIT_AUTH_SECRET')
if CHAINLIT_AUTH_SECRET:
    os.environ["CHAINLIT_AUTH_SECRET"] = CHAINLIT_AUTH_SECRET
else:
    import secrets
    temp_secret = secrets.token_urlsafe(32)
    os.environ["CHAINLIT_AUTH_SECRET"] = temp_secret
    logger.warning("Using temporary Chainlit auth secret. Set CHAINLIT_AUTH_SECRET in .env for production.")

# Disable Chainlit's data layer to prevent database conflicts
cl.data_layer = None

# Create a no-op data layer to prevent errors
class NoOpDataLayer:
    async def create_thread(self, *args, **kwargs):
        return {"id": str(uuid.uuid4())}
    
    async def update_thread(self, *args, **kwargs):
        pass
    
    async def delete_thread(self, *args, **kwargs):
        pass
    
    async def list_threads(self, *args, **kwargs):
        return []
    
    async def get_thread(self, *args, **kwargs):
        return None
    
    async def create_step(self, *args, **kwargs):
        pass
    
    async def update_step(self, *args, **kwargs):
        pass
    
    async def delete_step(self, *args, **kwargs):
        pass

# Set the no-op data layer
cl.data_layer = NoOpDataLayer()

# Environment variables
DATABASE_URL = os.getenv('DATABASE_URL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-this')
FLASK_SERVER_URL = os.getenv('FLASK_SERVER_URL', 'http://localhost:5000')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Validate environment variables
if not DATABASE_URL:
    logger.error("DATABASE_URL is required")
    raise ValueError("DATABASE_URL is required")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY is required")
    raise ValueError("OPENAI_API_KEY is required")
if not JWT_SECRET_KEY:
    logger.error("JWT_SECRET_KEY is required for secure authentication")
    raise ValueError("JWT_SECRET_KEY is required for secure authentication")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Initialize the static embedding model
logger.info("Loading static embedding model for unified assistant...")
try:
    embedding_model = SentenceTransformer('sentence-transformers/static-retrieval-mrl-en-v1')
    logger.info("Static embedding model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading embedding model: {e}")
    embedding_model = None

# Global connection pool
connection_pool = None

async def init_connection_pool():
    global connection_pool
    logger.info("Attempting to initialize connection pool...")
    if connection_pool is None:
        try:
            connection_pool = await asyncpg.create_pool(
                DATABASE_URL,
                min_size=1,
                max_size=10,
                command_timeout=30,
                server_settings={'jit': 'off'}
            )
            logger.info("Connection pool initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise ValueError(f"Failed to create database connection pool: {e}")

async def get_db_connection():
    """Get database connection with proper error handling"""
    global connection_pool
    if connection_pool is None:
        await init_connection_pool()
    try:
        conn = await asyncio.wait_for(connection_pool.acquire(), timeout=10.0)
        return conn
    except asyncio.TimeoutError:
        logger.error("Database connection timeout after 10 seconds")
        raise HTTPException(status_code=503, detail="Database connection timeout")
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=503, detail=f"Database connection error: {str(e)}")

async def release_db_connection(conn):
    """Release database connection"""
    global connection_pool
    if connection_pool and conn:
        await connection_pool.release(conn)

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
    """Generate embedding using static-retrieval-mrl-en-v1 model"""
    try:
        if not embedding_model:
            logger.error("Embedding model not loaded")
            return None
        if not text or not text.strip():
            return None
        embedding = embedding_model.encode(text.strip(), normalize_embeddings=True)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Error generating static embedding: {e}")
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
            await release_db_connection(conn)
    except Exception as e:
        logger.error(f"Error logging thread update: {e}")

@cl.header_auth_callback
def header_auth_callback(headers: Dict) -> Optional[cl.User]:
    """Handle authentication via headers"""
    auth_header = headers.get("authorization") or headers.get("Authorization")
    if not auth_header:
        logger.info("No authorization header found")
        return None
    
    try:
        scheme, token = auth_header.split(" ", 1)
        if scheme.lower() != "bearer":
            logger.warning("Invalid authorization scheme")
            return None
    except ValueError:
        logger.warning("Invalid authorization header format")
        return None
    
    logger.info(f"Received token: {token[:10]}...")
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=['HS256'])
        user_id = payload.get('sub') or payload.get('user_id')
        user_email = payload.get('email')
        company_id = payload.get('company_id')
        
        if not user_id:
            logger.warning("No user ID in token")
            return None
        
        # Try to get user email from database if not in token
        if not user_email:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                async def get_user_email():
                    try:
                        conn = await get_db_connection()
                        try:
                            user = await conn.fetchrow(
                                "SELECT email FROM users WHERE id = $1 AND is_active = true",
                                int(user_id)
                            )
                            return user['email'] if user else None
                        finally:
                            await release_db_connection(conn)
                    except Exception as e:
                        logger.error(f"Error getting user email: {e}")
                        return None
                
                user_email = loop.run_until_complete(get_user_email())
                if not user_email:
                    user_email = f"user_{user_id}"
            except Exception as e:
                logger.error(f"Error in async email lookup: {e}")
                user_email = f"user_{user_id}"
        
        user = cl.User(
            identifier=user_email or str(user_id),
            metadata={
                "user_id": user_id,
                "email": user_email,
                "company_id": company_id,
                "role": payload.get('role', 'user'),
                "authenticated_at": datetime.utcnow().isoformat(),
                "token_validated_locally": True,
                "embedding_model": "static-retrieval-mrl-en-v1"
            }
        )
        logger.info(f"User authenticated: {user_email} (ID: {user_id})")
        return user
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token has expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid JWT token: {e}")
        return None

async def search_faqs_by_embedding(query: str, company_id: int) -> Optional[dict]:
    """Search FAQs using static embeddings and cosine similarity"""
    try:
        if not company_id:
            logger.warning("No company_id provided for FAQ search")
            return None
            
        query_embedding = generate_static_embedding(query)
        if not query_embedding:
            logger.warning("Failed to generate embedding for query")
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
            
            logger.info(f"Found {len(faqs)} FAQs for company {company_id}")
            
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
                        # Convert PostgreSQL array to Python list
                        faq_embedding = list(faq['embedding'])
                        similarity = cosine_similarity(
                            [query_embedding],
                            [faq_embedding]
                        )[0][0]
                        similarities.append((similarity, faq))
                    except Exception as e:
                        logger.error(f"Error calculating similarity: {e}")
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
                
    except Exception as e:
        logger.error(f"FAQ search error: {e}")
        traceback.print_exc()
        return {
            'answer': f"I encountered an error while searching. Please try again. Error: {str(e)}",
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
            
            # Create companies table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS companies (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    code VARCHAR(50) NOT NULL UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create users table
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
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)") 
            
            # Insert demo data
            company_exists = await conn.fetchval("SELECT COUNT(*) FROM companies")
            if company_exists == 0:
                await conn.execute("""
                    INSERT INTO companies (name, code) VALUES 
                    ('Demo Company', 'demo'),
                    ('Test Company', 'test'),
                    ('Default Company', 'default')
                """)
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise
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
        except Exception as e:
            logger.error(f"Error storing message: {e}")
            return False
        finally:
            if conn:
                await release_db_connection(conn)

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
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error while processing your question. Please try again."

# Initialize chatbot
chatbot = ChatbotEngine(OPENAI_API_KEY)

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    """Password authentication callback"""
    logger.info(f"Auth attempt for username: {username}")
    try:
        response = requests.post(
            f"{FLASK_SERVER_URL}/api/login",
            json={"email": username, "password": password},
            timeout=10
        )
        logger.info(f"Flask login response status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            token = data.get('token')
            user_info = data.get('user', {})
            return cl.User(
                identifier=user_info.get('email', username),
                metadata={
                    "id": user_info.get('id'),
                    "email": user_info.get('email', username),
                    "company_id": user_info.get('company_id'),
                    "company_name": user_info.get('company_name'),
                    "company_code": user_info.get('company_code'),
                    "token": token,
                    "authenticated_at": datetime.utcnow().isoformat(),
                    "embedding_model": "static-retrieval-mrl-en-v1"
                }
            )
        else:
            logger.warning(f"Authentication failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logger.error(f"Auth callback error: {e}")
        return None

@cl.on_chat_start
async def start():
    """Initialize chat session"""
    try:
        logger.info("Initializing connection pool...")
        await init_connection_pool()
        logger.info("Initializing database...")
        await db_manager.init_database()
        
        current_user = cl.user_session.get("user")
        if not current_user:
            await cl.Message(content="❌ **Authentication Error**: No user found. Please refresh and try again.").send()
            return
        
        user_metadata = current_user.metadata or {}
        user_email = user_metadata.get('email') or current_user.identifier
        company_id = user_metadata.get('company_id')
        
        # Validate user in database if we have company info
        if user_email and company_id:
            conn = await get_db_connection()
            try:
                db_user = await conn.fetchrow(
                    """
                    SELECT u.id, u.email, u.company_id, c.name as company_name
                    FROM users u
                    LEFT JOIN companies c ON u.company_id = c.id
                    WHERE u.email = $1 AND u.company_id = $2 AND u.is_active = true
                    """,
                    user_email, company_id
                )
                if db_user:
                    user_email = db_user['email']
            except Exception as e:
                logger.error(f"Error validating user: {e}")
            finally:
                await release_db_connection(conn)
        
        # Set session variables
        cl.user_session.set("authenticated", True)
        cl.user_session.set("user_email", user_email)
        cl.user_session.set("company_id", company_id)
        cl.user_session.set("user_metadata", user_metadata)
        session_id = str(uuid.uuid4())
        cl.user_session.set("session_id", session_id)
        
        logger.info(f"Chat Start - User: {user_email}, Company ID: {company_id}")
        welcome_message = f"""🎉 **Welcome to FAQ & Policy Assistant!**

👋 **Hello, {user_email}!**
🔐 **Authentication:** ✅ Verified
⚡ **Engine:** static-retrieval-mrl-en-v1
🏢 **Company ID:** {company_id or 'Not set'}

💬 **How can I help you today?**

You can ask me questions about your company's policies, procedures, or any other topics covered in your knowledge base."""
        
        await cl.Message(content=welcome_message).send()
        
    except Exception as e:
        logger.error(f"Chat start error: {e}")
        traceback.print_exc()
        await cl.Message(content=f"❌ **Initialization Error**: {str(e)}").send()

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
        user_email = cl.user_session.get("user_email")

        logger.info(f"Processing query from {user_email} (Company: {company_id}): {query}")

        if not query:
            await cl.Message(
                content="❓ **Please ask a question**\n\nI'm ready to help with your FAQs and policies!"
            ).send()
            return

        # Store user message
        if session_id:
            await db_manager.store_message(session_id, 'user', query, 'user_message')

        # Show typing indicator
        async with cl.Step(name="searching") as step:
            step.output = "🔍 Searching knowledge base..."

        # Search for relevant information
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

        # Generate response
        if result and result.get('source') == 'faq':
            confidence_emoji = "🎯" if result['confidence'] > 0.9 else "✅" if result['confidence'] > 0.7 else "💭"
            response = f"""{confidence_emoji} **Here's what I found:**

{result['answer']}

---
*Confidence: {result['confidence']:.1%} | Source: {result['source']} | Model: {result['embedding_type']}*"""
        else:
            # Fallback to AI generation
            async with cl.Step(name="generating") as step:
                step.output = "🤖 Generating AI response..."
            
            ai_response = await chatbot.generate_response(query, [], company_id=company_id)
            response = f"""🤖 **AI Assistant Response:**

{ai_response}

---
*Note: This response was generated by AI since no specific FAQ was found in your company's knowledge base.*"""

        await cl.Message(content=response).send()
        
        # Store assistant message
        if session_id:
            await db_manager.store_message(session_id, 'assistant', response, 'assistant_message')

    except Exception as e:
        logger.error(f"Message handling error: {e}")
        traceback.print_exc()
        await cl.Message(
            content=f"❌ **Error**: I encountered an issue processing your request. Please try again.\n\nError details: {str(e)}"
        ).send()

# Cleanup function
async def cleanup():
    """Cleanup resources on shutdown"""
    global connection_pool
    if connection_pool:
        await connection_pool.close()
        logger.info("Connection pool closed.")

# Error handling middleware
@cl.on_chat_end
async def on_chat_end():
    """Handle chat end event"""
    try:
        session_id = cl.user_session.get("session_id")
        user_email = cl.user_session.get("user_email")
        if session_id and user_email:
            logger.info(f"Chat ended for user: {user_email}, session: {session_id}")
    except Exception as e:
        logger.error(f"Error in chat end handler: {e}")

if __name__ == "__main__":
    import atexit
    
    # Check if port is in use
    def check_port(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    # Set default port
    default_port = int(os.getenv("CHAINLIT_PORT", "8001"))
    
    if check_port(default_port):
        new_port = default_port + 1
        logger.warning(f"Port {default_port} is already in use. Trying port {new_port}...")
        os.environ["CHAINLIT_PORT"] = str(new_port)
        logger.info(f"Chainlit will now run on port {new_port}. Access at: http://localhost:{new_port}")

    # Register cleanup function
    atexit.register(lambda: asyncio.run(cleanup()))
    
    # Print startup information
    logger.info("🤖 Unified FAQ & Policy Assistant Starting...")
    logger.info(f"📍 Chainlit server at: http://localhost:{os.getenv('CHAINLIT_PORT', '8001')}")
    logger.info(f"🗄️ Database: {'✅ Connected' if DATABASE_URL else '❌ Missing DATABASE_URL'}")
    logger.info(f"🤖 OpenAI: {'✅ Configured' if OPENAI_API_KEY else '❌ Missing OPENAI_API_KEY'}")
    logger.info(f"⚡ Embedding Model: static-retrieval-mrl-en-v1")
    logger.info("🚫 Chainlit Data Layer: Disabled (prevents database conflicts)")
    logger.info(f"🔗 Flask Backend: {FLASK_SERVER_URL}")
    
    # Run Chainlit server
    try:
        cl.run()
    except Exception as e:
        logger.error(f"Chainlit server error: {e}")
        traceback.print_exc()