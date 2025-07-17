import os
import asyncpg
import chainlit as cl
import jwt
import bcrypt
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from typing import Optional, Union
import asyncio
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta
import traceback
import requests
from urllib.parse import urlparse, parse_qs

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-this')
FLASK_SERVER_URL = os.getenv('FLASK_SERVER_URL', 'http://localhost:5000')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Initialize FastAPI app
app = FastAPI()
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
print("🚀 Loading static embedding model for Chainlit...")
embedding_model = SentenceTransformer('sentence-transformers/static-retrieval-mrl-en-v1')
print("✅ Static embedding model loaded successfully! (100x-400x faster on CPU)")

# Configure Chainlit for iframe embedding
os.environ["CHAINLIT_ALLOW_ORIGINS"] = "*"

# Pydantic models for request validation
class LoginRequest(BaseModel):
    email: str
    password: str

class SignupRequest(BaseModel):
    email: str
    password: str
    company_code: str

# Database connection
async def get_db_connection():
    """Get database connection"""
    try:
        return await asyncpg.connect(DATABASE_URL)
    except Exception as e:
        print(f"Database connection error: {e}")
        raise

# Hash password
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

# Verify password
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))

# Generate JWT token
def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=ALGORITHM)

# Utility function to ensure string conversion
def ensure_string(value: Union[str, list, None]) -> str:
    """Ensure value is converted to string, handling lists and None values"""
    if value is None:
        return ""
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)

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
                safe_metadata,
                datetime.utcnow()
            )
        finally:
            await conn.close()
    except Exception as e:
        print(f"Error logging thread update: {e}")

# FastAPI endpoints (keeping existing ones)
@app.post("/api/login")
async def login(request: LoginRequest):
    conn = await get_db_connection()
    try:
        user = await conn.fetchrow(
            """
            SELECT u.id, u.email, u.password_hash, u.company_id, c.name as company_name, c.code as company_code
            FROM users u
            JOIN companies c ON u.company_id = c.id
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
        await conn.close()

@app.post("/api/signup")
async def signup(request: SignupRequest):
    conn = await get_db_connection()
    try:
        company = await conn.fetchrow(
            "SELECT id, name, code FROM companies WHERE code = $1",
            request.company_code
        )
        if not company:
            raise HTTPException(status_code=400, detail="Invalid company code")

        existing_user = await conn.fetchrow(
            "SELECT id FROM users WHERE email = $1",
            request.email
        )
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")

        hashed_password = hash_password(request.password)

        user = await conn.fetchrow(
            """
            INSERT INTO users (email, password_hash, company_id, is_active)
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
        await conn.close()

@app.get("/api/verify-token")
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")

        conn = await get_db_connection()
        try:
            user = await conn.fetchrow(
                """
                SELECT u.id, u.email, u.company_id, c.name as company_name, c.code as company_code
                FROM users u
                JOIN companies c ON u.company_id = c.id
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
        finally:
            await conn.close()
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Enhanced token verification for Chainlit
def verify_token_chainlit(token: str) -> Optional[dict]:
    """Verify JWT token using Flask backend"""
    try:
        # First try local verification
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
        token = cl.user_session.get("token")
        if token:
            return token
        
        # Check environment variable (for testing)
        env_token = os.getenv("CHAINLIT_TOKEN")
        if env_token:
            return env_token
            
        # Try to get from query string if available
        # This is a workaround since Chainlit doesn't expose URL params directly
        # The token should be passed when launching Chainlit
        try:
            # Check if there's a way to access request context
            import chainlit.context as ctx
            if hasattr(ctx, 'context') and ctx.context:
                # Try to get token from context if available
                context_token = getattr(ctx.context, 'token', None)
                if context_token:
                    return context_token
        except:
            pass
            
        return None
    except Exception as e:
        print(f"Error getting token from context: {e}")
        return None

async def authenticate_with_flask(token: str) -> Optional[dict]:
    """Authenticate token with Flask backend"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{FLASK_SERVER_URL}/api/chainlit-auth", 
                              headers=headers, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Flask authentication failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error authenticating with Flask: {e}")
        return None

async def get_user_info(user_id: int) -> Optional[dict]:
    """Get user information from database"""
    conn = await get_db_connection()
    try:
        user = await conn.fetchrow(
            """
            SELECT u.id, u.email, u.company_id, c.name as company_name, c.code as company_code
            FROM users u
            JOIN companies c ON u.company_id = c.id
            WHERE u.id = $1 AND u.is_active = true
            """,
            user_id
        )
        return dict(user) if user else None
    finally:
        await conn.close()

def generate_static_embedding(text: str) -> Optional[list]:
    """Generate embedding using static-retrieval-mrl-en-v1 model"""
    try:
        embedding = embedding_model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    except Exception as e:
        print(f"Error generating static embedding: {e}")
        return None

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
                WHERE company_id = $1
                """,
                company_id
            )
            if not faqs:
                return None

            similarities = []
            for faq in faqs:
                if faq['embedding']:
                    try:
                        faq_embedding = faq['embedding']
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
            await conn.close()
    except Exception as e:
        print(f"FAQ search error: {e}")
        return None

async def search_faqs_by_keywords(query: str, company_id: int) -> Optional[dict]:
    """Fallback keyword-based search if embeddings fail"""
    conn = await get_db_connection()
    try:
        faqs = await conn.fetch(
            """
            SELECT question, answer
            FROM faq_data
            WHERE company_id = $1 
            AND (LOWER(question) LIKE $2 OR LOWER(answer) LIKE $2)
            ORDER BY 
                CASE WHEN LOWER(question) LIKE $2 THEN 1 ELSE 2 END,
                LENGTH(question)
            LIMIT 1
            """,
            company_id, f'%{query.lower()}%'
        )
        if faqs:
            faq = faqs[0]
            return {
                'answer': faq['answer'],
                'question': faq['question'],
                'confidence': 0.8,
                'source': 'faq',
                'embedding_type': 'keyword_fallback'
            }
        return None
    finally:
        await conn.close()

# Note: @cl.auth_callback is not available in this version of Chainlit
# We'll handle authentication manually in the chat start function

@cl.on_chat_start
async def start():
    """Initialize chat session with authentication"""
    try:
        # Get token from various sources
        token = get_token_from_chainlit_context()
        
        # Alternative: Try to get token from environment or pre-set session
        if not token:
            # Check if token was passed via environment for testing
            token = os.getenv("CHAINLIT_TOKEN")
        
        if not token:
            await cl.Message(
                content="""❌ **Authentication Required**

Please access this chat through your authenticated dashboard.

**For Testing:** Set the CHAINLIT_TOKEN environment variable:
```bash
export CHAINLIT_TOKEN="your_jwt_token_here"
chainlit run copilot_faq.py --port 8001
```

If you're seeing this message, it means:
1. You accessed Chainlit directly without authentication
2. Your session has expired
3. The token is missing from the URL or environment

**To continue:** Please log in through your main application and access the chat from there.
""",
            ).send()
            return

        # Store token in session for future use
        cl.user_session.set("token", token)

        # Verify token with Flask backend
        auth_response = await authenticate_with_flask(token)
        
        if not auth_response or not auth_response.get("authenticated"):
            await cl.Message(
                content=f"""❌ **Authentication Failed**

Your authentication token is invalid or expired.

**Debug Info:**
- Token present: {'✅' if token else '❌'}
- Token length: {len(token) if token else 0}
- Flask server: {FLASK_SERVER_URL}
- Auth response: {auth_response}

**To continue:** Please log in again through your main application.
""",
            ).send()
            return

        user_info = auth_response.get("user")
        if not user_info:
            await cl.Message(
                content="""❌ **User Information Missing**

Could not retrieve your user information.

**To continue:** Please contact support or try logging in again.
""",
            ).send()
            return

        # Store user info in session
        cl.user_session.set("authenticated", True)
        cl.user_session.set("user", user_info)
        
        company_name = user_info.get('company_name', 'Unknown Company')
        user_email = user_info.get('email', 'Unknown User')
        
        await cl.Message(
            content=f"""🎉 **Welcome to your Ultra-Fast FAQ Assistant!**

**Hello, {user_email}!**  
**Company:** {company_name}

I'm here to help you find answers from your company's documents using cutting-edge static embeddings.

🚀 **Powered by:** static-retrieval-mrl-en-v1 model (100x-400x faster on CPU!)

💡 **Just ask me anything about your company's policies, products, or processes!**

**Authentication Status:** ✅ Verified  
**Database:** {'✅ Connected' if DATABASE_URL else '❌ Not configured'}

**System Status:**
- Flask Backend: {'✅ Connected' if FLASK_SERVER_URL else '❌ Not configured'}
- User ID: {user_info.get('id')}
- Company ID: {user_info.get('company_id')}
""",
        ).send()
        
        # Log chat start
        try:
            await log_thread_update(
                user_id=int(user_info.get('id')),
                thread_id="chat_start",
                message_type="system",
                content="Chat session started",
                metadata={"company_id": user_info.get('company_id')}
            )
        except Exception as log_error:
            print(f"Chat start logging error: {log_error}")
            
    except Exception as e:
        print(f"Chat start error: {e}")
        print(f"Chat start error traceback: {traceback.format_exc()}")
        await cl.Message(
            content=f"""❌ **Initialization Error**

There was an issue starting the FAQ assistant.

**Error:** {str(e)}

**Debug Info:**
- Database URL: {'✅ Set' if DATABASE_URL else '❌ Missing'}
- Flask Server: {FLASK_SERVER_URL}
- Token: {'✅ Present' if get_token_from_chainlit_context() else '❌ Missing'}

Please try refreshing the page or contact support.
""",
        ).send()

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages and provide FAQ responses"""
    try:
        # Check if user is authenticated
        if not cl.user_session.get("authenticated"):
            await cl.Message(
                content="❌ **Authentication Required**\n\nPlease restart the chat and ensure you're properly authenticated.",
            ).send()
            return

        user = cl.user_session.get("user")
        if not user:
            await cl.Message(
                content="❌ **Session Expired**\n\nPlease refresh the page and log in again.",
            ).send()
            return

        query = message.content.strip()
        company_id = user.get('company_id')
        company_name = user.get('company_name', 'Unknown Company')
        user_id = user.get('id')

        if not company_id:
            await cl.Message(
                content="❌ **Company Not Associated**\n\nYour account is not associated with any company. Please contact your administrator.",
            ).send()
            return

        if not query:
            await cl.Message(
                content=f"❓ **Please ask a question**\n\nI'm here to help with {company_name}'s policies, products, or processes.\n\n⚡ **Performance:** Using ultra-fast static embeddings for instant responses!",
            ).send()
            return

        # Show processing message
        processing_msg = cl.Message(content="🚀 Searching with ultra-fast static embeddings...")
        await processing_msg.send()

        # Search for FAQ answers
        result = None
        if DATABASE_URL:
            result = await search_faqs_by_embedding(query, company_id)

        if not result and DATABASE_URL:
            result = await search_faqs_by_keywords(query, company_id)

        # Remove processing message
        await processing_msg.remove()

        if result:
            confidence_emoji = "🎯" if result['confidence'] > 0.9 else "✅" if result['confidence'] > 0.7 else "💭"
            source_text = "from your company's documents" if result['source'] == 'faq' else "general guidance"
            embedding_info = ""
            if result.get('embedding_type'):
                if result['embedding_type'] == 'static-retrieval-mrl-en-v1':
                    embedding_info = " ⚡ *Ultra-fast static embeddings*"
                elif result['embedding_type'] == 'keyword_fallback':
                    embedding_info = " 🔍 *Keyword search*"

            response = f"""{confidence_emoji} **Here's what I found {source_text}:**

{result['answer']}

---
*Confidence: {result['confidence']:.1%} | Company: {company_name}{embedding_info}*"""

            if result['confidence'] < 0.7:
                response += "\n\n💡 **Tip:** Try rephrasing your question or using different keywords for better results."
                
            # Log successful response
            try:
                await log_thread_update(
                    user_id=int(user_id),
                    thread_id=cl.user_session.get("id", "default"),
                    message_type="assistant_response",
                    content=response,
                    metadata={
                        "query": query,
                        "confidence": result['confidence'],
                        "embedding_type": result.get('embedding_type')
                    }
                )
            except Exception as log_error:
                print(f"Response logging error: {log_error}")
        else:
            response = f"""❓ **No specific information found**

I couldn't find relevant information for your question in {company_name}'s documents.

**🔍 Try these suggestions:**
• Rephrase your question using different keywords  
• Break complex questions into simpler parts  
• Check if the topic is covered in your company's uploaded documents

**📞 Need more help?** Contact your administrator to ensure relevant documents have been uploaded to the system.

**🔧 System Status:**
• Database: {'✅ Connected' if DATABASE_URL else '❌ Not configured'}
• OpenAI: {'✅ Connected' if OPENAI_API_KEY else '❌ Not configured'}
• Embedding Model: ✅ static-retrieval-mrl-en-v1 (100x-400x faster!)

If this is a fresh installation, please make sure:
1. Your database has company and FAQ data for {company_name}
2. Run your admin panel to upload and process some PDFs first
3. Documents are processed with the new static embedding model"""

        await cl.Message(content=response).send()

    except Exception as e:
        print(f"Message handling error: {e}")
        print(f"Message handling error traceback: {traceback.format_exc()}")
        await cl.Message(
            content=f"❌ **Sorry, there was an error processing your request.**\n\nError details: `{str(e)}`\n\nPlease try again or contact support if the issue persists.",
        ).send()

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Office Hours",
            message="What are our office hours?",
            icon="🕒"
        ),
        cl.Starter(
            label="Return Policy", 
            message="What is our return policy?",
            icon="↩️"
        ),
        cl.Starter(
            label="Contact Support",
            message="How do I contact support?",
            icon="📞"
        ),
        cl.Starter(
            label="Company Benefits",
            message="What benefits do we offer?",
            icon="🎁"
        ),
        cl.Starter(
            label="Performance Test",
            message="Tell me about our company policies",
            icon="⚡"
        ),
    ]

@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="faq",
            markdown_description="The default FAQ assistant powered by ultra-fast static embeddings.",
            icon="🚀",
        ),
        cl.ChatProfile(
            name="support",
            markdown_description="Enhanced support mode with detailed explanations.",
            icon="🛠️",
        ),
    ]

if __name__ == "__main__":
    import uvicorn
    print("🤖 Chainlit FAQ Assistant starting with ultra-fast static embeddings...")
    print(f"📍 FastAPI server at: http://localhost:5000")
    print(f"📍 Chainlit server at: http://localhost:8001 (run separately with 'chainlit run app.py --port 8001')")
    print(f"🗄️ Database: {'✅ Connected' if DATABASE_URL else '❌ Missing DATABASE_URL'}")
    print(f"🤖 OpenAI: {'✅ Configured' if OPENAI_API_KEY else '❌ Missing OPENAI_API_KEY'}")
    print(f"⚡ Embedding Model: static-retrieval-mrl-en-v1 (100x-400x faster on CPU!)")
    print(f"🔗 Flask Backend: {FLASK_SERVER_URL}")
    print("🌐 CORS enabled for iframe embedding")
    print("💡 All FAQ searches now use ultra-fast static embeddings!")
    print("🔐 Authentication integrated with Flask backend!")
    uvicorn.run(app, host="0.0.0.0", port=5000)