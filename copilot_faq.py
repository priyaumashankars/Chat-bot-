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
from typing import Optional
import asyncio
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-this')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Initialize FastAPI app
app = FastAPI()
security = HTTPBearer()

# Add CORS middleware for frontend compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to specific origins if needed, e.g., ["http://localhost:3000"]
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

# FastAPI Login endpoint
@app.post("/api/login")
async def login(request: LoginRequest):
    conn = await get_db_connection()
    try:
        user = await conn.fetchrow(
            """
            SELECT u.id, u.email, u.password, u.company_id, c.name as company_name, c.code as company_code
            FROM users u
            JOIN companies c ON u.company_id = c.id
            WHERE u.email = $1 AND u.is_active = true
            """,
            request.email
        )
        if not user:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        if not verify_password(request.password, user["password"]):
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

# FastAPI Signup endpoint
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
            INSERT INTO users (email, password, company_id, is_active)
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

# FastAPI Token verification endpoint
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

# Chainlit backend logic
def verify_token(token: str) -> Optional[dict]:
    """Verify JWT token and return payload"""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
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

@cl.on_chat_start
async def start():
    """Initialize chat session"""
    try:
        await cl.Message(
            content=f"""🎉 **Welcome to your Ultra-Fast FAQ Assistant!**

I'm here to help you find answers from the documents using cutting-edge static embeddings.

🚀 **Powered by:** static-retrieval-mrl-en-v1 model (100x-400x faster on CPU!)

💡 **Just ask me anything about your company's policies, products, or processes!**
""",
        ).send()
        cl.user_session.set("authenticated", False)
        cl.user_session.set("user", None)
    except Exception as e:
        print(f"Chat start error: {e}")
        await cl.Message(
            content="❌ **Initialization Error**\n\nThere was an issue starting the FAQ assistant.",
        ).send()

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages and provide FAQ responses"""
    try:
        query = message.content.strip()
        token = cl.query_params.get('token', [None])[0]
        authenticated = False
        user = None

        if token:
            payload = verify_token(token)
            if payload:
                user_id = payload.get('sub')
                if user_id:
                    user_info = await get_user_info(int(user_id))
                    if user_info:
                        authenticated = True
                        user = user_info
                        cl.user_session.set("authenticated", True)
                        cl.user_session.set("user", user_info)
                    else:
                        await cl.get_ui().send_message({"type": "custom", "data": "auth_required"}, parent=True)
                else:
                    await cl.get_ui().send_message({"type": "custom", "data": "auth_required"}, parent=True)
            else:
                await cl.get_ui().send_message({"type": "custom", "data": "auth_required"}, parent=True)
        else:
            await cl.get_ui().send_message({"type": "custom", "data": "auth_required"}, parent=True)

        if not authenticated:
            await cl.Message(
                content="❌ **Authentication Required**\n\nPlease login to access the FAQ assistant.",
            ).send()
            return

        user = cl.user_session.get("user")
        if not user:
            await cl.Message(
                content="❌ **User Not Found**\n\nYour user account could not be found in the database. Please contact support.",
            ).send()
            return

        company_id = user.get('company_id')
        company_name = user.get('company_name', 'Unknown Company')

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

        processing_msg = cl.Message(content="🚀 Searching with ultra-fast static embeddings...")
        await processing_msg.send()

        result = None
        if DATABASE_URL:
            result = await search_faqs_by_embedding(query, company_id)

        if not result and DATABASE_URL:
            result = await search_faqs_by_keywords(query, company_id)

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
        await cl.Message(
            content=f"❌ **Sorry, there was an error processing your request.**\n\nError details: {str(e)}\n\nPlease try again or contact support if the issue persists.",
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
    print("🌐 CORS enabled for iframe embedding")
    print("💡 All FAQ searches now use ultra-fast static embeddings!")
    print("📝 Users can now chat directly with the FAQ system at lightning speed!")
    uvicorn.run(app, host="0.0.0.0", port=5000)