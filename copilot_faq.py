import os
import asyncpg
import chainlit as cl
import jwt
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from typing import Optional
import asyncio

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-this')

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Configure Chainlit for iframe embedding
os.environ["CHAINLIT_ALLOW_ORIGINS"] = "*"

async def get_db_connection():
    """Get database connection"""
    try:
        return await asyncpg.connect(DATABASE_URL)
    except Exception as e:
        print(f"Database connection error: {e}")
        raise

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
        user = await conn.fetchrow("""
            SELECT u.id, u.email, u.company_id, c.name as company_name, c.code as company_code
            FROM users u
            JOIN companies c ON u.company_id = c.id
            WHERE u.id = $1 AND u.is_active = true
        """, user_id)
        return dict(user) if user else None
    finally:
        await conn.close()

async def search_faqs_by_embedding(query: str, company_id: int) -> Optional[dict]:
    """Search FAQs using OpenAI embeddings and cosine similarity"""
    if not client:
        return None
        
    try:
        # Generate embedding for the query
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = response.data[0].embedding
        
        conn = await get_db_connection()
        try:
            # Get all FAQs for the company
            faqs = await conn.fetch("""
                SELECT question, answer, embedding
                FROM faq_data
                WHERE company_id = $1
            """, company_id)
            
            if not faqs:
                return None
            
            # Calculate similarities
            similarities = []
            for faq in faqs:
                if faq['embedding']:  # Make sure embedding exists
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
            
            # Sort by similarity and get the best match
            similarities.sort(key=lambda x: x[0], reverse=True)
            best_match = similarities[0]
            
            # Return the best match if confidence is high enough
            if best_match[0] >= 0.7:  # Minimum confidence threshold
                return {
                    'answer': best_match[1]['answer'],
                    'question': best_match[1]['question'],
                    'confidence': float(best_match[0]),
                    'source': 'faq'
                }
            else:
                return {
                    'answer': "I couldn't find a specific answer to your question. Please try rephrasing or contact support for more help.",
                    'confidence': float(best_match[0]),
                    'source': 'system'
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
        # Simple keyword search in questions and answers
        faqs = await conn.fetch("""
            SELECT question, answer
            FROM faq_data
            WHERE company_id = $1 
            AND (LOWER(question) LIKE $2 OR LOWER(answer) LIKE $2)
            ORDER BY 
                CASE WHEN LOWER(question) LIKE $2 THEN 1 ELSE 2 END,
                LENGTH(question)
            LIMIT 1
        """, company_id, f'%{query.lower()}%')
        
        if faqs:
            faq = faqs[0]
            return {
                'answer': faq['answer'],
                'question': faq['question'],
                'confidence': 0.8,  # Default confidence for keyword search
                'source': 'faq'
            }
        return None
        
    finally:
        await conn.close()

@cl.on_chat_start
async def start():
    """Initialize chat session"""
    try:
        # Set welcome message
        await cl.Message(
            content=f"""🎉 **Welcome to your FAQ Assistant!**

I'm here to help you find answers from the documents.
""",
        ).send()
        
        # Set initial session state
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
        
        # For demo purposes, we'll use a default user
        # In production, you would get this from proper authentication flow
        user_info = {
            'id': 1,
            'email': 'demo@example.com',
            'company_id': 1,
            'company_name': 'Demo Company'
        }
        
        if not query:
            await cl.Message(
                content="❓ **Please ask a question**\n\nI'm here to help! Try asking about your company's policies, products, or processes.",
            ).send()
            return
        
        # Show processing message
        processing_msg = cl.Message(content="🔍 Searching for relevant information...")
        await processing_msg.send()
        
        # Try embedding-based search first
        result = None
        if client and DATABASE_URL:
            result = await search_faqs_by_embedding(query, user_info['company_id'])
        
        # Fallback to keyword search if embedding search fails
        if not result and DATABASE_URL:
            result = await search_faqs_by_keywords(query, user_info['company_id'])
        
        # Remove processing message
        await processing_msg.remove()
        
        if result:
            # Format the response
            confidence_emoji = "🎯" if result['confidence'] > 0.9 else "✅" if result['confidence'] > 0.7 else "💭"
            source_text = "from your company's documents" if result['source'] == 'faq' else "general guidance"
            
            response = f"""{confidence_emoji} **Here's what I found {source_text}:**

{result['answer']}

---
*Confidence: {result['confidence']:.1%} | Company: {user_info['company_name']}*"""

            if result['confidence'] < 0.7:
                response += "\n\n💡 **Tip:** Try rephrasing your question or using different keywords for better results."
            
        else:
            response = f"""❓ **No specific information found**

I couldn't find relevant information for your question in {user_info['company_name']}'s documents.

**🔍 Try these suggestions:**
• Rephrase your question using different keywords
• Break complex questions into simpler parts  
• Check if the topic is covered in your company's uploaded documents

**📞 Need more help?** Contact your administrator to ensure relevant documents have been uploaded to the system.

**🔧 System Status:**
• Database: {'✅ Connected' if DATABASE_URL else '❌ Not configured'}
• OpenAI: {'✅ Connected' if OPENAI_API_KEY else '❌ Not configured'}

If this is a fresh installation, please make sure:
1. Your database has company and FAQ data
2. Run your admin panel to upload and process some PDFs first"""
        
        # Send the response
        await cl.Message(content=response).send()
        
    except Exception as e:
        print(f"Message handling error: {e}")
        await cl.Message(
            content=f"❌ **Sorry, there was an error processing your request.**\n\nError details: {str(e)}\n\nPlease try again or contact support if the issue persists.",
        ).send()

# Set Chainlit configuration
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Office Hours",
            message="What are our office hours?",
        ),
        cl.Starter(
            label="Return Policy", 
            message="What is our return policy?",
        ),
        cl.Starter(
            label="Contact Support",
            message="How do I contact support?",
        ),
        cl.Starter(
            label="Company Benefits",
            message="What benefits do we offer?",
        ),
    ]

if __name__ == "__main__":
    print("🤖 Chainlit FAQ Assistant starting...")
    print(f"📍 Access at: http://localhost:8001 (or specified port)")
    print(f"🗄️ Database: {'✅ Connected' if DATABASE_URL else '❌ Missing DATABASE_URL'}")
    print(f"🤖 OpenAI: {'✅ Configured' if OPENAI_API_KEY else '❌ Missing OPENAI_API_KEY'}")
    print("🌐 CORS enabled for iframe embedding")
    print("📝 Users can now chat directly with the FAQ system!")