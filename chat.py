import os
import asyncpg
import openai
import uuid
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from dotenv import load_dotenv
import chainlit as cl
import asyncio
from chainlit.data.chainlit_data_layer import ChainlitDataLayer
from typing import Any, Dict, List, Optional, Union

# Custom ChainlitDataLayer to fix parameter type mismatch
class CustomChainlitDataLayer(ChainlitDataLayer):
    def __init__(self, database_url):
        super().__init__(database_url=database_url)
   
    def _fix_parameter_types(self, params: Any) -> Any:
        """Convert string representations to proper types for PostgreSQL"""
        if isinstance(params, dict):
            fixed_params = {}
            for key, value in params.items():
                if isinstance(value, str):
                    # Handle boolean conversions
                    if value.lower() in ['true', 'json']:
                        fixed_params[key] = True
                    elif value.lower() in ['false', 'text']:
                        fixed_params[key] = False
                    # Handle null/none conversions
                    elif value.lower() in ['null', 'none']:
                        fixed_params[key] = None
                    # Handle numeric strings
                    elif value.isdigit():
                        fixed_params[key] = int(value)
                    elif self._is_float(value):
                        fixed_params[key] = float(value)
                    else:
                        fixed_params[key] = value
                else:
                    fixed_params[key] = value
            return fixed_params
        elif isinstance(params, (list, tuple)):
            fixed_params = []
            for value in params:
                if isinstance(value, str):
                    # Handle boolean conversions
                    if value.lower() in ['true', 'json']:
                        fixed_params.append(True)
                    elif value.lower() in ['false', 'text']:
                        fixed_params.append(False)
                    # Handle null/none conversions
                    elif value.lower() in ['null', 'none']:
                        fixed_params.append(None)
                    # Handle numeric strings
                    elif value.isdigit():
                        fixed_params.append(int(value))
                    elif self._is_float(value):
                        fixed_params.append(float(value))
                    else:
                        fixed_params.append(value)
                else:
                    fixed_params.append(value)
            return tuple(fixed_params) if isinstance(params, tuple) else fixed_params
        else:
            return params
   
    def _is_float(self, value: str) -> bool:
        """Check if string represents a float"""
        try:
            float(value)
            return True
        except ValueError:
            return False
   
    async def execute_query(self, query: str, params: Any = None):
        """Override to handle parameter type conversion properly"""
        try:
            if params is not None:
                params = self._fix_parameter_types(params)
           
            return await super().execute_query(query, params)
           
        except Exception as e:
            raise
   
    async def create_step(self, step_dict: Dict) -> Optional[str]:
        """Override create_step to handle type conversions"""
        try:
            # Pre-process step_dict to ensure proper types
            if 'metadata' in step_dict and isinstance(step_dict['metadata'], str):
                try:
                    step_dict['metadata'] = json.loads(step_dict['metadata'])
                except json.JSONDecodeError:
                    pass
           
            # Ensure boolean fields are properly typed
            boolean_fields = ['streaming', 'is_async', 'show_input', 'disable_feedback']
            for field in boolean_fields:
                if field in step_dict and isinstance(step_dict[field], str):
                    step_dict[field] = step_dict[field].lower() in ['true', '1', 'yes', 'on']
           
            # Ensure numeric fields are properly typed
            numeric_fields = ['start', 'end', 'generation_count', 'prompt_token_count', 'completion_token_count']
            for field in numeric_fields:
                if field in step_dict and isinstance(step_dict[field], str):
                    try:
                        if '.' in step_dict[field]:
                            step_dict[field] = float(step_dict[field])
                        else:
                            step_dict[field] = int(step_dict[field])
                    except ValueError:
                        pass
           
            return await super().create_step(step_dict)
           
        except Exception as e:
            return None

# Load environment variables
load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Validate environment variables
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is required")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is required")

# Override default Chainlit data layer with DATABASE_URL
cl.data_layer = CustomChainlitDataLayer(database_url=DATABASE_URL)

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Global connection pool
connection_pool = None

async def init_connection_pool():
    """Initialize async connection pool"""
    global connection_pool
    if connection_pool is None:
        try:
            connection_pool = await asyncpg.create_pool(
                DATABASE_URL,
                min_size=2,
                max_size=20,
                command_timeout=60,
                server_settings={
                    'jit': 'off'
                }
            )
        except Exception as e:
            raise

async def get_db_connection():
    """Get database connection from pool with timeout"""
    global connection_pool
    if connection_pool is None:
        await init_connection_pool()
   
    try:
        conn = await asyncio.wait_for(
            connection_pool.acquire(),
            timeout=15.0
        )
        return conn
    except asyncio.TimeoutError:
        raise
    except Exception as e:
        raise

def safe_strftime(date_obj, format_str='%Y-%m-%d %H:%M', default="Not set"):
    """Safely format datetime object, returning default if None"""
    return date_obj.strftime(format_str) if date_obj else default

class DatabaseManager:
    def __init__(self):
        pass
   
    async def init_database(self):
        """Create tables if they don't exist and fix existing data"""
        conn = None
        try:
            conn = await get_db_connection()
           
            # Create faq_data table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS faq_data (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    embedding FLOAT8[],
                    doc_id INTEGER,
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
           
            # Create chat_sessions table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) UNIQUE NOT NULL,
                    doc_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
           
            # Create chat_messages table (drop foreign key constraint for now)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    role VARCHAR(50) NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
           
            # Create or replace function for updating updated_at column
            await conn.execute("""
                CREATE OR REPLACE FUNCTION update_updated_at_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = CURRENT_TIMESTAMP;
                    RETURN NEW;
                END;
                $$ language 'plpgsql';
            """)
           
            # Create trigger for auto-updating updated_at field
            await conn.execute("""
                DROP TRIGGER IF EXISTS update_doc_data_updated_at ON doc_data;
                CREATE TRIGGER update_doc_data_updated_at
                    BEFORE UPDATE ON doc_data
                    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
            """)
           
            # Fix existing documents with NULL status - set them to 'completed'
            result = await conn.execute("""
                UPDATE doc_data
                SET doc_status = 'completed', updated_at = CURRENT_TIMESTAMP
                WHERE doc_status IS NULL OR doc_status = ''
            """)
           
            # Update FAQ counts for documents
            await conn.execute("""
                UPDATE doc_data
                SET faq_count = (
                    SELECT COUNT(*)
                    FROM faq_data
                    WHERE faq_data.doc_id = doc_data.id
                )
                WHERE faq_count = 0 OR faq_count IS NULL
            """)
           
        except Exception as e:
            raise
        finally:
            if conn:
                await connection_pool.release(conn)

    async def store_message(self, session_id: str, role: str, message: str) -> bool:
        """Store a chat message in the database"""
        conn = None
        try:
            conn = await get_db_connection()
           
            # Ensure session exists first
            session_exists = await conn.fetchrow(
                "SELECT session_id FROM chat_sessions WHERE session_id = $1",
                session_id
            )
           
            if not session_exists:
                await conn.execute("""
                    INSERT INTO chat_sessions (session_id, doc_id, created_at, last_activity)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (session_id) DO UPDATE SET last_activity = $4
                """, session_id, None, datetime.now(), datetime.now())
            else:
                # Update last activity
                await conn.execute("""
                    UPDATE chat_sessions
                    SET last_activity = $1
                    WHERE session_id = $2
                """, datetime.now(), session_id)
           
            # Insert the message
            await conn.execute("""
                INSERT INTO chat_messages (session_id, role, message, timestamp)
                VALUES ($1, $2, $3, $4)
            """, session_id, role, message, datetime.now())
           
            return True
           
        except Exception as e:
            return False
        finally:
            if conn:
                await connection_pool.release(conn)

    async def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get conversation history for a session"""
        conn = None
        try:
            conn = await get_db_connection()
           
            history_rows = await conn.fetch("""
                SELECT role, message, timestamp FROM chat_messages
                WHERE session_id = $1
                ORDER BY timestamp DESC
                LIMIT $2
            """, session_id, limit)
           
            # Reverse to get chronological order
            conversation_history = [
                {
                    'role': row['role'],
                    'message': row['message'],
                    'timestamp': row['timestamp']
                }
                for row in reversed(history_rows)
            ]
           
            return conversation_history
           
        except Exception as e:
            return []
        finally:
            if conn:
                await connection_pool.release(conn)

    async def create_session(self, session_id: str, doc_id: Optional[int] = None) -> bool:
        """Create a new chat session"""
        conn = None
        try:
            conn = await get_db_connection()
           
            await conn.execute("""
                INSERT INTO chat_sessions (session_id, doc_id, created_at, last_activity)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (session_id) DO UPDATE SET
                    last_activity = $4,
                    doc_id = COALESCE($2, chat_sessions.doc_id)
            """, session_id, doc_id, datetime.now(), datetime.now())
           
            return True
           
        except Exception as e:
            return False
        finally:
            if conn:
                await connection_pool.release(conn)

class ChatbotEngine:
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)
   
    def generate_embedding(self, text):
        """Generate embedding using OpenAI text-embedding-3-small model"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            return None
   
    def find_relevant_context(self, question, faqs_with_embeddings, threshold=0.7):
        """Find relevant FAQs for context"""
        question_embedding = self.generate_embedding(question)
        if not question_embedding:
            return []
       
        relevant_faqs = []
        for faq in faqs_with_embeddings:
            if faq.get('embedding'):
                similarity = cosine_similarity(
                    [question_embedding],
                    [faq['embedding']]
                )[0][0]
               
                if similarity > threshold:
                    relevant_faqs.append({
                        'question': faq['question'],
                        'answer': faq['answer'],
                        'similarity': similarity
                    })
       
        # Sort by similarity and return top 3
        relevant_faqs.sort(key=lambda x: x['similarity'], reverse=True)
        return relevant_faqs[:3]
   
    async def get_document_content(self, doc_id):
        """Get document content for context"""
        conn = None
        try:
            conn = await get_db_connection()
            result = await conn.fetchrow("SELECT doc_content FROM doc_data WHERE id = $1", doc_id)
            return result['doc_content'] if result else None
        except Exception as e:
            return None
        finally:
            if conn:
                await connection_pool.release(conn)
   
    async def get_available_documents(self):
        """Get list of available documents - now includes ALL documents, not just completed ones"""
        conn = None
        try:
            conn = await get_db_connection()
           
            # First, let's see what documents exist
            all_docs = await conn.fetch("""
                SELECT id, doc_name, faq_count, created_at, updated_at, doc_status, file_size
                FROM doc_data
                ORDER BY doc_name
            """)
           
            # For demo purposes, if doc_status is null or empty, treat as completed
            processed_docs = []
            for doc in all_docs:
                doc_dict = dict(doc)
                if not doc_dict.get('doc_status') or doc_dict['doc_status'].strip() == '':
                    doc_dict['doc_status'] = 'completed'  # Assume completed for demo
                processed_docs.append(doc_dict)
           
            # Return all documents for now (you can filter by status later)
            return processed_docs
           
        except Exception as e:
            return []
        finally:
            if conn:
                await connection_pool.release(conn)
   
    async def update_document_status(self, doc_id, status, faq_count=None):
        """Update document status and FAQ count"""
        conn = None
        try:
            conn = await get_db_connection()
            if faq_count is not None:
                await conn.execute("""
                    UPDATE doc_data
                    SET doc_status = $1, faq_count = $2, updated_at = CURRENT_TIMESTAMP
                    WHERE id = $3
                """, status, faq_count, doc_id)
            else:
                await conn.execute("""
                    UPDATE doc_data
                    SET doc_status = $1, updated_at = CURRENT_TIMESTAMP
                    WHERE id = $2
                """, status, doc_id)
        except Exception as e:
            raise
        finally:
            if conn:
                await connection_pool.release(conn)
   
    async def generate_response(self, question, conversation_history, doc_id=None):
        """Generate a conversational response using document context and chat history"""
        conn = None
        try:
            # Get relevant FAQs for context
            conn = await get_db_connection()
           
            if doc_id:
                faqs = await conn.fetch("""
                    SELECT question, answer, embedding
                    FROM faq_data
                    WHERE doc_id = $1
                """, doc_id)
            else:
                faqs = await conn.fetch("""
                    SELECT question, answer, embedding
                    FROM faq_data
                    LIMIT 100
                """)
           
            # Convert to list of dicts for compatibility
            faqs_list = [dict(faq) for faq in faqs]
           
            # Find relevant context (only if embeddings exist)
            relevant_context = []
            faqs_with_embeddings = [faq for faq in faqs_list if faq.get('embedding')]
            if faqs_with_embeddings:
                relevant_context = self.find_relevant_context(question, faqs_with_embeddings)
           
            # Get document content if available
            doc_content = ""
            if doc_id:
                doc_content = await self.get_document_content(doc_id)
                if doc_content and len(doc_content) > 3000:
                    # Truncate for token limits
                    doc_content = doc_content[:3000] + "..."
           
            # Build context from relevant FAQs
            context_text = ""
            if relevant_context:
                context_text = "Based on the following information from the document:\n\n"
                for ctx in relevant_context:
                    context_text += f"Q: {ctx['question']}\nA: {ctx['answer']}\n\n"
            elif faqs_list:
                # If no relevant context found but FAQs exist, use first few FAQs
                context_text = "Based on available document information:\n\n"
                for faq in faqs_list[:3]:
                    context_text += f"Q: {faq['question']}\nA: {faq['answer']}\n\n"
           
            # Build conversation history for context
            history_text = ""
            if conversation_history:
                history_text = "Previous conversation:\n"
                for msg in conversation_history[-6:]:  # Last 6 messages for context
                    role = "User" if msg['role'] == 'user' else "Assistant"
                    history_text += f"{role}: {msg['message']}\n"
                history_text += "\n"
           
            # Create the prompt for conversational response
            system_prompt = """You are a helpful AI assistant that answers questions based on uploaded PDF documents.
           
            Guidelines:
            1. Be conversational and friendly
            2. If you have relevant information from the document, use it to answer
            3. If the question is not covered in the document, politely say so and offer general help
            4. Maintain context from the conversation history
            5. Ask follow-up questions when appropriate
            6. Be concise but informative
            7. Always be helpful and engaging"""
           
            user_prompt = f"""
            {history_text}
           
            {context_text}
           
            Current question: {question}
           
            Please provide a helpful, conversational response based on the available information."""
           
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=800,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
           
        except Exception as e:
            return "I'm sorry, I encountered an error while processing your question. Please try again."
        finally:
            if conn:
                await connection_pool.release(conn)

# Initialize components
db_manager = DatabaseManager()
chatbot = ChatbotEngine(OPENAI_API_KEY)

@cl.on_chat_start
async def start():
    """Initialize chat session"""
    try:
        # Initialize database and connection pool
        await init_connection_pool()
        await db_manager.init_database()
       
        session_id = str(uuid.uuid4())
        cl.user_session.set("session_id", session_id)
        cl.user_session.set("doc_id", None)
        cl.user_session.set("doc_name", None)
       
        # Create session in database
        success = await db_manager.create_session(session_id)
       
        # Get available documents
        available_docs = await chatbot.get_available_documents()
       
        # Create document selection message
        if available_docs:
            docs_list = "\n".join([
                f"• **{doc['doc_name']}** ({doc.get('faq_count', 0)} FAQs) - Status: {doc.get('doc_status', 'unknown')} - Updated: {safe_strftime(doc.get('updated_at'), default='Never')}"
                for doc in available_docs
            ])
            welcome_message = f"""👋 **Welcome to the PDF Document Assistant!**
 
📚 **Available Documents:**
{docs_list}
 
You can:
• Ask questions about any of these documents
• Type `select document [name]` to focus on a specific document
• Ask general questions that will search across all documents
 
💬 **How can I help you today?**"""
        else:
            welcome_message = """👋 **Welcome to the PDF Document Assistant!**
 
❌ **No documents are currently available.**
 
Please contact your administrator to upload documents, or check back later.
 
💬 **Feel free to ask me questions and I'll do my best to help!**"""
       
        await cl.Message(content=welcome_message).send()
       
        # Store welcome message
        await db_manager.store_message(session_id, 'assistant', welcome_message)
       
    except Exception as e:
        error_message = "Sorry, there was an error initializing the chat. Please refresh the page."
        await cl.Message(content=error_message).send()

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
   
    user_message = message.content.strip()
    session_id = cl.user_session.get("session_id")
    doc_id = cl.user_session.get("doc_id")
   
    if not session_id:
        await cl.Message(content="Session error. Please refresh the page.").send()
        return
   
    # Store user message first
    success = await db_manager.store_message(session_id, 'user', user_message)
   
    # Handle document selection
    if user_message.lower().startswith('select document'):
        await handle_document_selection(user_message)
        return
   
    # Handle document listing
    if user_message.lower() in ['list documents', 'show documents', 'available documents']:
        await handle_document_listing()
        return
   
    # Handle reset/clear document selection
    if user_message.lower() in ['reset', 'clear selection', 'show all documents']:
        cl.user_session.set("doc_id", None)
        cl.user_session.set("doc_name", None)
        response_message = "✅ Document selection cleared. I'll now search across all available documents."
        await cl.Message(content=response_message).send()
       
        # Store assistant response
        await db_manager.store_message(session_id, 'assistant', response_message)
        return
   
    # Get conversation history
    try:
        conversation_history = await db_manager.get_conversation_history(session_id, limit=12)
    except Exception as e:
        conversation_history = []
   
    # Generate response
    try:
        response = await chatbot.generate_response(
            user_message,
            conversation_history,
            doc_id
        )
       
        # Add context information if a specific document is selected
        if doc_id:
            doc_name = cl.user_session.get("doc_name")
            response = f"📄 *Searching in: {doc_name}*\n\n{response}"
       
        # Store assistant response
        success = await db_manager.store_message(session_id, 'assistant', response)
       
        await cl.Message(content=response).send()
       
    except Exception as e:
        error_response = f"I'm sorry, I encountered an error while processing your question. Please try again."
        await cl.Message(content=error_response).send()
       
        # Store error response
        await db_manager.store_message(session_id, 'assistant', error_response)

async def handle_document_selection(user_message):
    """Handle document selection commands"""
    session_id = cl.user_session.get("session_id")
   
    # Extract document name from command
    parts = user_message.lower().split('select document', 1)
    if len(parts) < 2:
        response_message = "Please specify a document name. Example: `select document filename.pdf`"
        await cl.Message(content=response_message).send()
        await db_manager.store_message(session_id, 'assistant', response_message)
        return
   
    doc_name_query = parts[1].strip()
   
    # Find matching document
    available_docs = await chatbot.get_available_documents()
    matching_doc = None
   
    for doc in available_docs:
        if doc_name_query.lower() in doc['doc_name'].lower():
            matching_doc = doc
            break
   
    if matching_doc:
        cl.user_session.set("doc_id", matching_doc['id'])
        cl.user_session.set("doc_name", matching_doc['doc_name'])
        response_message = (
            f"✅ **Document selected:** {matching_doc['doc_name']}\n\n"
            f"📊 **Available FAQs:** {matching_doc.get('faq_count', 0)}\n\n"
            f"📅 **Status:** {matching_doc.get('doc_status', 'unknown')}\n\n"
            f"📅 **Last Updated:** {safe_strftime(matching_doc.get('updated_at'), default='Never')}\n\n"
            f"I'll now focus my responses on this document. Ask me anything about it!"
        )
    else:
        docs_list = "\n".join([f"• {doc['doc_name']}" for doc in available_docs])
        response_message = (
            f"❌ **Document not found:** '{doc_name_query}'\n\n"
            f"📚 **Available documents:**\n{docs_list}\n\n"
            f"Please try again with an exact or partial document name."
        )
   
    await cl.Message(content=response_message).send()
    await db_manager.store_message(session_id, 'assistant', response_message)

async def handle_document_listing():
    """Handle document listing requests"""
    session_id = cl.user_session.get("session_id")
    available_docs = await chatbot.get_available_documents()
   
    if available_docs:
        docs_list = "\n".join([
            f"• **{doc['doc_name']}** ({doc['faq_count']} FAQs) - Updated: {safe_strftime(doc.get('updated_at'), default='Never updated')}"
            for doc in available_docs
        ])
        current_doc = cl.user_session.get("doc_name")
        current_selection = f"\n\n🎯 **Currently selected:** {current_doc}" if current_doc else "\n\n🌐 **Currently searching:** All documents"
       
        response_message = (
            f"📚 **Available Documents:**\n{docs_list}{current_selection}\n\n"
            f"💡 **Tip:** Type `select document [name]` to focus on a specific document."
        )
    else:
        response_message = (
            "❌ **No documents are currently available.**\n\n"
            "Please contact your administrator to upload documents."
        )
   
    await cl.Message(content=response_message).send()
    await db_manager.store_message(session_id, 'assistant', response_message)

# Cleanup function for graceful shutdown
async def cleanup():
    """Cleanup resources on shutdown"""
    global connection_pool
    if connection_pool:
        await connection_pool.close()

# Register cleanup function
import atexit
atexit.register(lambda: asyncio.run(cleanup()) if connection_pool else None)