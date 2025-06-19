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
from chainlit.input_widget import Select
import asyncio
from chainlit.data.chainlit_data_layer import ChainlitDataLayer
from typing import Any, Dict, List, Optional, Union

# Custom ChainlitDataLayer to fix parameter type mismatch
class CustomChainlitDataLayer(ChainlitDataLayer):
    def __init__(self, database_url):
        super().__init__(database_url=database_url)
   
    def _fix_parameter_types(self, params: Any) -> Any:
        if isinstance(params, dict):
            fixed_params = {}
            for key, value in params.items():
                if isinstance(value, str):
                    if value.lower() in ['true', 'json']:
                        fixed_params[key] = True
                    elif value.lower() in ['false', 'text']:
                        fixed_params[key] = False
                    elif value.lower() in ['null', 'none']:
                        fixed_params[key] = None
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
                    if value.lower() in ['true', 'json']:
                        fixed_params.append(True)
                    elif value.lower() in ['false', 'text']:
                        fixed_params.append(False)
                    elif value.lower() in ['null', 'none']:
                        fixed_params.append(None)
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
        try:
            float(value)
            return True
        except ValueError:
            return False
   
    async def execute_query(self, query: str, params: Any = None):
        try:
            if params is not None:
                params = self._fix_parameter_types(params)
            return await super().execute_query(query, params)
        except Exception as e:
            raise
   
    async def create_step(self, step_dict: Dict) -> Optional[str]:
        try:
            if 'metadata' in step_dict and isinstance(step_dict['metadata'], str):
                try:
                    step_dict['metadata'] = json.loads(step_dict['metadata'])
                except json.JSONDecodeError:
                    pass
           
            boolean_fields = ['streaming', 'is_async', 'show_input', 'disable_feedback']
            for field in boolean_fields:
                if field in step_dict and isinstance(step_dict[field], str):
                    step_dict[field] = step_dict[field].lower() in ['true', '1', 'yes', 'on']
           
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
            raise

async def get_db_connection():
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
    return date_obj.strftime(format_str) if date_obj else default

class DatabaseManager:
    def __init__(self):
        pass
   
    async def init_database(self):
        conn = None
        try:
            conn = await get_db_connection()
           
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
           
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) UNIQUE NOT NULL,
                    doc_id INTEGER,
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
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
           
            await conn.execute("""
                CREATE OR REPLACE FUNCTION update_updated_at_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = CURRENT_TIMESTAMP;
                    RETURN NEW;
                END;
                $$ language 'plpgsql';
            """)
           
            await conn.execute("""
                DROP TRIGGER IF EXISTS update_doc_data_updated_at ON doc_data;
                CREATE TRIGGER update_doc_data_updated_at
                    BEFORE UPDATE ON doc_data
                    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
            """)
           
            await conn.execute("""
                UPDATE doc_data
                SET doc_status = 'completed', updated_at = CURRENT_TIMESTAMP
                WHERE doc_status IS NULL OR doc_status = ''
            """)
           
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
        conn = None
        try:
            conn = await get_db_connection()
           
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
                await conn.execute("""
                    UPDATE chat_sessions
                    SET last_activity = $1
                    WHERE session_id = $2
                """, datetime.now(), session_id)
           
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
        conn = None
        try:
            conn = await get_db_connection()
           
            history_rows = await conn.fetch("""
                SELECT role, message, timestamp FROM chat_messages
                WHERE session_id = $1
                ORDER BY timestamp DESC
                LIMIT $2
            """, session_id, limit)
           
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

    async def get_policy_name_by_id(self, doc_id: int) -> Optional[str]:
        conn = None
        try:
            conn = await get_db_connection()
            result = await conn.fetchrow(
                "SELECT doc_name FROM doc_data WHERE id = $1",
                doc_id
            )
            return result['doc_name'] if result else None
        except Exception as e:
            return None
        finally:
            if conn:
                await connection_pool.release(conn)

    async def get_policy_id_by_name(self, doc_name: str) -> Optional[int]:
        conn = None
        try:
            conn = await get_db_connection()
            result = await conn.fetchrow(
                "SELECT id FROM doc_data WHERE doc_name ILIKE $1",
                f"%{doc_name}%"
            )
            return result['id'] if result else None
        except Exception as e:
            return None
        finally:
            if conn:
                await connection_pool.release(conn)

    async def get_policy_path_by_id(self, doc_id: int) -> Optional[str]:
        conn = None
        try:
            conn = await get_db_connection()
            result = await conn.fetchrow(
                "SELECT doc_path FROM doc_data WHERE id = $1",
                doc_id
            )
            return result['doc_path'] if result else None
        except Exception as e:
            return None
        finally:
            if conn:
                await connection_pool.release(conn)

    async def get_policy_details_by_name(self, doc_name: str) -> Optional[Dict]:
        """Get policy details including name, path, and ID by name."""
        conn = None
        try:
            conn = await get_db_connection()
            result = await conn.fetchrow(
                "SELECT id, doc_name, doc_path FROM doc_data WHERE doc_name ILIKE $1",
                f"%{doc_name}%"
            )
            if result:
                return {
                    'id': result['id'],
                    'doc_name': result['doc_name'],
                    'doc_path': result['doc_path']
                }
            return None
        except Exception as e:
            return None
        finally:
            if conn:
                await connection_pool.release(conn)

class ChatbotEngine:
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)
   
    def generate_embedding(self, text):
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            return None
   
    def find_relevant_context(self, question, faqs_with_embeddings, threshold=0.7):
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
       
        relevant_faqs.sort(key=lambda x: x['similarity'], reverse=True)
        return relevant_faqs[:3]
   
    async def get_document_content(self, doc_id):
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
        conn = None
        try:
            conn = await get_db_connection()
           
            all_docs = await conn.fetch("""
                SELECT id, doc_name, doc_path, faq_count, created_at, updated_at, doc_status, file_size
                FROM doc_data
                ORDER BY doc_name
            """)
           
            processed_docs = []
            for doc in all_docs:
                doc_dict = dict(doc)
                if not doc_dict.get('doc_status') or doc_dict['doc_status'].strip() == '':
                    doc_dict['doc_status'] = 'completed'
                processed_docs.append(doc_dict)
           
            return processed_docs
           
        except Exception as e:
            return []
        finally:
            if conn:
                await connection_pool.release(conn)
   
    async def update_document_status(self, doc_id, status, faq_count=None):
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
        conn = None
        try:
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
           
            faqs_list = [dict(faq) for faq in faqs]
           
            relevant_context = []
            faqs_with_embeddings = [faq for faq in faqs_list if faq.get('embedding')]
            if faqs_with_embeddings:
                relevant_context = self.find_relevant_context(question, faqs_with_embeddings)
           
            doc_content = ""
            if doc_id:
                doc_content = await self.get_document_content(doc_id)
                if doc_content and len(doc_content) > 3000:
                    doc_content = doc_content[:3000] + "..."
           
            context_text = ""
            if relevant_context:
                context_text = "Based on the following information from the document:\n\n"
                for ctx in relevant_context:
                    context_text += f"Q: {ctx['question']}\nA: {ctx['answer']}\n\n"
            elif faqs_list:
                context_text = "Based on available document information:\n\n"
                for faq in faqs_list[:3]:
                    context_text += f"Q: {faq['question']}\nA: {faq['answer']}\n\n"
           
            history_text = ""
            if conversation_history:
                history_text = "Previous conversation:\n"
                for msg in conversation_history[-6:]:
                    role = "User" if msg['role'] == 'user' else "Assistant"
                    history_text += f"{role}: {msg['message']}\n"
                history_text += "\n"
           
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
    try:
        await init_connection_pool()
        await db_manager.init_database()
        session_id = str(uuid.uuid4())
        cl.user_session.set("session_id", session_id)
        cl.user_session.set("doc_id", None)
        cl.user_session.set("doc_name", None)
        
        success = await db_manager.create_session(session_id)
        available_docs = await chatbot.get_available_documents()
        
        if available_docs:
            # Create policy options for the select widget
            policy_options = ["All Policies (Search across all)"]
            policy_options.extend([f"{doc['doc_name']} ({doc.get('faq_count', 0)} FAQs)" for doc in available_docs])
            
            # Create the settings with policy selector
            settings = await cl.ChatSettings(
                [
                    Select(
                        id="PolicySelector",
                        label="📚 Select Policy",
                        values=policy_options,
                        initial_index=0,
                    )
                ]
            ).send()
            
            # Store available docs in session for later use
            cl.user_session.set("available_docs", available_docs)
            
            # Process initial selection
            selected_policy = settings.get("PolicySelector", policy_options[0])
            await handle_policy_widget_selection(selected_policy)
            
            docs_list = "\n".join([
                f"• **{doc['doc_name']}** ({doc.get('faq_count', 0)} FAQs) - Status: {doc.get('doc_status', 'unknown')} - Updated: {safe_strftime(doc.get('updated_at'), default='Never')}"
                for doc in available_docs
            ])
            
            welcome_message = f"""👋 **Welcome to the Policy Assistant!**

🎯 **Current Selection:** {selected_policy}

📚 **Available Policies:**
{docs_list}

You can:
• Use the **Policy Selector** above to switch between policies
• Type `policy name [id]` to get the name of a policy by ID
• Type `policy id [name]` to get the ID of a policy by name
• Type `download [policy_name]` to download a policy document
• Type `list policies` to see all available policies
• Ask questions about policies based on your current selection

💬 **How can I help you today?**

*Note: Use the dropdown selector above to change your policy focus, or use text commands for advanced features.*
"""
        else:
            welcome_message = """👋 **Welcome to the Policy Assistant!**

❌ **No policies are currently available.**

Please contact your administrator to upload policies, or check back later.

💬 **Feel free to ask me questions and I'll do my best to help!**"""
        
        await cl.Message(content=welcome_message).send()
        await db_manager.store_message(session_id, 'assistant', welcome_message)
        
    except Exception as e:
        print(f"Error in chat start: {str(e)}")
        error_message = "Sorry, there was an error initializing the chat. Please refresh the page."
        await cl.Message(content=error_message).send()

@cl.on_settings_update
async def setup_agent(settings):
    """Handle settings updates, particularly policy selection."""
    selected_policy = settings.get("PolicySelector", "All Policies (Search across all)")
    await handle_policy_widget_selection(selected_policy)

async def handle_policy_widget_selection(selected_policy: str):
    """Handle policy selection from the widget."""
    session_id = cl.user_session.get("session_id")
    available_docs = cl.user_session.get("available_docs", [])
    
    if selected_policy == "All Policies (Search across all)":
        # Reset to search all policies
        cl.user_session.set("doc_id", None)
        cl.user_session.set("doc_name", None)
        response_message = "🌐 **Policy Selection:** All Policies\n\nI'll now search across all available policies to answer your questions."
    else:
        # Extract policy name from the selection (remove the FAQ count part)
        policy_display_name = selected_policy.split(" (")[0]
        
        # Find the matching document
        matching_doc = None
        for doc in available_docs:
            if doc['doc_name'] == policy_display_name:
                matching_doc = doc
                break
        
        if matching_doc:
            cl.user_session.set("doc_id", matching_doc['id'])
            cl.user_session.set("doc_name", matching_doc['doc_name'])
            response_message = (
                f"📄 **Policy Selected:** {matching_doc['doc_name']}\n\n"
                f"📊 **Available FAQs:** {matching_doc.get('faq_count', 0)}\n"
                f"📅 **Status:** {matching_doc.get('doc_status', 'unknown')}\n"
                f"📅 **Last Updated:** {safe_strftime(matching_doc.get('updated_at'), default='Never')}\n\n"
                f"I'll now focus my responses on this specific policy. Use `download {matching_doc['doc_name']}` to download."
            )
        else:
            response_message = f"❌ **Error:** Could not find policy '{policy_display_name}'. Please try again."
    
    await cl.Message(content=response_message).send()
    if session_id:
        await db_manager.store_message(session_id, 'assistant', response_message)

@cl.on_message
async def main(message: cl.Message):
    user_message = message.content.strip()
    session_id = cl.user_session.get("session_id")
    doc_id = cl.user_session.get("doc_id")
   
    if not session_id:
        await cl.Message(content="Session error. Please refresh the page.").send()
        return
   
    await db_manager.store_message(session_id, 'user', user_message)
   
    # Handle download commands (enhanced to support multiple formats)
    if user_message.lower().startswith(('download ', 'get ', 'fetch ')):
        await handle_policy_download(user_message)
        return
   
    if user_message.lower().startswith('policy name'):
        await handle_policy_name_query(user_message)
        return
   
    if user_message.lower().startswith('policy id'):
        await handle_policy_id_query(user_message)
        return
   
    if user_message.lower().startswith('select policy'):
        await handle_policy_selection(user_message)
        return
   
    if user_message.lower() in ['list policies', 'show policies', 'available policies']:
        await handle_policy_listing()
        return
   
    if user_message.lower() in ['reset', 'clear selection', 'show all policies']:
        cl.user_session.set("doc_id", None)
        cl.user_session.set("doc_name", None)
        response_message = "✅ Policy selection cleared. I'll now search across all available policies. You can also use the dropdown selector above to change your selection."
        await cl.Message(content=response_message).send()
        await db_manager.store_message(session_id, 'assistant', response_message)
        return
   
    try:
        conversation_history = await db_manager.get_conversation_history(session_id, limit=12)
    except Exception as e:
        conversation_history = []
   
    try:
        response = await chatbot.generate_response(
            user_message,
            conversation_history,
            doc_id
        )
       
        if doc_id:
            doc_name = cl.user_session.get("doc_name")
            response = f"📄 *Searching in: {doc_name}*\n\n{response}"
        else:
            response = f"🌐 *Searching across: All Policies*\n\n{response}"
       
        success = await db_manager.store_message(session_id, 'assistant', response)
        await cl.Message(content=response).send()
       
    except Exception as e:
        error_response = f"I'm sorry, I encountered an error while processing your question. Please try again."
        await cl.Message(content=error_response).send()
        await db_manager.store_message(session_id, 'assistant', error_response)

async def handle_policy_selection(user_message):
    session_id = cl.user_session.get("session_id")
    parts = user_message.lower().split('select policy', 1)
    
    if len(parts) < 2:
        response_message = "Please specify a policy name. Example: `select policy filename.pdf`\n\n💡 **Tip:** You can also use the dropdown selector above for easier policy selection."
        await cl.Message(content=response_message).send()
        await db_manager.store_message(session_id, 'assistant', response_message)
        return
   
    doc_name_query = parts[1].strip()
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
            f"✅ **Policy selected:** {matching_doc['doc_name']}\n\n"
            f"📊 **Available FAQs:** {matching_doc.get('faq_count', 0)}\n\n"
            f"📅 **Status:** {matching_doc.get('doc_status', 'unknown')}\n\n"
            f"📅 **Last Updated:** {safe_strftime(matching_doc.get('updated_at'), default='Never')}\n\n"
            f"I'll now focus my responses on this policy. Use `download {matching_doc['doc_name']}` to download.\n\n"
            f"💡 **Tip:** The dropdown selector above has also been updated to reflect your selection."
        )
        await cl.Message(content=response_message).send()
        await db_manager.store_message(session_id, 'assistant', response_message)
    else:
        docs_list = "\n".join([
            f"• **{doc['doc_name']}** ({doc.get('faq_count', 0)} FAQs)"
            for doc in available_docs
        ])
        response_message = (
            f"❌ **Policy not found:** '{doc_name_query}'\n\n"
            f"📚 **Available policies:**\n{docs_list}\n\n"
            f"💡 **Tip:** Try using the dropdown selector above for easier selection."
        )
        await cl.Message(content=response_message).send()
        await db_manager.store_message(session_id, 'assistant', response_message)

async def handle_policy_download(user_message):
    """Handle download commands with one-click PDF download functionality."""
    session_id = cl.user_session.get("session_id")
    
    # Extract policy name from command
    parts = user_message.lower().split(None, 1)
    if len(parts) < 2:
        current_doc_name = cl.user_session.get("doc_name")
        if current_doc_name:
            await download_policy_file(current_doc_name, session_id)
        else:
            response_message = (
                "Please specify a policy name to download. Examples:\n"
                "• `download policy_name.pdf`\n"
                "• `get employee_handbook`\n"
                "• `fetch privacy_policy`\n\n"
                "💡 **Tip:** If you have a policy selected, just type `download` without a name."
            )
            await cl.Message(content=response_message).send()
            await db_manager.store_message(session_id, 'assistant', response_message)
        return
    
    doc_name_query = parts[1].strip()
    await download_policy_file(doc_name_query, session_id)

async def download_policy_file(doc_name_query, session_id):
    """Download a policy file and send it to the user."""
    try:
        # Get policy details
        policy_details = await db_manager.get_policy_details_by_name(doc_name_query)
        
        if not policy_details:
            available_docs = await chatbot.get_available_documents()
            docs_list = "\n".join([
                f"• **{doc['doc_name']}**"
                for doc in available_docs
            ])
            response_message = (
                f"❌ **Policy not found:** '{doc_name_query}'\n\n"
                f"📚 **Available policies:**\n{docs_list}"
            )
            await cl.Message(content=response_message).send()
            await db_manager.store_message(session_id, 'assistant', response_message)
            return
        
        doc_path = policy_details['doc_path']
        doc_name = policy_details['doc_name']
        
        # Check if file exists
        if not os.path.exists(doc_path):
            response_message = f"❌ **File not found:** The policy file '{doc_name}' is not available on the server."
            await cl.Message(content=response_message).send()
            await db_manager.store_message(session_id, 'assistant', response_message)
            return
        
        # Get file size
        file_size = os.path.getsize(doc_path)
        file_size_mb = file_size / (1024 * 1024)
        
        # Send download confirmation message first
        download_msg = (
            f"📥 **Preparing download:** {doc_name}\n\n"
            f"📊 **File size:** {file_size_mb:.2f} MB\n"
            f"⏳ **Status:** Processing..."
        )
        status_message = await cl.Message(content=download_msg).send()
        
        # Create the file element for download
        try:
            # Read file content
            with open(doc_path, 'rb') as f:
                file_content = f.read()
            
            # Create file element
            file_element = cl.File(
                name=doc_name,
                content=file_content,
                display="inline"  # This enables inline display/download
            )
            
            # Update status message
            success_msg = (
                f"✅ **Download ready:** {doc_name}\n\n"
                f"📊 **File size:** {file_size_mb:.2f} MB\n"
                f"📁 **Status:** Ready for download\n\n"
                f"Click the file below to download:"
            )
            
            # Send the file with success message
            await cl.Message(
                content=success_msg,
                elements=[file_element]
            ).send()
            
            await db_manager.store_message(session_id, 'assistant', f"Downloaded: {doc_name}")
            
        except Exception as file_error:
            error_msg = f"❌ **Download failed:** Could not prepare file for download. Error: {str(file_error)}"
            await cl.Message(content=error_msg).send()
            await db_manager.store_message(session_id, 'assistant', error_msg)
            
    except Exception as e:
        error_response = f"❌ **Error:** Could not process download request. Please try again or contact support."
        await cl.Message(content=error_response).send()
        await db_manager.store_message(session_id, 'assistant', error_response)

async def handle_policy_name_query(user_message):
    """Handle 'policy name [id]' queries."""
    session_id = cl.user_session.get("session_id")
    parts = user_message.split()
    
    if len(parts) < 3 or not parts[2].isdigit():
        response_message = "Please provide a valid policy ID. Example: `policy name 123`"
        await cl.Message(content=response_message).send()
        await db_manager.store_message(session_id, 'assistant', response_message)
        return
    
    doc_id = int(parts[2])
    policy_name = await db_manager.get_policy_name_by_id(doc_id)
    
    if policy_name:
        response_message = f"📄 **Policy ID {doc_id}:** {policy_name}"
        
        # Add download button
        download_button_msg = f"\n\n💾 **Quick actions:**\n• Type `download {policy_name}` to download\n• Type `select policy {policy_name}` to focus on this policy"
        response_message += download_button_msg
    else:
        response_message = f"❌ **Policy not found:** No policy exists with ID {doc_id}"
    
    await cl.Message(content=response_message).send()
    await db_manager.store_message(session_id, 'assistant', response_message)

async def handle_policy_id_query(user_message):
    """Handle 'policy id [name]' queries."""
    session_id = cl.user_session.get("session_id")
    parts = user_message.split('policy id', 1)
    
    if len(parts) < 2:
        response_message = "Please provide a policy name. Example: `policy id employee_handbook.pdf`"
        await cl.Message(content=response_message).send()
        await db_manager.store_message(session_id, 'assistant', response_message)
        return
    
    doc_name = parts[1].strip()
    policy_id = await db_manager.get_policy_id_by_name(doc_name)
    
    if policy_id:
        response_message = f"🆔 **Policy '{doc_name}'** has ID: {policy_id}"
        
        # Add quick actions
        quick_actions = f"\n\n💾 **Quick actions:**\n• Type `download {doc_name}` to download\n• Type `select policy {doc_name}` to focus on this policy"
        response_message += quick_actions
    else:
        available_docs = await chatbot.get_available_documents()
        docs_list = "\n".join([f"• **{doc['doc_name']}**" for doc in available_docs])
        response_message = (
            f"❌ **Policy not found:** '{doc_name}'\n\n"
            f"📚 **Available policies:**\n{docs_list}"
        )
    
    await cl.Message(content=response_message).send()
    await db_manager.store_message(session_id, 'assistant', response_message)

async def handle_policy_listing():
    """Handle policy listing requests."""
    session_id = cl.user_session.get("session_id")
    available_docs = await chatbot.get_available_documents()
    
    if available_docs:
        docs_info = []
        for doc in available_docs:
            doc_info = (
                f"📄 **{doc['doc_name']}**\n"
                f"   • ID: {doc['id']}\n"
                f"   • FAQs: {doc.get('faq_count', 0)}\n"
                f"   • Status: {doc.get('doc_status', 'unknown')}\n"
                f"   • Updated: {safe_strftime(doc.get('updated_at'), default='Never')}\n"
                f"   • Download: `download {doc['doc_name']}`"
            )
            docs_info.append(doc_info)
        
        response_message = (
            f"📚 **Available Policies ({len(available_docs)} total):**\n\n" +
            "\n\n".join(docs_info) +
            "\n\n💡 **Tips:**\n"
            "• Use `download [policy_name]` to download any policy\n"
            "• Use `select policy [policy_name]` to focus on a specific policy\n"
            "• Use the dropdown selector above for easier navigation"
        )
    else:
        response_message = "❌ **No policies available.** Please contact your administrator to upload policies."
    
    await cl.Message(content=response_message).send()
    await db_manager.store_message(session_id, 'assistant', response_message)

# Enhanced welcome message function with download capabilities
async def send_enhanced_welcome_message(available_docs, selected_policy):
    """Send an enhanced welcome message with download options."""
    if available_docs:
        docs_list = []
        for doc in available_docs:
            doc_entry = (
                f"• **{doc['doc_name']}** ({doc.get('faq_count', 0)} FAQs)\n"
                f"  📥 Quick download: `download {doc['doc_name']}`"
            )
            docs_list.append(doc_entry)
        
        docs_display = "\n".join(docs_list)
        
        welcome_message = f"""👋 **Welcome to the Enhanced Policy Assistant!**

🎯 **Current Selection:** {selected_policy}

📚 **Available Policies:**
{docs_display}

🔧 **Available Commands:**
• **Policy Selection:**
  - Use the dropdown selector above
  - `select policy [name]` - Focus on specific policy
  - `reset` - Clear selection (search all policies)

• **Information Commands:**
  - `list policies` - Show all available policies
  - `policy name [id]` - Get policy name by ID
  - `policy id [name]` - Get policy ID by name

• **Download Commands:**
  - `download [policy_name]` - Download specific policy
  - `download` - Download currently selected policy
  - `get [policy_name]` - Alternative download command
  - `fetch [policy_name]` - Another download option

💬 **Ask me anything about the policies, and I'll provide detailed answers!**

*Note: All downloads are one-click and work directly in your browser.*"""
    else:
        welcome_message = """👋 **Welcome to the Policy Assistant!**

❌ **No policies are currently available.**

Please contact your administrator to upload policies, or check back later.

💬 **Feel free to ask me questions and I'll do my best to help!**"""
    
    return welcome_message
