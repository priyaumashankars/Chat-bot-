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

    async def store_message(self, session_id: str, role: str, message: str, message_type: str = 'user_message') -> bool:
        conn = None
        try:
            conn = await get_db_connection()
           
            session_exists = await conn.fetchrow(
                "SELECT session_id FROM chat_sessions WHERE session_id = $1",
                session_id
            )
           
            if not session_exists:
                client_type = 'copilot' if cl.context.session.client_type == 'copilot' else 'web'
                
                await conn.execute("""
                    INSERT INTO chat_sessions (session_id, doc_id, client_type, created_at, last_activity)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (session_id) DO UPDATE SET last_activity = $5
                """, session_id, None, client_type, datetime.now(), datetime.now())
            else:
                await conn.execute("""
                    UPDATE chat_sessions
                    SET last_activity = $1
                    WHERE session_id = $2
                """, datetime.now(), session_id)
           
            await conn.execute("""
                INSERT INTO chat_messages (session_id, role, message, message_type, timestamp)
                VALUES ($1, $2, $3, $4, $5)
            """, session_id, role, message, message_type, datetime.now())
           
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
                SELECT role, message, message_type, timestamp FROM chat_messages
                WHERE session_id = $1
                ORDER BY timestamp DESC
                LIMIT $2
            """, session_id, limit)
           
            conversation_history = [
                {
                    'role': row['role'],
                    'message': row['message'],
                    'message_type': row['message_type'],
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

    async def create_session(self, session_id: str, doc_id: Optional[int] = None, client_type: str = 'web') -> bool:
        conn = None
        try:
            conn = await get_db_connection()
           
            await conn.execute("""
                INSERT INTO chat_sessions (session_id, doc_id, client_type, created_at, last_activity)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (session_id) DO UPDATE SET
                    last_activity = $5,
                    doc_id = COALESCE($2, chat_sessions.doc_id),
                    client_type = $3
            """, session_id, doc_id, client_type, datetime.now(), datetime.now())
           
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
   
    async def generate_response(self, question, conversation_history, doc_id=None, is_copilot=False):
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
                    if msg['message_type'] != 'system_message':
                        role = "User" if msg['role'] == 'user' else "Assistant"
                        history_text += f"{role}: {msg['message']}\n"
                history_text += "\n"
           
            if is_copilot:
                system_prompt = """You are an AI assistant integrated as a copilot in a policy management system.
               
                Guidelines:
                1. Be concise and actionable - users expect quick, focused responses in copilot mode
                2. Prioritize direct answers over explanatory text
                3. If you have relevant policy information, provide specific details
                4. For complex queries, break down the response into key points
                5. Suggest specific actions when appropriate (e.g., "download policy X", "check section Y")
                6. Be contextually aware of the user's current task
                7. Keep responses under 200 words unless more detail is specifically requested"""
            else:
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
           
            Please provide a helpful{',' if is_copilot else ', conversational'} response based on the available information."""
           
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=600 if is_copilot else 800,
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

# Copilot Function Handlers
@cl.action_callback("download_policy")
async def handle_download_action(action):
    policy_name = action.value
    session_id = cl.user_session.get("session_id")
    
    if session_id and policy_name:
        await download_policy_file(policy_name, session_id, is_copilot=True)
    else:
        await cl.Message(content="❌ Unable to download policy. Please try again.").send()

@cl.action_callback("select_policy")
async def handle_select_action(action):
    policy_name = action.value
    session_id = cl.user_session.get("session_id")
    
    if session_id and policy_name:
        await handle_policy_selection_by_name(policy_name, is_copilot=True)
    else:
        await cl.Message(content="❌ Unable to select policy. Please try again.").send()

@cl.on_chat_start
async def start():
    try:
        await init_connection_pool()
        await db_manager.init_database()
        session_id = str(uuid.uuid4())
        cl.user_session.set("session_id", session_id)
        cl.user_session.set("doc_id", None)
        cl.user_session.set("doc_name", None)
        
        is_copilot = cl.context.session.client_type == "copilot"
        client_type = "copilot" if is_copilot else "web"
        
        success = await db_manager.create_session(session_id, client_type=client_type)
        available_docs = await chatbot.get_available_documents()
        
        if is_copilot:
            if available_docs:
                docs_summary = f"{len(available_docs)} policies available"
                welcome_message = f"""🤖 **Policy Copilot Ready**

📚 **Status:** {docs_summary}

**Quick Commands:**
• Ask about any policy
• Type policy names for quick access
• Request downloads or summaries

**Ready to help!** 🚀"""
            else:
                welcome_message = """🤖 **Policy Copilot**

❌ No policies currently available.

Contact your administrator to upload policies."""
        
        else:
            if available_docs:
                policy_options = ["All Policies (Search across all)"]
                policy_options.extend([f"{doc['doc_name']} ({doc.get('faq_count', 0)} FAQs)" for doc in available_docs])
                
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
                
                cl.user_session.set("available_docs", available_docs)
                
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
    if cl.context.session.client_type != "copilot":
        selected_policy = settings.get("PolicySelector", "All Policies (Search across all)")
        await handle_policy_widget_selection(selected_policy)

async def handle_policy_widget_selection(selected_policy: str):
    session_id = cl.user_session.get("session_id")
    available_docs = cl.user_session.get("available_docs", [])
    
    if selected_policy == "All Policies (Search across all)":
        cl.user_session.set("doc_id", None)
        cl.user_session.set("doc_name", None)
        response_message = "🌐 **Policy Selection:** All Policies\n\nI'll now search across all available policies to answer your questions."
    else:
        policy_display_name = selected_policy.split(" (")[0]
        
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
    is_copilot = cl.context.session.client_type == "copilot"
   
    if not session_id:
        await cl.Message(content="Session error. Please refresh the page.").send()
        return
    
    # Handle system messages for Copilot
    if is_copilot and message.type == "system_message":
        response = f"Received system message: {user_message}"
        await cl.Message(content=response).send()
        await db_manager.store_message(session_id, 'assistant', response, message_type='system_response')
        return
    
    # Store user message
    await db_manager.store_message(session_id, 'user', user_message, message_type=message.type)
    
    # Handle special commands
    if user_message.lower().startswith("list policies"):
        await handle_list_policies(session_id, is_copilot)
        return
    
    if user_message.lower().startswith("download "):
        policy_name = user_message[9:].strip()
        await download_policy_file(policy_name, session_id, is_copilot)
        return
    
    if user_message.lower().startswith("policy name "):
        policy_id_str = user_message[12:].strip()
        await handle_policy_name_lookup(policy_id_str, session_id, is_copilot)
        return
    
    if user_message.lower().startswith("policy id "):
        policy_name = user_message[10:].strip()
        await handle_policy_id_lookup(policy_name, session_id, is_copilot)
        return
    
    # Handle Copilot test function manually
    if is_copilot and user_message.lower().startswith("test "):
        msg = user_message[5:].strip()
        response = f"You sent: {msg}"
        await cl.Message(content=response).send()
        await db_manager.store_message(session_id, 'assistant', response)
        return
    
    # Check if user is asking about a specific policy (copilot mode)
    if is_copilot:
        policy_match = await check_policy_mention(user_message)
        if policy_match:
            await suggest_policy_actions(policy_match, user_message)
            return
    
    try:
        conversation_history = await db_manager.get_conversation_history(session_id, limit=10)
        
        response = await chatbot.generate_response(
            user_message, 
            conversation_history, 
            doc_id=doc_id,
            is_copilot=is_copilot
        )
        
        await cl.Message(content=response).send()
        
        await db_manager.store_message(session_id, 'assistant', response)
        
    except Exception as e:
        print(f"Error in main message handler: {str(e)}")
        error_response = "I'm sorry, I encountered an error while processing your question. Please try again."
        await cl.Message(content=error_response).send()
        await db_manager.store_message(session_id, 'assistant', error_response)

async def handle_list_policies(session_id: str, is_copilot: bool = False):
    try:
        available_docs = await chatbot.get_available_documents()
        
        if not available_docs:
            response = "❌ **No policies available.**"
        else:
            if is_copilot:
                response = f"📚 **Available Policies ({len(available_docs)}):**\n\n"
                for doc in available_docs:
                    response += f"• **{doc['doc_name']}** (ID: {doc['id']}, FAQs: {doc.get('faq_count', 0)})\n"
                
                response += "\n**Quick Actions:**\n"
                response += "• Type policy name to get details\n"
                response += "• Use `download [policy_name]` to download\n"
                
            else:
                response = f"📚 **Available Policies ({len(available_docs)}):**\n\n"
                for doc in available_docs:
                    response += f"• **{doc['doc_name']}** (ID: {doc['id']})\n"
                    response += f"  - FAQs: {doc.get('faq_count', 0)}\n"
                    response += f"  - Status: {doc.get('doc_status', 'unknown')}\n"
                    response += f"  - Updated: {safe_strftime(doc.get('updated_at'), default='Never')}\n"
                    response += f"  - Size: {doc.get('file_size', 0)} bytes\n\n"
                
                response += "**Commands:**\n"
                response += "• `policy name [id]` - Get policy name by ID\n"
                response += "• `policy id [name]` - Get policy ID by name\n"
                response += "• `download [policy_name]` - Download policy file\n"
        
        await cl.Message(content=response).send()
        await db_manager.store_message(session_id, 'assistant', response)
        
    except Exception as e:
        print(f"Error listing policies: {str(e)}")
        error_response = "❌ Error retrieving policy list. Please try again."
        await cl.Message(content=error_response).send()
        await db_manager.store_message(session_id, 'assistant', error_response)

async def download_policy_file(policy_name: str, session_id: str, is_copilot: bool = False):
    try:
        policy_details = await db_manager.get_policy_details_by_name(policy_name)
        
        if not policy_details:
            response = f"❌ **Policy not found:** '{policy_name}'"
            await cl.Message(content=response).send()
            await db_manager.store_message(session_id, 'assistant', response)
            return
        
        policy_path = policy_details['doc_path']
        policy_full_name = policy_details['doc_name']
        
        if not os.path.exists(policy_path):
            response = f"❌ **File not found:** {policy_full_name}\nPath: {policy_path}"
            await cl.Message(content=response).send()
            await db_manager.store_message(session_id, 'assistant', response)
            return
        
        try:
            file_element = cl.File(
                name=os.path.basename(policy_path),
                path=policy_path,
                display="inline"
            )
            
            if is_copilot:
                response = f"📄 **{policy_full_name}** - Download ready"
            else:
                response = f"📄 **Download Ready:** {policy_full_name}\n\n📁 **File:** {os.path.basename(policy_path)}"
            
            await cl.Message(
                content=response,
                elements=[file_element]
            ).send()
            
            await db_manager.store_message(session_id, 'assistant', f"Downloaded policy: {policy_full_name}")
            
        except Exception as file_error:
            print(f"File creation error: {str(file_error)}")
            response = f"❌ **Download Error:** Could not prepare file for download.\nPolicy: {policy_full_name}"
            await cl.Message(content=response).send()
            await db_manager.store_message(session_id, 'assistant', response)
            
    except Exception as e:
        print(f"Error in download_policy_file: {str(e)}")
        error_response = f"❌ **Error downloading policy:** {policy_name}"
        await cl.Message(content=error_response).send()
        await db_manager.store_message(session_id, 'assistant', error_response)

async def handle_policy_name_lookup(policy_id_str: str, session_id: str, is_copilot: bool = False):
    try:
        policy_id = int(policy_id_str)
        policy_name = await db_manager.get_policy_name_by_id(policy_id)
        
        if policy_name:
            if is_copilot:
                response = f"📄 **ID {policy_id}:** {policy_name}"
            else:
                response = f"📄 **Policy Name for ID {policy_id}:**\n\n**{policy_name}**"
        else:
            response = f"❌ **No policy found with ID:** {policy_id}"
        
        await cl.Message(content=response).send()
        await db_manager.store_message(session_id, 'assistant', response)
        
    except ValueError:
        response = f"❌ **Invalid ID format:** '{policy_id_str}' (must be a number)"
        await cl.Message(content=response).send()
        await db_manager.store_message(session_id, 'assistant', response)
    except Exception as e:
        print(f"Error in policy name lookup: {str(e)}")
        error_response = "❌ **Error looking up policy name.**"
        await cl.Message(content=error_response).send()
        await db_manager.store_message(session_id, 'assistant', error_response)

async def handle_policy_id_lookup(policy_name: str, session_id: str, is_copilot: bool = False):
    try:
        policy_id = await db_manager.get_policy_id_by_name(policy_name)
        
        if policy_id:
            if is_copilot:
                response = f"🔢 **'{policy_name}':** ID {policy_id}"
            else:
                response = f"🔢 **Policy ID for '{policy_name}':**\n\n**ID: {policy_id}**"
        else:
            response = f"❌ **No policy found matching:** '{policy_name}'"
        
        await cl.Message(content=response).send()
        await db_manager.store_message(session_id, 'assistant', response)
        
    except Exception as e:
        print(f"Error in policy ID lookup: {str(e)}")
        error_response = "❌ **Error looking up policy ID.**"
        await cl.Message(content=error_response).send()
        await db_manager.store_message(session_id, 'assistant', error_response)

async def check_policy_mention(user_message: str) -> Optional[Dict]:
    try:
        available_docs = await chatbot.get_available_documents()
        user_lower = user_message.lower()
        
        for doc in available_docs:
            doc_name_lower = doc['doc_name'].lower()
            if doc_name_lower in user_lower or any(word in user_lower for word in doc_name_lower.split() if len(word) > 3):
                return {
                    'id': doc['id'],
                    'name': doc['doc_name'],
                    'path': doc['doc_path'],
                    'faq_count': doc.get('faq_count', 0)
                }
        return None
    except Exception as e:
        print(f"Error checking policy mention: {str(e)}")
        return None

async def suggest_policy_actions(policy_match: Dict, user_message: str):
    try:
        policy_name = policy_match['name']
        policy_id = policy_match['id']
        faq_count = policy_match.get('faq_count', 0)
        
        actions = [
            cl.Action(
                name="select_policy",
                value=policy_name,
                label=f"📄 Select {policy_name}",
                description=f"Focus on {policy_name} ({faq_count} FAQs)"
            ),
            cl.Action(
                name="download_policy",
                value=policy_name,
                label=f"⬇️ Download {policy_name}",
                description=f"Download policy document"
            )
        ]
        
        response = f"🎯 **Detected Policy:** {policy_name}\n\n📊 **FAQs Available:** {faq_count}\n\n**Quick Actions:**"
        
        await cl.Message(
            content=response,
            actions=actions
        ).send()
        
        session_id = cl.user_session.get("session_id")
        conversation_history = await db_manager.get_conversation_history(session_id, limit=5)
        
        policy_response = await chatbot.generate_response(
            user_message, 
            conversation_history, 
            doc_id=policy_id,
            is_copilot=True
        )
        
        await cl.Message(content=f"**About {policy_name}:**\n\n{policy_response}").send()
        
        if session_id:
            await db_manager.store_message(session_id, 'assistant', response)
            await db_manager.store_message(session_id, 'assistant', f"About {policy_name}: {policy_response}")
        
    except Exception as e:
        print(f"Error suggesting policy actions: {str(e)}")
        session_id = cl.user_session.get("session_id")
        conversation_history = await db_manager.get_conversation_history(session_id, limit=5)
        response = await chatbot.generate_response(user_message, conversation_history, is_copilot=True)
        await cl.Message(content=response).send()

async def handle_policy_selection_by_name(policy_name: str, is_copilot: bool = False):
    try:
        policy_details = await db_manager.get_policy_details_by_name(policy_name)
        
        if not policy_details:
            response = f"❌ **Policy not found:** '{policy_name}'"
            await cl.Message(content=response).send()
            return
        
        cl.user_session.set("doc_id", policy_details['id'])
        cl.user_session.set("doc_name", policy_details['doc_name'])
        
        if is_copilot:
            response = f"✅ **Selected:** {policy_details['doc_name']}\n\n🎯 **Ready to answer questions about this policy**"
        else:
            response = (
                f"✅ **Policy Selected:** {policy_details['doc_name']}\n\n"
                f"🆔 **ID:** {policy_details['id']}\n"
                f"📁 **Path:** {policy_details['doc_path']}\n\n"
                f"I'll now focus my responses on this specific policy."
            )
        
        await cl.Message(content=response).send()
        
        session_id = cl.user_session.get("session_id")
        if session_id:
            await db_manager.store_message(session_id, 'assistant', response)
        
    except Exception as e:
        print(f"Error in handle_policy_selection_by_name: {str(e)}")
        error_response = f"❌ **Error selecting policy:** {policy_name}"
        await cl.Message(content=error_response).send()

async def cleanup():
    global connection_pool
    if connection_pool:
        await connection_pool.close()

if __name__ == "__main__":
    import atexit
    atexit.register(lambda: asyncio.run(cleanup()))
    cl.run()