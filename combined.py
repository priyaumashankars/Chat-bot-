import os
import asyncpg
import openai
import PyPDF2
import uuid
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from dotenv import load_dotenv
import chainlit as cl
import tempfile
import asyncio
from chainlit.data.chainlit_data_layer import ChainlitDataLayer
from typing import Any, Dict, List, Optional, Union
import shutil
 
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
            print(f"Database query error: {e}")
            print(f"Query: {query}")
            print(f"Params: {params}")
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
            print(f"Error in create_step: {e}")
            return None
 
# Load environment variables
load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
 
# Override default Chainlit data layer with DATABASE_URL
if DATABASE_URL:
    cl.data_layer = CustomChainlitDataLayer(database_url=DATABASE_URL)
 
# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)
 
# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
 
# Global connection pool
connection_pool = None
 
async def init_connection_pool():
    """Initialize async connection pool"""
    global connection_pool
    if connection_pool is None:
        try:
            connection_pool = await asyncpg.create_pool(
                DATABASE_URL,
                min_size=1,
                max_size=10,
                command_timeout=30,
                server_settings={
                    'jit': 'off'
                }
            )
            print("Database connection pool initialized")
        except Exception as e:
            print(f"Error initializing connection pool: {e}")
            raise
 
async def get_db_connection():
    """Get database connection from pool with timeout"""
    global connection_pool
    if connection_pool is None:
        await init_connection_pool()
    
    try:
        return await asyncio.wait_for(
            connection_pool.acquire(),
            timeout=10.0
        )
    except asyncio.TimeoutError:
        print("Database connection timeout")
        raise
    except Exception as e:
        print(f"Error acquiring database connection: {e}")
        raise
 
class DatabaseManager:
    def __init__(self):
        pass
    
    async def init_database(self):
        """Create tables if they don't exist"""
        conn = None
        try:
            conn = await get_db_connection()
            
            # Create faq_data table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS faq_data (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    embedding FLOAT8[] NOT NULL,
                    doc_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create document_status table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS document_status (
                    id SERIAL PRIMARY KEY,
                    doc_id INTEGER NOT NULL,
                    doc_status VARCHAR(50) NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
            
            # Create chat_messages table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    role VARCHAR(50) NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            print("Database tables initialized")
            
        except Exception as e:
            print(f"Error initializing database: {e}")
            raise
        finally:
            if conn:
                await connection_pool.release(conn)
 
class DocumentProcessor:
    @staticmethod
    def extract_text_from_pdf(file_path):
        """Extract raw text from PDF and refine it via OpenAI model"""
        # Step 1: Extract raw text locally
        raw_text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    raw_text += page.extract_text() + "\n"
        except Exception as e:
            return f"Error reading PDF: {e}"
        
        # Step 2: Send extracted text to OpenAI for refinement/summarization
        if not raw_text.strip():
            return "No text found in PDF."
 
        # Prepare prompt to clean or summarize text
        prompt = f"Please extract and clean the important textual content from the following PDF text. Remove any formatting artifacts and present only the meaningful content:\n\n{raw_text}\n\nExtracted and cleaned text:"
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts and cleans important text from PDFs, removing formatting artifacts while preserving meaningful content."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.1,
            )
            extracted_text = response.choices[0].message.content.strip()
            return extracted_text
        except Exception as e:
            return f"OpenAI API error: {e}"
 
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
            print(f"Error generating embedding: {e}")
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
            print(f"Error getting document content: {e}")
            return None
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
                    WHERE embedding IS NOT NULL AND doc_id = $1
                """, doc_id)
            else:
                faqs = await conn.fetch("""
                    SELECT question, answer, embedding
                    FROM faq_data
                    WHERE embedding IS NOT NULL
                """)
            
            # Convert to list of dicts for compatibility
            faqs_list = [dict(faq) for faq in faqs]
            
            # Find relevant context
            relevant_context = self.find_relevant_context(question, faqs_list)
            
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
            3. If the question is not covered in the document, politely say so
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
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=800,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error while processing your question. Please try again."
        finally:
            if conn:
                await connection_pool.release(conn)
    
    def generate_faqs_from_text(self, text):
        """Generate FAQs from document text"""
        # Limit text to avoid token limits
        if len(text) > 8000:
            limited_text = text[:4000] + "\n...\n" + text[-4000:]
        else:
            limited_text = text
        
        if len(limited_text.strip()) < 200:
            print("Insufficient text content for FAQ generation")
            return []
        
        prompt = f"""
        CRITICAL INSTRUCTIONS - READ CAREFULLY:
        
        You are analyzing a specific document. You must create questions and answers using ONLY the information explicitly written in this document.
        
        STRICT REQUIREMENTS:
        1. Read ONLY the document content below
        2. Create questions about topics that are EXPLICITLY mentioned in this document
        3. Every answer must start with "According to the document:" or "The document states:"
        4. Every answer must contain actual information from the document - do not add external knowledge
        5. If the document doesn't have enough specific information, create fewer questions (even just 2-3 is fine)
        6. Do NOT create questions about general topics - only about what this specific document discusses
        
        DOCUMENT CONTENT TO ANALYZE:
        ===START DOCUMENT===
        {limited_text}
        ===END DOCUMENT===
        
        Based ONLY on the above document content, create 3-5 questions and answers in this exact JSON format:
        
        [
            {{
                "question": "What does the document say about [specific thing mentioned in document]?",
                "answer": "According to the document: [specific information from the document]"
            }}
        ]
        
        Remember: Only ask about what is specifically written in the document above. Do not use any external knowledge.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a document analyzer that creates FAQs strictly from provided document content. You are forbidden from using any external knowledge."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.0
            )
            
            faq_text = response.choices[0].message.content.strip()
            
            # Extract JSON from the response
            start_idx = faq_text.find('[')
            end_idx = faq_text.rfind(']') + 1
            
            if start_idx == -1 or end_idx == 0:
                print("No JSON array found in response")
                return []
                
            json_str = faq_text[start_idx:end_idx]
            faqs = json.loads(json_str)
            
            return faqs if isinstance(faqs, list) else []
            
        except Exception as e:
            print(f"Error generating FAQs: {e}")
            return []
 
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
        
        # Store session in database
        conn = None
        try:
            conn = await get_db_connection()
            await conn.execute("""
                INSERT INTO chat_sessions (session_id, doc_id)
                VALUES ($1, $2)
            """, session_id, None)
        except Exception as e:
            print(f"Error creating session: {e}")
        finally:
            if conn:
                await connection_pool.release(conn)
        
        # Send welcome message
        await cl.Message(
            content="👋 Welcome! I'm your PDF document assistant. Please upload a PDF file to get started, or ask me questions about any previously uploaded documents.",
        ).send()
        
    except Exception as e:
        print(f"Error in chat start: {e}")
        await cl.Message(
            content="Sorry, there was an error initializing the chat. Please refresh the page."
        ).send()
 
@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    
    # Check if message contains files
    if hasattr(message, 'elements') and message.elements:
        for element in message.elements:
            if hasattr(element, 'path') and element.path.endswith('.pdf'):
                await handle_file_upload(element.path, element.name)
                return
    
    # Check for file upload request
    user_message = message.content.strip().lower()
    if any(keyword in user_message for keyword in ['upload', 'file', 'pdf', 'document']):
        await cl.Message(
            content="Please use the attachment button (📎) in the message input to upload your PDF file, or copy and paste the file content if you're unable to upload directly."
        ).send()
        return
    
    # Check if user is trying to paste file content
    content = message.content.strip()
    if len(content) > 1000 and any(indicator in content.lower() for indicator in ['pdf', 'document', 'page', 'chapter']):
        await handle_manual_upload(content)
        return
    
    # Handle regular chat
    session_id = cl.user_session.get("session_id")
    doc_id = cl.user_session.get("doc_id")
    
    if not session_id:
        await cl.Message(content="Session error. Please refresh the page.").send()
        return
    
    # Store user message and get conversation history
    conn = None
    try:
        conn = await get_db_connection()
        
        # Store user message
        await conn.execute("""
            INSERT INTO chat_messages (session_id, role, message)
            VALUES ($1, $2, $3)
        """, session_id, 'user', message.content)
        
        # Get conversation history
        history_rows = await conn.fetch("""
            SELECT role, message FROM chat_messages
            WHERE session_id = $1
            ORDER BY timestamp ASC
        """, session_id)
        
        conversation_history = [{'role': row['role'], 'message': row['message']} for row in history_rows]
        
    except Exception as e:
        print(f"Error handling message: {e}")
        conversation_history = []
    finally:
        if conn:
            await connection_pool.release(conn)
    
    # Generate response
    try:
        response = await chatbot.generate_response(
            message.content,
            conversation_history,
            doc_id
        )
        
        # Store assistant response
        conn = None
        try:
            conn = await get_db_connection()
            await conn.execute("""
                INSERT INTO chat_messages (session_id, role, message)
                VALUES ($1, $2, $3)
            """, session_id, 'assistant', response)
        except Exception as e:
            print(f"Error storing response: {e}")
        finally:
            if conn:
                await connection_pool.release(conn)
        
        await cl.Message(content=response).send()
        
    except Exception as e:
        print(f"Error in main message handler: {e}")
        await cl.Message(
            content=f"I'm sorry, I encountered an error while processing your question: {str(e)}"
        ).send()
 
async def handle_file_upload(file_path, filename):
    """Handle PDF file upload and processing"""
    if not filename.lower().endswith('.pdf'):
        await cl.Message(content="❌ Only PDF files are supported.").send()
        return
    
    conn = None
    try:
        # Show processing message
        await cl.Message(content="📄 Processing your PDF document...").send()
        
        # Extract text from PDF
        text = DocumentProcessor.extract_text_from_pdf(file_path)
        
        if not text or len(text.strip()) < 200:
            await cl.Message(content="❌ Could not extract sufficient text from PDF. Please ensure the PDF contains readable text.").send()
            return
        
        # Save to permanent location
        permanent_filename = str(uuid.uuid4()) + '_' + filename
        permanent_path = os.path.join(UPLOAD_FOLDER, permanent_filename)
        shutil.copy2(file_path, permanent_path)
        
        # Insert document into database
        conn = await get_db_connection()
        
        doc_id = await conn.fetchval("""
            INSERT INTO doc_data (doc_name, doc_path, doc_content, doc_status)
            VALUES ($1, $2, $3, $4) RETURNING id
        """, filename, permanent_path, text, 'processing')
        
        await cl.Message(content="🤖 Generating FAQs from document...").send()
        
        # Generate FAQs from the extracted text
        faqs = chatbot.generate_faqs_from_text(text)
        
        if not faqs:
            await conn.execute("UPDATE doc_data SET doc_status = $1 WHERE id = $2", 'failed', doc_id)
            await cl.Message(content="❌ Could not generate FAQs from document. The document might not contain sufficient structured information.").send()
            return
        
        # Store FAQs with embeddings
        stored_faqs = 0
        for faq in faqs:
            embedding = chatbot.generate_embedding(faq['question'])
            if embedding:
                await conn.execute("""
                    INSERT INTO faq_data (question, answer, embedding, doc_id)
                    VALUES ($1, $2, $3, $4)
                """, faq['question'], faq['answer'], embedding, doc_id)
                stored_faqs += 1
        
        # Update document status and session
        await conn.execute("UPDATE doc_data SET doc_status = $1 WHERE id = $2", 'completed', doc_id)
        
        session_id = cl.user_session.get("session_id")
        await conn.execute("UPDATE chat_sessions SET doc_id = $1 WHERE session_id = $2", doc_id, session_id)
        
        # Update session variables
        cl.user_session.set("doc_id", doc_id)
        cl.user_session.set("doc_name", filename)
        
        await cl.Message(
            content=f"✅ **Document processed successfully!**\n\n"
                   f"📋 **Document:** {filename}\n"
                   f"🤖 **FAQs Generated:** {stored_faqs}\n\n"
                   f"You can now ask me questions about this document!"
        ).send()
        
    except Exception as e:  
        print(f"Error processing document: {e}")
        await cl.Message(content=f"❌ Error processing document: {str(e)}").send()
    finally:
        if conn:
            await connection_pool.release(conn)
 
async def handle_manual_upload(content):
    """Handle manual file content input"""
    session_id = cl.user_session.get("session_id")
    conn = None
    
    try:
        doc_name = f"Pasted_Document_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        await cl.Message(content="📄 Processing your document content...").send()
        
        conn = await get_db_connection()
        
        doc_id = await conn.fetchval("""
            INSERT INTO doc_data (doc_name, doc_path, doc_content, doc_status)
            VALUES ($1, $2, $3, $4) RETURNING id
        """, doc_name, "", content, 'processing')
        
        await cl.Message(content="🤖 Generating FAQs from document...").send()
        
        # Generate FAQs
        faqs = chatbot.generate_faqs_from_text(content)
        
        if faqs:
            stored_faqs = 0
            for faq in faqs:
                embedding = chatbot.generate_embedding(faq['question'])
                if embedding:
                    await conn.execute("""
                        INSERT INTO faq_data (question, answer, embedding, doc_id)
                        VALUES ($1, $2, $3, $4)
                    """, faq['question'], faq['answer'], embedding, doc_id)
                    stored_faqs += 1
            
            await conn.execute("UPDATE doc_data SET doc_status = $1 WHERE id = $2", 'completed', doc_id)
            await conn.execute("UPDATE chat_sessions SET doc_id = $1 WHERE session_id = $2", doc_id, session_id)
            
            cl.user_session.set("doc_id", doc_id)
            cl.user_session.set("doc_name", doc_name)
            
            await cl.Message(
                content=f"✅ **Document content processed!**\n\n"
                       f"🤖 **FAQs Generated:** {stored_faqs}\n\n"
                       f"You can now ask me questions about this content!"
            ).send()
        else:
            await conn.execute("UPDATE doc_data SET doc_status = $1 WHERE id = $2", 'failed', doc_id)
            await cl.Message(content="❌ Could not generate FAQs from the provided content.").send()
            
    except Exception as e:
        print(f"Error processing pasted content: {e}")
        await cl.Message(content=f"❌ Error processing document content: {str(e)}").send()
    finally:
        if conn:
            await connection_pool.release(conn)
 
# Cleanup function for graceful shutdown
async def cleanup():
    """Cleanup resources on shutdown"""
    global connection_pool
    if connection_pool:
        await connection_pool.close()
        print("Database connection pool closed")
 
# Register cleanup function
import atexit
atexit.register(lambda: asyncio.run(cleanup()) if connection_pool else None)
 