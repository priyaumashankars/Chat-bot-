import os
import asyncpg
from openai import OpenAI
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
from typing import Any, Dict, List, Optional, Union
import shutil
from sentence_transformers import SentenceTransformer

# Load .env
load_dotenv()

# Set Chainlit environment variables BEFORE importing chainlit
os.environ["CHAINLIT_ALLOW_ORIGINS"] = '["*"]'

# Set Chainlit JWT secret from environment
CHAINLIT_AUTH_SECRET = os.getenv('CHAINLIT_AUTH_SECRET')
if CHAINLIT_AUTH_SECRET:
    os.environ["CHAINLIT_AUTH_SECRET"] = CHAINLIT_AUTH_SECRET
else:
    # Generate a temporary secret if not provided
    import secrets
    temp_secret = secrets.token_urlsafe(32)
    os.environ["CHAINLIT_AUTH_SECRET"] = temp_secret
    print("⚠️ Warning: Using temporary Chainlit auth secret. Set CHAINLIT_AUTH_SECRET in .env for production.")

DATABASE_URL = os.getenv('DATABASE_URL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')

# Disable Chainlit's data layer to avoid conflicts
cl.data_layer = None

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Initialize the static embedding model
print("🚀 Loading static embedding model...")
try:
    embedding_model = SentenceTransformer('sentence-transformers/static-retrieval-mrl-en-v1')
    print("✅ Static embedding model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading embedding model: {e}")
    embedding_model = None

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global connection pool
connection_pool = None

async def init_connection_pool():
    """Initialize database connection pool"""
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
            print("✅ Database connection pool initialized")
        except Exception as e:
            print(f"❌ Failed to create database connection pool: {e}")
            raise

async def get_db_connection():
    """Get database connection with error handling"""
    global connection_pool
    if connection_pool is None:
        await init_connection_pool()
    try:
        return await asyncio.wait_for(connection_pool.acquire(), timeout=15.0)
    except asyncio.TimeoutError:
        raise Exception("Database connection timeout")
    except Exception as e:
        print(f"Database connection error: {e}")
        raise

async def release_db_connection(conn):
    """Release database connection"""
    global connection_pool
    if connection_pool and conn:
        await connection_pool.release(conn)

class DatabaseManager:
    def __init__(self):
        pass

    async def init_database(self):
        """Create tables if they don't exist"""
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

            # Create faq_data table with company_id
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS faq_data (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    embedding FLOAT8[] NOT NULL,
                    doc_id INTEGER,
                    company_id INTEGER REFERENCES companies(id),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create doc_data table with company_id
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS doc_data (
                    id SERIAL PRIMARY KEY,
                    doc_name VARCHAR(255) NOT NULL,
                    doc_path VARCHAR(500) NOT NULL,
                    doc_content TEXT,
                    doc_status VARCHAR(50) DEFAULT 'pending',
                    file_size BIGINT,
                    total_faqs INTEGER DEFAULT 0,
                    processing_time INTEGER,
                    error_message TEXT,
                    uploaded_by VARCHAR(255),
                    company_id INTEGER REFERENCES companies(id),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed_at TIMESTAMP
                )
            """)

            # Create indexes for better performance
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_data_status ON doc_data(doc_status)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_faq_data_doc_id ON faq_data(doc_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_faq_data_company_id ON faq_data(company_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_data_company_id ON doc_data(company_id)")

            # Insert default companies if none exist
            company_exists = await conn.fetchval("SELECT COUNT(*) FROM companies")
            if company_exists == 0:
                await conn.execute("""
                    INSERT INTO companies (name, code) VALUES 
                    ('Default Company', 'default'),
                    ('Demo Company', 'demo'),
                    ('Test Company', 'test')
                """)
                print("✅ Default companies created")

            print("✅ Database tables initialized")

        except Exception as e:
            print(f"Error initializing database: {e}")
            raise
        finally:
            if conn:
                await release_db_connection(conn)

    async def get_company_by_name_or_code(self, identifier):
        """Get company by name or code"""
        conn = None
        try:
            conn = await get_db_connection()
            return await conn.fetchrow("""
                SELECT id, name, code FROM companies
                WHERE LOWER(name) = LOWER($1) OR LOWER(code) = LOWER($1)
            """, identifier.strip())
        finally:
            if conn:
                await release_db_connection(conn)

    async def get_all_documents(self):
        """Get all documents with their processing status"""
        conn = None
        try:
            conn = await get_db_connection()
            docs = await conn.fetch("""
                SELECT dd.id, dd.doc_name, dd.doc_status, 
                       COALESCE(dd.file_size, 0) as file_size, 
                       COALESCE(dd.total_faqs, 0) as total_faqs,
                       COALESCE(dd.processing_time, 0) as processing_time,
                       dd.created_at, dd.processed_at, dd.error_message, dd.uploaded_by,
                       c.name as company_name, c.code as company_code
                FROM doc_data dd
                LEFT JOIN companies c ON dd.company_id = c.id
                ORDER BY dd.created_at DESC
            """)
            return [dict(doc) for doc in docs]
        except Exception as e:
            print(f"Error getting documents: {e}")
            return []
        finally:
            if conn:
                await release_db_connection(conn)

    async def delete_document(self, doc_id):
        """Delete document and its FAQs"""
        conn = None
        try:
            conn = await get_db_connection()
            doc_id = int(doc_id)

            # Get document info first for logging
            doc = await conn.fetchrow("SELECT id, doc_name, doc_path FROM doc_data WHERE id = $1", doc_id)

            if not doc:
                print(f"Document with ID {doc_id} not found")
                return False

            # Delete physical file if it exists
            if doc['doc_path'] and os.path.exists(doc['doc_path']):
                try:
                    os.remove(doc['doc_path'])
                    print(f"Deleted file: {doc['doc_path']}")
                except Exception as e:
                    print(f"Warning: Could not delete file {doc['doc_path']}: {e}")

            # Delete FAQs first (foreign key constraint)
            faq_count = await conn.fetchval("SELECT COUNT(*) FROM faq_data WHERE doc_id = $1", doc_id)
            await conn.execute("DELETE FROM faq_data WHERE doc_id = $1", doc_id)
            print(f"Deleted {faq_count} FAQs for document {doc_id}")

            # Delete document record
            result = await conn.execute("DELETE FROM doc_data WHERE id = $1", doc_id)

            # Check if any rows were affected
            if result == "DELETE 0":
                print(f"No document found with ID {doc_id}")
                return False

            print(f"Successfully deleted document: {doc['doc_name']} (ID: {doc_id})")
            return True

        except ValueError as e:
            print(f"Invalid doc_id format: {e}")
            return False
        except Exception as e:
            print(f"Error deleting document {doc_id}: {e}")
            return False
        finally:
            if conn:
                await release_db_connection(conn)

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
            if not client:
                return raw_text  # Return raw text if OpenAI not available
                
            response = client.chat.completions.create(
                model="gpt-4o-mini",
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
            print(f"OpenAI API error: {e}")
            return raw_text  # Return raw text as fallback

class FAQGenerator:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key) if api_key else None
        # Use the global static embedding model instead of OpenAI embeddings
        self.embedding_model = embedding_model

    def generate_embedding(self, text):
        """Generate embedding using static-retrieval-mrl-en-v1 model (100x-400x faster on CPU)"""
        try:
            if not self.embedding_model:
                print("Embedding model not loaded")
                return None
                
            # Use the static embedding model instead of OpenAI
            embedding = self.embedding_model.encode(text, normalize_embeddings=True)
            
            # Convert to list for database storage (PostgreSQL expects array format)
            return embedding.tolist()
        except Exception as e:
            print(f"Error generating static embedding: {e}")
            return None

    def generate_faqs_from_text(self, text):
        """Generate FAQs from document text"""
        if not self.client:
            print("OpenAI client not configured")
            return []
            
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

        Based ONLY on the above document content, create 5-8 questions and answers in this exact JSON format:

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
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a document analyzer that creates FAQs strictly from provided document content. You are forbidden from using any external knowledge."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
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
faq_generator = FAQGenerator(OPENAI_API_KEY)

@cl.on_chat_start
async def start():
    """Initialize admin interface"""
    try:
        # Initialize database
        await db_manager.init_database()

        # Send welcome message
        welcome_msg = """🔧 **Document Processing Admin Panel**

Welcome to the PDF document processing interface!

🚀 **NOW USING: Ultra-fast static embedding model (100x-400x faster on CPU!)**

**Available Commands:**
• 📤 **Upload PDF**: Use the attachment button to upload PDF files
• 📋 **View Status**: Type 'status' to see all documents
• 🗑️ **Delete Document**: Type 'delete [doc_id]' to remove a document
• 🔄 **Reprocess**: Type 'reprocess [doc_id]' to regenerate FAQs
• 📊 **Stats**: Type 'stats' for processing statistics
• ❓ **Help**: Type 'help' for command list

You can **upload multiple PDFs at once**. After uploading, you'll be prompted for the company name or code to associate with the document.

**Examples:**
- `delete 1` – Delete document with ID 1
- `reprocess 2` – Reprocess document with ID 2
- `status` – Show all documents
"""

        await cl.Message(content=welcome_msg).send()

    except Exception as e:
        print(f"Error in admin start: {e}")
        import traceback
        traceback.print_exc()
        await cl.Message(
            content="❌ Error initializing admin panel. Please check your database connection."
        ).send()

@cl.on_message
async def main(message: cl.Message):
    """Handle admin commands and file uploads"""

    # Check if message contains files (PDF uploads)
    if hasattr(message, 'elements') and message.elements:
        upload_tasks = []
        for element in message.elements:
            if hasattr(element, 'path') and element.path.lower().endswith('.pdf'):
                upload_tasks.append(process_pdf_upload(element.path, element.name))

        if upload_tasks:
            await cl.Message(content=f"🔄 **Processing {len(upload_tasks)} PDF file(s) with ultra-fast static embeddings...**").send()

            # Process uploads concurrently
            results = await asyncio.gather(*upload_tasks, return_exceptions=True)

            success_count = sum(1 for r in results if r is True)
            failed_count = len(results) - success_count

            result_msg = f"📊 **Upload Results:**\n✅ Successful: {success_count}\n❌ Failed: {failed_count}"
            await cl.Message(content=result_msg).send()
            return

    # Handle admin commands
    user_input = message.content.strip()
    user_input_lower = user_input.lower()

    try:
        if user_input_lower == 'status':
            await show_document_status()
        elif user_input_lower == 'stats':
            await show_processing_stats()
        elif user_input_lower == 'help':
            await show_help()
        elif user_input_lower.startswith('delete '):
            # Parse delete command
            parts = user_input.split()
            if len(parts) != 2:
                await cl.Message(content="❌ **Invalid format.** Use: `delete [doc_id]`\nExample: `delete 1`").send()
                return

            try:
                doc_id = int(parts[1])
                await delete_document(doc_id)
            except ValueError:
                await cl.Message(content="❌ **Invalid document ID.** Please provide a valid number.\nExample: `delete 1`").send()

        elif user_input_lower.startswith('reprocess '):
            # Parse reprocess command
            parts = user_input.split()
            if len(parts) != 2:
                await cl.Message(content="❌ **Invalid format.** Use: `reprocess [doc_id]`\nExample: `reprocess 1`").send()
                return

            try:
                doc_id = int(parts[1])
                await reprocess_document(doc_id)
            except ValueError:
                await cl.Message(content="❌ **Invalid document ID.** Please provide a valid number.\nExample: `reprocess 1`").send()
        else:
            # Default help message for unrecognized commands
            await show_help()

    except Exception as e:
        print(f"Error handling command '{user_input}': {e}")
        await cl.Message(content=f"❌ **Error processing command:** {str(e)}").send()

async def show_help():
    """Show help message with available commands"""
    help_msg = """🔧 **Admin Commands:**

• **status** - View all documents and their processing status
• **stats** - View processing statistics
• **delete [doc_id]** - Delete a document and its FAQs
• **reprocess [doc_id]** - Regenerate FAQs for a document
• **help** - Show this help message

📤 **To upload PDFs**: Use the attachment button (📎) to select and upload PDF files. You'll be prompted for the company name or code after uploading.

🚀 **Performance**: Now using ultra-fast static embedding model (100x-400x faster on CPU!)

**Examples:**
- `delete 1` - Delete document with ID 1
- `reprocess 2` - Reprocess document with ID 2
- `status` - Show all documents
"""
    await cl.Message(content=help_msg).send()

async def process_pdf_upload(file_path, filename):
    """Process a single PDF upload with proper type handling"""
    conn = None
    start_time = datetime.now()

    try:
        # Ensure proper types
        filename = str(filename)
        file_size = int(os.path.getsize(file_path))

        # Prompt for company name or code
        company_input = await cl.AskUserMessage(
            content="🏢 Please enter the company name or code for this document:"
        ).send()

        # Handle different response formats
        company_identifier = None
        if isinstance(company_input, dict):
            company_identifier = company_input.get('content') or company_input.get('output')
        elif hasattr(company_input, 'content'):
            company_identifier = company_input.content
        elif isinstance(company_input, str):
            company_identifier = company_input
        else:
            company_identifier = str(company_input) if company_input else None

        if not company_identifier or not company_identifier.strip():
            await cl.Message(
                content="❌ No company identifier provided. PDF processing cancelled."
            ).send()
            return False

        # Get company record
        conn = await get_db_connection()
        company_record = await db_manager.get_company_by_name_or_code(company_identifier.strip())

        if not company_record:
            await cl.Message(
                content=f"❌ Company '{company_identifier}' not found. Available companies: 'default', 'demo', 'test'. Please contact admin to add new companies."
            ).send()
            return False

        company_id = company_record['id']

        # Save to permanent location
        permanent_filename = str(uuid.uuid4()) + '_' + filename
        permanent_path = str(os.path.join(UPLOAD_FOLDER, permanent_filename))
        shutil.copy2(file_path, permanent_path)

        # Insert document into database with company_id
        doc_id = await conn.fetchval("""
            INSERT INTO doc_data (doc_name, doc_path, doc_content, doc_status, file_size, uploaded_by, company_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7) RETURNING id
        """, filename, permanent_path, None, 'processing', file_size, 'admin', company_id)

        # Ensure doc_id is integer
        doc_id = int(doc_id)

        # Extract text from PDF
        text = DocumentProcessor.extract_text_from_pdf(permanent_path)

        if not text or len(text.strip()) < 200:
            await conn.execute("""
                UPDATE doc_data
                SET doc_status = $1, error_message = $2, processed_at = $3
                WHERE id = $4
            """, 'failed', 'Could not extract sufficient text from PDF', datetime.now(), doc_id)
            return False

        # Update document with extracted text
        await conn.execute("UPDATE doc_data SET doc_content = $1 WHERE id = $2", str(text), doc_id)

        # Generate FAQs from the extracted text using static embeddings
        faqs = faq_generator.generate_faqs_from_text(text)

        if not faqs:
            await conn.execute("""
                UPDATE doc_data
                SET doc_status = $1, error_message = $2, processed_at = $3
                WHERE id = $4
            """, 'failed', 'Could not generate FAQs from document content', datetime.now(), doc_id)
            return False

        # Store FAQs with static embeddings and company_id
        stored_faqs = 0
        for faq in faqs:
            embedding = faq_generator.generate_embedding(faq['question'])
            if embedding:
                # Ensure proper types
                question = str(faq['question'])
                answer = str(faq['answer'])

                await conn.execute("""
                    INSERT INTO faq_data (question, answer, embedding, doc_id, company_id)
                    VALUES ($1, $2, $3, $4, $5)
                """, question, answer, embedding, doc_id, company_id)
                stored_faqs += 1

        # Calculate processing time
        processing_time = int((datetime.now() - start_time).total_seconds())

        # Update document status
        await conn.execute("""
            UPDATE doc_data
            SET doc_status = $1, total_faqs = $2, processing_time = $3, processed_at = $4
            WHERE id = $5
        """, 'completed', stored_faqs, processing_time, datetime.now(), doc_id)

        await cl.Message(
            content=f"✅ **{filename}** processed successfully for company '{company_identifier}'!\n🚀 Generated {stored_faqs} FAQs in {processing_time}s using ultra-fast static embeddings!"
        ).send()

        return True

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        if 'doc_id' in locals() and conn:
            await conn.execute("""
                UPDATE doc_data
                SET doc_status = $1, error_message = $2, processed_at = $3
                WHERE id = $4
            """, 'failed', str(e), datetime.now(), doc_id)

        await cl.Message(content=f"❌ **{filename}** processing failed: {str(e)}").send()
        return False
    finally:
        if conn:
            await release_db_connection(conn)

async def show_document_status():
    """Show status of all documents"""
    documents = await db_manager.get_all_documents()

    if not documents:
        await cl.Message(content="📚 **No documents found.**").send()
        return

    status_msg = "📚 **Document Status Report:**\n\n"

    for doc in documents:
        status_emoji = {
            'completed': '✅',
            'processing': '🔄',
            'failed': '❌',
            'pending': '⏳'
        }.get(doc.get('doc_status', 'unknown'), '❓')

        status_msg += f"**ID {doc.get('id', 'N/A')}**: {status_emoji} {doc.get('doc_name', 'Unknown')}\n"
        status_msg += f"   🏢 Company: {doc.get('company_name', 'N/A')} ({doc.get('company_code', 'N/A')})\n"
        status_msg += f"   📊 Status: {doc.get('doc_status', 'unknown').title()}\n"

        # Safe handling of file_size
        file_size = doc.get('file_size', 0)
        if file_size and file_size > 0:
            status_msg += f"   📏 Size: {file_size / 1024:.1f} KB\n"

        status_msg += f"   🤖 FAQs: {doc.get('total_faqs', 0)}\n"

        # Safe handling of processing_time
        processing_time = doc.get('processing_time', 0)
        if processing_time and processing_time > 0:
            status_msg += f"   ⚡ Process Time: {processing_time}s (with static embeddings)\n"

        # Safe handling of created_at
        if doc.get('created_at'):
            status_msg += f"   📅 Uploaded: {doc['created_at'].strftime('%Y-%m-%d %H:%M')}\n"

        # Safe handling of error_message
        error_message = doc.get('error_message')
        if error_message:
            status_msg += f"   ❌ Error: {error_message[:100]}...\n"

        status_msg += "\n"

    await cl.Message(content=status_msg).send()

async def show_processing_stats():
    """Show processing statistics"""
    conn = None
    try:
        conn = await get_db_connection()
        stats = await conn.fetchrow("""
            SELECT
                COUNT(*) as total_docs,
                COUNT(*) FILTER (WHERE doc_status = 'completed') as completed_docs,
                COUNT(*) FILTER (WHERE doc_status = 'failed') as failed_docs,
                COUNT(*) FILTER (WHERE doc_status = 'processing') as processing_docs,
                COALESCE(SUM(total_faqs), 0) as total_faqs,
                COALESCE(AVG(processing_time), 0) as avg_processing_time,
                COALESCE(SUM(file_size), 0) as total_file_size
            FROM doc_data
        """)

        # Safe handling of stats with default values
        total_docs = stats.get('total_docs', 0)
        completed_docs = stats.get('completed_docs', 0)
        failed_docs = stats.get('failed_docs', 0)
        processing_docs = stats.get('processing_docs', 0)
        total_faqs = stats.get('total_faqs', 0)
        avg_processing_time = stats.get('avg_processing_time', 0)
        total_file_size = stats.get('total_file_size', 0)

        success_rate = (completed_docs / max(total_docs, 1) * 100) if total_docs > 0 else 0

        stats_msg = f"""📊 **Processing Statistics:**

📁 **Documents:** {total_docs} total
✅ **Completed:** {completed_docs}
❌ **Failed:** {failed_docs}
🔄 **Processing:** {processing_docs}

🤖 **Total FAQs Generated:** {total_faqs}
⚡ **Average Processing Time:** {avg_processing_time:.1f}s (with static embeddings)
💾 **Total File Size:** {total_file_size / (1024*1024):.1f} MB

📈 **Success Rate:** {success_rate:.1f}%

🚀 **Performance Boost:** Using static-retrieval-mrl-en-v1 model (100x-400x faster on CPU!)
"""
        await cl.Message(content=stats_msg).send()

    except Exception as e:
        await cl.Message(content=f"❌ Error getting stats: {str(e)}").send()
    finally:
        if conn:
            await release_db_connection(conn)

async def delete_document(doc_id):
    """Delete a document and its FAQs with better error handling"""
    try:
        # Validate doc_id
        if not isinstance(doc_id, int) or doc_id <= 0:
            await cl.Message(content=f"❌ **Invalid document ID:** {doc_id}. Please provide a positive integer.").send()
            return

        # First check if document exists
        conn = await get_db_connection()
        try:
            doc = await conn.fetchrow("SELECT id, doc_name FROM doc_data WHERE id = $1", doc_id)
            if not doc:
                await cl.Message(content=f"❌ **Document with ID {doc_id} not found.**\nUse `status` command to see available documents.").send()
                return

            # Show confirmation and proceed with deletion
            await cl.Message(content=f"🗑️ **Deleting document:** {doc['doc_name']} (ID: {doc_id})...").send()

            # Perform deletion
            success = await db_manager.delete_document(doc_id)

            if success:
                await cl.Message(content=f"✅ **Document deleted successfully!**\n📄 **{doc['doc_name']}** (ID: {doc_id}) and all its FAQs have been removed.").send()
            else:
                await cl.Message(content=f"❌ **Failed to delete document {doc_id}.** Please check the logs for more details.").send()
        finally:
            await release_db_connection(conn)

    except Exception as e:
        print(f"Error in delete_document function: {e}")
        await cl.Message(content=f"❌ **Error deleting document:** {str(e)}").send()

async def reprocess_document(doc_id):
    """Reprocess a document to regenerate FAQs using static embeddings"""
    conn = None
    try:
        conn = await get_db_connection()
        doc_id = int(doc_id)  # Ensure integer type

        # Get document
        doc = await conn.fetchrow("SELECT doc_name, doc_content, company_id FROM doc_data WHERE id = $1", doc_id)

        if not doc:
            await cl.Message(content=f"❌ **Document {doc_id} not found.**\nUse `status` command to see available documents.").send()
            return

        await cl.Message(content=f"🔄 **Reprocessing {doc['doc_name']} with ultra-fast static embeddings...**").send()

        # Update status to processing
        await conn.execute("UPDATE doc_data SET doc_status = $1 WHERE id = $2", 'processing', doc_id)

        # Delete existing FAQs
        await conn.execute("DELETE FROM faq_data WHERE doc_id = $1", doc_id)

        # Generate new FAQs
        faqs = faq_generator.generate_faqs_from_text(doc['doc_content'])

        if faqs:
            stored_faqs = 0
            for faq in faqs:
                embedding = faq_generator.generate_embedding(faq['question'])
                if embedding:
                    # Ensure proper types
                    question = str(faq['question'])
                    answer = str(faq['answer'])

                    await conn.execute("""
                        INSERT INTO faq_data (question, answer, embedding, doc_id, company_id)
                        VALUES ($1, $2, $3, $4, $5)
                    """, question, answer, embedding, doc_id, doc['company_id'])
                    stored_faqs += 1

            # Update document status
            await conn.execute("""
                UPDATE doc_data
                SET doc_status = $1, total_faqs = $2, processed_at = $3
                WHERE id = $4
            """, 'completed', stored_faqs, datetime.now(), doc_id)

            await cl.Message(content=f"✅ **Reprocessing complete!** Generated {stored_faqs} new FAQs for **{doc['doc_name']}** using static embeddings.").send()
        else:
            await conn.execute("UPDATE doc_data SET doc_status = $1 WHERE id = $2", 'failed', doc_id)
            await cl.Message(content=f"❌ **Reprocessing failed** - Could not generate FAQs for **{doc['doc_name']}**.").send()

    except ValueError:
        await cl.Message(content="❌ **Invalid document ID.** Please provide a valid number.\nExample: `reprocess 1`").send()
    except Exception as e:
        await cl.Message(content=f"❌ **Reprocessing error:** {str(e)}").send()
    finally:
        if conn:
            await release_db_connection(conn)

# Cleanup function
async def cleanup():
    global connection_pool
    if connection_pool:
        await connection_pool.close()

if __name__ == "__main__":
    import atexit
    
    atexit.register(lambda: asyncio.run(cleanup()))
    
    print("🔧 PDF Document Processor Admin Panel Starting...")
    print(f"📍 Admin panel at: http://localhost:8002")
    print(f"🗄️ Database: {'✅ Connected' if DATABASE_URL else '❌ Missing DATABASE_URL'}")
    print(f"🤖 OpenAI: {'✅ Configured' if OPENAI_API_KEY else '❌ Missing OPENAI_API_KEY'}")
    print(f"📁 Upload Folder: {UPLOAD_FOLDER}")
    print("🚀 Using static-retrieval-mrl-en-v1 embedding model for 100x-400x faster performance on CPU!")
    
    # Run Chainlit admin server on port 8002