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
 
# Load .env
load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
 
# Disable Chainlit's data layer to avoid conflicts
cl.data_layer = None
 
# Initialize OpenAI client properly for v1.84.0
client = OpenAI(api_key=OPENAI_API_KEY)
 
# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
 
async def get_db_connection():
    """Get direct database connection"""
    try:
        return await asyncpg.connect(DATABASE_URL)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        raise
 
class DatabaseManager:
    def __init__(self):
        pass
   
    async def init_database(self):
        """Create tables if they don't exist"""
        conn = await get_db_connection()
        try:
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
           
            # Create doc_data table with additional fields for admin tracking
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed_at TIMESTAMP
                )
            """)
           
            # Create indexes for better performance
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_data_status ON doc_data(doc_status)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_faq_data_doc_id ON faq_data(doc_id)")
           
            print("Database tables initialized")
           
        except Exception as e:
            print(f"Error initializing database: {e}")
            raise
        finally:
            await conn.close()
 
    async def get_all_documents(self):
        """Get all documents with their processing status"""
        conn = await get_db_connection()
        try:
            docs = await conn.fetch("""
                SELECT id, doc_name, doc_status, 
                       COALESCE(file_size, 0) as file_size, 
                       COALESCE(total_faqs, 0) as total_faqs,
                       COALESCE(processing_time, 0) as processing_time,
                       created_at, processed_at, error_message, uploaded_by
                FROM doc_data
                ORDER BY created_at DESC
            """)
            return [dict(doc) for doc in docs]
        except Exception as e:
            print(f"Error getting documents: {e}")
            return []
        finally:
            await conn.close()
 
    async def delete_document(self, doc_id):
        """Delete document and its FAQs"""
        conn = await get_db_connection()
        try:
            doc_id = int(doc_id)  # Ensure integer type
           
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
           
            # Delete from document_status table if exists
            await conn.execute("DELETE FROM document_status WHERE doc_id = $1", doc_id)
           
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
            await conn.close()
 
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
 
class FAQGenerator:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
   
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
                model="gpt-4",
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
       
        # Get existing documents
        documents = await db_manager.get_all_documents()
       
        # Create document status display
        if documents:
            doc_list = "📚 **Current Documents:**\n\n"
            for doc in documents[:10]:  # Show last 10 documents
                status_emoji = {
                    'completed': '✅',
                    'processing': '🔄',
                    'failed': '❌',
                    'pending': '⏳'
                }.get(doc.get('doc_status', 'unknown'), '❓')
               
                doc_list += f"{status_emoji} **ID: {doc.get('id', 'N/A')} - {doc.get('doc_name', 'Unknown')}**\n"
                doc_list += f"   📊 Status: {doc.get('doc_status', 'unknown').title()}\n"
                doc_list += f"   🤖 FAQs: {doc.get('total_faqs', 0)}\n"
                
                if doc.get('created_at'):
                    doc_list += f"   📅 Uploaded: {doc['created_at'].strftime('%Y-%m-%d %H:%M')}\n\n"
                else:
                    doc_list += "\n"
        else:
            doc_list = "📚 **No documents uploaded yet.**\n\n"
       
        # Send welcome message
        welcome_msg = f"""🔧 **Document Insertion Admin Panel**
 
Welcome to the PDF document processing interface!
 
{doc_list}
 
**Available Commands:**
• 📤 **Upload PDF**: Use the attachment button to upload PDF files
• 📋 **View Status**: Type 'status' to see all documents
• 🗑️ **Delete Document**: Type 'delete [doc_id]' to remove a document
• 🔄 **Reprocess**: Type 'reprocess [doc_id]' to regenerate FAQs
• 📊 **Stats**: Type 'stats' for processing statistics
• ❓ **Help**: Type 'help' for command list
 
**Upload multiple PDFs at once** - Each will be processed automatically and FAQs will be generated in the background.

**Example commands:**
- `delete 1` - Delete document with ID 1
- `reprocess 2` - Reprocess document with ID 2
"""
       
        await cl.Message(content=welcome_msg).send()
       
    except Exception as e:
        print(f"Error in admin start: {e}")
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
            await cl.Message(content=f"🔄 **Processing {len(upload_tasks)} PDF file(s)...**").send()
           
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
 
📤 **To upload PDFs**: Use the attachment button (📎) to select and upload PDF files.

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
       
        # Save to permanent location
        permanent_filename = str(uuid.uuid4()) + '_' + filename
        permanent_path = str(os.path.join(UPLOAD_FOLDER, permanent_filename))
        shutil.copy2(file_path, permanent_path)
       
        # Get database connection
        conn = await get_db_connection()
       
        # Insert document into database
        doc_id = await conn.fetchval("""
            INSERT INTO doc_data (doc_name, doc_path, doc_content, doc_status, file_size, uploaded_by)
            VALUES ($1, $2, $3, $4, $5, $6) RETURNING id
        """, filename, permanent_path, None, 'processing', file_size, 'admin')
       
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
       
        # Generate FAQs from the extracted text
        faqs = faq_generator.generate_faqs_from_text(text)
       
        if not faqs:
            await conn.execute("""
                UPDATE doc_data
                SET doc_status = $1, error_message = $2, processed_at = $3
                WHERE id = $4
            """, 'failed', 'Could not generate FAQs from document content', datetime.now(), doc_id)
            return False
       
        # Store FAQs with embeddings
        stored_faqs = 0
        for faq in faqs:
            embedding = faq_generator.generate_embedding(faq['question'])
            if embedding:
                # Ensure proper types
                question = str(faq['question'])
                answer = str(faq['answer'])
               
                await conn.execute("""
                    INSERT INTO faq_data (question, answer, embedding, doc_id)
                    VALUES ($1, $2, $3, $4)
                """, question, answer, embedding, doc_id)
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
            content=f"✅ **{filename}** processed successfully!\n🤖 Generated {stored_faqs} FAQs in {processing_time}s"
        ).send()
       
        return True
       
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        if conn and 'doc_id' in locals():
            await conn.execute("""
                UPDATE doc_data
                SET doc_status = $1, error_message = $2, processed_at = $3
                WHERE id = $4
            """, 'failed', str(e), datetime.now(), doc_id)
       
        await cl.Message(content=f"❌ **{filename}** processing failed: {str(e)}").send()
        return False
    finally:
        if conn:
            await conn.close()
 
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
        status_msg += f"   📊 Status: {doc.get('doc_status', 'unknown').title()}\n"
        
        # Safe handling of file_size
        file_size = doc.get('file_size', 0)
        if file_size and file_size > 0:
            status_msg += f"   📏 Size: {file_size / 1024:.1f} KB\n"
        
        status_msg += f"   🤖 FAQs: {doc.get('total_faqs', 0)}\n"
        
        # Safe handling of processing_time
        processing_time = doc.get('processing_time', 0)
        if processing_time and processing_time > 0:
            status_msg += f"   ⏱️ Process Time: {processing_time}s\n"
        
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
    conn = await get_db_connection()
    try:
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
⏱️ **Average Processing Time:** {avg_processing_time:.1f}s
💾 **Total File Size:** {total_file_size / (1024*1024):.1f} MB
 
📈 **Success Rate:** {success_rate:.1f}%
"""
       
        await cl.Message(content=stats_msg).send()
       
    except Exception as e:
        await cl.Message(content=f"❌ Error getting stats: {str(e)}").send()
    finally:
        await conn.close()
 
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
        finally:
            await conn.close()
        
        # Show confirmation and proceed with deletion
        await cl.Message(content=f"🗑️ **Deleting document:** {doc['doc_name']} (ID: {doc_id})...").send()
        
        # Perform deletion
        success = await db_manager.delete_document(doc_id)
        
        if success:
            await cl.Message(content=f"✅ **Document deleted successfully!**\n📄 **{doc['doc_name']}** (ID: {doc_id}) and all its FAQs have been removed.").send()
        else:
            await cl.Message(content=f"❌ **Failed to delete document {doc_id}.** Please check the logs for more details.").send()
            
    except Exception as e:
        print(f"Error in delete_document function: {e}")
        await cl.Message(content=f"❌ **Error deleting document:** {str(e)}").send()
 
async def reprocess_document(doc_id):
    """Reprocess a document to regenerate FAQs"""
    conn = await get_db_connection()
    try:
        doc_id = int(doc_id)  # Ensure integer type
       
        # Get document
        doc = await conn.fetchrow("SELECT doc_name, doc_content FROM doc_data WHERE id = $1", doc_id)
       
        if not doc:
            await cl.Message(content=f"❌ **Document {doc_id} not found.**\nUse `status` command to see available documents.").send()
            return
       
        await cl.Message(content=f"🔄 **Reprocessing {doc['doc_name']}...**").send()
       
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
                        INSERT INTO faq_data (question, answer, embedding, doc_id)
                        VALUES ($1, $2, $3, $4)
                    """, question, answer, embedding, doc_id)
                    stored_faqs += 1
           
            # Update document status
            await conn.execute("""
                UPDATE doc_data
                SET doc_status = $1, total_faqs = $2, processed_at = $3
                WHERE id = $4
            """, 'completed', stored_faqs, datetime.now(), doc_id)
           
            await cl.Message(content=f"✅ **Reprocessing complete!** Generated {stored_faqs} new FAQs for **{doc['doc_name']}**.").send()
        else:
            await conn.execute("UPDATE doc_data SET doc_status = $1 WHERE id = $2", 'failed', doc_id)
            await cl.Message(content=f"❌ **Reprocessing failed** - Could not generate FAQs for **{doc['doc_name']}**.").send()
       
    except ValueError:
        await cl.Message(content="❌ **Invalid document ID.** Please provide a valid number.\nExample: `reprocess 1`").send()
    except Exception as e:
        await cl.Message(content=f"❌ **Reprocessing error:** {str(e)}").send()
    finally:
        await conn.close()
 
if __name__ == "__main__":
    print("PDF Document Processor Admin Panel Starting...")
    print(f"Database URL: {'✅ Configured' if DATABASE_URL else '❌ Missing'}")
    print(f"OpenAI API Key: {'✅ Configured' if OPENAI_API_KEY else '❌ Missing'}")
    print(f"Upload Folder: {UPLOAD_FOLDER}")