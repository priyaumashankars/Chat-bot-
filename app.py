import os
import psycopg2
from psycopg2.extras import RealDictCursor
import openai
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import PyPDF2
import uuid
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from dotenv import load_dotenv


# Initialize Flask app
app = Flask(__name__)
CORS(app)

load_dotenv()  # Load environment variables from .env file

DATABASE_URL = os.getenv('DATABASE_URL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class DatabaseManager:
    def __init__(self, db_url):
        self.db_url = db_url
        self.init_database()
    
    def get_connection(self):
        return psycopg2.connect(self.db_url)
    
    def init_database(self):
        """Create tables if they don't exist"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Create faq_data table
        cursor.execute("""
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
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_status (
                id SERIAL PRIMARY KEY,
                doc_id INTEGER NOT NULL,
                doc_status VARCHAR(50) NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create doc_data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS doc_data (
                id SERIAL PRIMARY KEY,
                doc_name VARCHAR(255) NOT NULL,
                doc_path VARCHAR(500) NOT NULL,
                doc_content TEXT,
                doc_status VARCHAR(50) DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create chat_sessions table for conversation history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(255) UNIQUE NOT NULL,
                doc_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create chat_messages table for conversation history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(255) NOT NULL,
                role VARCHAR(50) NOT NULL,
                message TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        cursor.close()
        conn.close()


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
            if faq['embedding']:
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
    
    def get_document_content(self, doc_id):
        """Get document content for context"""
        try:
            conn = db_manager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT doc_content FROM doc_data WHERE id = %s", (doc_id,))
            result = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            return result[0] if result else None
        except Exception as e:
            print(f"Error getting document content: {e}")
            return None
    
    def generate_response(self, question, conversation_history, doc_id=None):
        """Generate a conversational response using document context and chat history"""
        try:
            # Get relevant FAQs for context
            conn = db_manager.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            if doc_id:
                cursor.execute("""
                    SELECT question, answer, embedding
                    FROM faq_data
                    WHERE embedding IS NOT NULL AND doc_id = %s
                """, (doc_id,))
            else:
                cursor.execute("""
                    SELECT question, answer, embedding
                    FROM faq_data
                    WHERE embedding IS NOT NULL
                """)
            
            faqs = cursor.fetchall()
            cursor.close()
            conn.close()
            
            # Find relevant context
            relevant_context = self.find_relevant_context(question, faqs)
            
            # Get document content if available
            doc_content = ""
            if doc_id:
                doc_content = self.get_document_content(doc_id)
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
                temperature=0.7,  # Slightly more creative for conversation
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error while processing your question. Please try again."
    
    def generate_faqs_from_text(self, text):
        """Generate FAQs from document text (keeping original functionality)"""
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
db_manager = DatabaseManager(DATABASE_URL)
chatbot = ChatbotEngine(OPENAI_API_KEY)

@app.route('/')
def index():
    return render_template('chatbot.html')

@app.route('/upload', methods=['POST'])
def upload_document():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are supported'}), 400
    
    try:
        # Save file
        filename = str(uuid.uuid4()) + '_' + file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Extract text from PDF
        text = DocumentProcessor.extract_text_from_pdf(file_path)
        
        if not text or len(text.strip()) < 200:
            return jsonify({'error': 'Could not extract sufficient text from PDF'}), 400
        
        # Insert document into database
        conn = db_manager.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO doc_data (doc_name, doc_path, doc_content, doc_status)
            VALUES (%s, %s, %s, %s) RETURNING id
        """, (file.filename, file_path, text, 'processing'))
        
        doc_id = cursor.fetchone()[0]
        
        # Insert initial status into document_status table
        cursor.execute("""
            INSERT INTO document_status (doc_id, doc_status)
            VALUES (%s, %s)
        """, (doc_id, 'processing'))
        
        conn.commit()
        
        # Generate FAQs from the extracted text
        faqs = chatbot.generate_faqs_from_text(text)
        
        if not faqs:
            # Update both tables for failed status
            cursor.execute("UPDATE doc_data SET doc_status = %s WHERE id = %s", ('failed', doc_id))
            cursor.execute("""
                INSERT INTO document_status (doc_id, doc_status)
                VALUES (%s, %s)
            """, (doc_id, 'failed'))
            conn.commit()
            cursor.close()
            conn.close()
            return jsonify({'error': 'Could not generate FAQs from document'}), 400
        
        # Store FAQs with embeddings
        stored_faqs = 0
        for faq in faqs:
            embedding = chatbot.generate_embedding(faq['question'])
            if embedding:
                cursor.execute("""
                    INSERT INTO faq_data (question, answer, embedding, doc_id)
                    VALUES (%s, %s, %s, %s)
                """, (faq['question'], faq['answer'], embedding, doc_id))
                stored_faqs += 1
        
        # Update both tables for completed status
        cursor.execute("UPDATE doc_data SET doc_status = %s WHERE id = %s", ('completed', doc_id))
        cursor.execute("""
            INSERT INTO document_status (doc_id, doc_status)
            VALUES (%s, %s)
        """, (doc_id, 'completed'))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({
            'message': 'PDF document processed successfully',
            'doc_id': doc_id,
            'faqs_generated': stored_faqs,
            'document_name': file.filename
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing document: {str(e)}'}), 500

@app.route('/chat/start', methods=['POST'])
def start_chat_session():
    """Start a new chat session"""
    try:
        data = request.json or {}
        doc_id = data.get('doc_id')
        
        session_id = str(uuid.uuid4())
        
        conn = db_manager.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO chat_sessions (session_id, doc_id)
            VALUES (%s, %s)
        """, (session_id, doc_id))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        # Get document name if doc_id provided
        doc_name = None
        if doc_id:
            conn = db_manager.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT doc_name FROM doc_data WHERE id = %s", (doc_id,))
            result = cursor.fetchone()
            doc_name = result[0] if result else None
            cursor.close()
            conn.close()
        
        welcome_message = f"Hello! I'm ready to answer questions about your document"
        if doc_name:
            welcome_message += f" '{doc_name}'"
        welcome_message += ". What would you like to know?"
        
        # Store welcome message
        conn = db_manager.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO chat_messages (session_id, role, message)
            VALUES (%s, %s, %s)
        """, (session_id, 'assistant', welcome_message))
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({
            'session_id': session_id,
            'doc_id': doc_id,
            'doc_name': doc_name,
            'welcome_message': welcome_message
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat/message', methods=['POST'])
def chat_message():
    """Handle chat messages"""
    try:
        data = request.json
        session_id = data.get('session_id')
        message = data.get('message', '').strip()
        
        if not session_id or not message:
            return jsonify({'error': 'Session ID and message are required'}), 400
        
        # Get session info
        conn = db_manager.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT doc_id FROM chat_sessions WHERE session_id = %s", (session_id,))
        session_result = cursor.fetchone()
        
        if not session_result:
            return jsonify({'error': 'Invalid session ID'}), 400
        
        doc_id = session_result[0]
        
        # Store user message
        cursor.execute("""
            INSERT INTO chat_messages (session_id, role, message)
            VALUES (%s, %s, %s)
        """, (session_id, 'user', message))
        
        # Get conversation history
        cursor.execute("""
            SELECT role, message FROM chat_messages
            WHERE session_id = %s
            ORDER BY timestamp ASC
        """, (session_id,))
        
        history = cursor.fetchall()
        conversation_history = [{'role': h[0], 'message': h[1]} for h in history]
        
        # Update last activity
        cursor.execute("""
            UPDATE chat_sessions SET last_activity = CURRENT_TIMESTAMP
            WHERE session_id = %s
        """, (session_id,))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        # Generate response
        response = chatbot.generate_response(message, conversation_history, doc_id)
        
        # Store assistant response
        conn = db_manager.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO chat_messages (session_id, role, message)
            VALUES (%s, %s, %s)
        """, (session_id, 'assistant', response))
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({
            'response': response,
            'session_id': session_id
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat/history/<session_id>')
def get_chat_history(session_id):
    """Get chat history for a session"""
    try:
        conn = db_manager.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT role, message, timestamp
            FROM chat_messages
            WHERE session_id = %s
            ORDER BY timestamp ASC
        """, (session_id,))
        
        messages = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return jsonify([dict(msg) for msg in messages])
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/documents', methods=['GET'])
def get_documents():
    try:
        conn = db_manager.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT d.id, d.doc_name, d.doc_status, d.created_at,
                   COUNT(f.id) as faq_count
            FROM doc_data d
            LEFT JOIN faq_data f ON d.id = f.doc_id
            WHERE d.doc_name LIKE '%.pdf'
            GROUP BY d.id, d.doc_name, d.doc_status, d.created_at
            ORDER BY d.created_at DESC
        """)
        
        documents = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return jsonify([dict(doc) for doc in documents])
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear_all', methods=['DELETE'])
def clear_all_data():
    """Clear all data - useful for testing"""
    try:
        conn = db_manager.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM chat_messages")
        cursor.execute("DELETE FROM chat_sessions")
        cursor.execute("DELETE FROM faq_data")
        cursor.execute("DELETE FROM document_status")
        cursor.execute("DELETE FROM doc_data")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'message': 'All data cleared successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)