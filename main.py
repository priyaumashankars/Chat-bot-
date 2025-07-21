from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime
import os
import asyncpg
import asyncio
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
import requests
import hashlib
import json
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Enhanced CORS configuration for Chainlit
CORS(app, 
     origins=["*"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
     supports_credentials=True)

# Configuration
app.config['SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-this')
DATABASE_URL = os.getenv('DATABASE_URL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
CHAINLIT_SERVER_URL = os.getenv('CHAINLIT_SERVER_URL', 'http://localhost:8000')

# Initialize OpenAI client (only for text processing, not embeddings)
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Initialize the static embedding model (runs 100x-400x faster on CPU)
print("🚀 Loading static embedding model...")
try:
    embedding_model = SentenceTransformer('sentence-transformers/static-retrieval-mrl-en-v1')
    print("✅ Static embedding model loaded successfully! (100x-400x faster on CPU)")
except Exception as e:
    print(f"❌ Error loading embedding model: {e}")
    embedding_model = None

# Database helper functions
async def get_db_connection():
    """Get database connection with error handling"""
    try:
        if not DATABASE_URL:
            raise ValueError("DATABASE_URL not configured")
        return await asyncpg.connect(DATABASE_URL)
    except Exception as e:
        print(f"Database connection error: {e}")
        raise

async def init_auth_tables():
    """Initialize authentication tables if they don't exist"""
    conn = None
    try:
        conn = await get_db_connection()
        
        # Create companies table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS companies (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                code VARCHAR(255) UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create users table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                company_id INTEGER REFERENCES companies(id),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            )
        """)
        
        # Create chainlit_users table for Chainlit integration
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS chainlit_users (
                id SERIAL PRIMARY KEY,
                identifier VARCHAR(255) UNIQUE NOT NULL,
                email VARCHAR(255),
                company_id INTEGER REFERENCES companies(id),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create user_sessions table for token management
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                token_hash VARCHAR(255) NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create faq_data table if it doesn't exist
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS faq_data (
                id SERIAL PRIMARY KEY,
                company_id INTEGER REFERENCES companies(id),
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                embedding FLOAT4[] NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create doc_data table if it doesn't exist
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS doc_data (
                id SERIAL PRIMARY KEY,
                company_id INTEGER REFERENCES companies(id),
                doc_name VARCHAR(255),
                doc_path VARCHAR(500),
                doc_content TEXT,
                doc_status VARCHAR(50) DEFAULT 'pending',
                faq_count INTEGER DEFAULT 0,
                file_size BIGINT DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert a default company if none exists
        company_exists = await conn.fetchval("SELECT COUNT(*) FROM companies")
        if company_exists == 0:
            await conn.execute("""
                INSERT INTO companies (name, code) VALUES 
                ('Default Company', 'default'),
                ('Test Company', 'test')
            """)
            print("✅ Default companies created")
        
        print("✅ Authentication tables initialized")
    except Exception as e:
        print(f"Error initializing tables: {e}")
        raise
    finally:
        if conn:
            await conn.close()

def generate_token(user_id, company_id):
    """Generate JWT token with extended expiration"""
    payload = {
        'user_id': user_id,
        'company_id': company_id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24),
        'iat': datetime.datetime.utcnow(),
        'embedding_model': 'static-retrieval-mrl-en-v1'
    }
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')

def verify_token(token):
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        print("Token expired")
        return None
    except jwt.InvalidTokenError as e:
        print(f"Invalid token: {e}")
        return None

# Helper function to extract token from request
def get_token_from_request():
    """Extract token from Authorization header or query parameter"""
    # Try Authorization header first
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        return auth_header[7:]
    
    # Try query parameter (for Chainlit iframe)
    token = request.args.get('token')
    if token:
        return token
    
    return None

# Utility function to run async code in sync context
def run_async(coro):
    """Helper to run async functions in Flask routes"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, create a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)

async def get_company_by_code(company_code):
    """Get company by code from database"""
    conn = None
    try:
        conn = await get_db_connection()
        company = await conn.fetchrow("""
            SELECT id, name, code FROM companies 
            WHERE code = $1
        """, company_code)
        return dict(company) if company else None
    except Exception as e:
        print(f"Error getting company: {e}")
        return None
    finally:
        if conn:
            await conn.close()

async def create_user(email, password_hash, company_id):
    """Create new user in database"""
    conn = None
    try:
        conn = await get_db_connection()
        user_id = await conn.fetchval("""
            INSERT INTO users (email, password_hash, company_id)
            VALUES ($1, $2, $3) RETURNING id
        """, email, password_hash, company_id)
        return user_id
    except Exception as e:
        print(f"Error creating user: {e}")
        raise
    finally:
        if conn:
            await conn.close()

async def get_user_by_email(email):
    """Get user by email from database"""
    conn = None
    try:
        conn = await get_db_connection()
        user = await conn.fetchrow("""
            SELECT id, email, password_hash, company_id, is_active
            FROM users WHERE email = $1
        """, email)
        return dict(user) if user else None
    except Exception as e:
        print(f"Error getting user: {e}")
        return None
    finally:
        if conn:
            await conn.close()

async def get_user_with_company(user_id):
    """Get user with company information"""
    conn = None
    try:
        conn = await get_db_connection()
        user = await conn.fetchrow("""
            SELECT u.id, u.email, u.company_id, c.name as company_name, c.code as company_code
            FROM users u
            JOIN companies c ON u.company_id = c.id
            WHERE u.id = $1 AND u.is_active = true
        """, user_id)
        return dict(user) if user else None
    except Exception as e:
        print(f"Error getting user with company: {e}")
        return None
    finally:
        if conn:
            await conn.close()

async def create_user_session(user_id, token):
    """Create a new session record in user_sessions table"""
    conn = None
    try:
        conn = await get_db_connection()
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        expires_at = datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        await conn.execute("""
            INSERT INTO user_sessions (user_id, token_hash, expires_at)
            VALUES ($1, $2, $3)
        """, user_id, token_hash, expires_at)
    except Exception as e:
        print(f"Error creating user session: {e}")
    finally:
        if conn:
            await conn.close()

async def create_chainlit_user(email, company_id, user_id):
    """Create or update Chainlit user for seamless integration"""
    conn = None
    try:
        conn = await get_db_connection()
        
        # Get company info
        company = await conn.fetchrow("SELECT name, code FROM companies WHERE id = $1", company_id)
        
        metadata = {
            'email': email,
            'company_id': company_id,
            'company_name': company['name'] if company else 'Unknown',
            'company_code': company['code'] if company else 'Unknown',
            'user_id': user_id,
            'embedding_model': 'static-retrieval-mrl-en-v1'
        }
        
        await conn.execute("""
            INSERT INTO chainlit_users (identifier, email, company_id, metadata, last_login)
            VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
            ON CONFLICT (identifier) DO UPDATE SET
                email = $2,
                company_id = $3,
                metadata = $4,
                last_login = CURRENT_TIMESTAMP
        """, email, email, company_id, json.dumps(metadata))
        
    except Exception as e:
        print(f"Error creating chainlit user: {e}")
    finally:
        if conn:
            await conn.close()

def generate_static_embedding(text):
    """Generate embedding using sentence transformer model"""
    try:
        if not embedding_model:
            print("Embedding model not loaded")
            return None
        if not text or not text.strip():
            return None
        embedding = embedding_model.encode(text.strip(), normalize_embeddings=True)
        return embedding.tolist()
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

async def search_faqs_by_embedding(query, company_id):
    """Search FAQs using embeddings and cosine similarity"""
    try:
        query_embedding = generate_static_embedding(query)
        
        if not query_embedding:
            return None
        
        conn = None
        try:
            conn = await get_db_connection()
            # Get all FAQs for the company
            faqs = await conn.fetch("""
                SELECT question, answer, embedding
                FROM faq_data
                WHERE company_id = $1 AND embedding IS NOT NULL
            """, company_id)
            
            if not faqs:
                return None
            
            # Calculate similarities
            similarities = []
            for faq in faqs:
                if faq['embedding']:
                    try:
                        faq_embedding = list(faq['embedding'])
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
            if best_match[0] >= 0.6:  # Lower threshold for better matches
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
            if conn:
                await conn.close()
            
    except Exception as e:
        print(f"FAQ search error: {e}")
        return None

async def search_faqs_by_keywords(query, company_id):
    """Fallback keyword-based search if embeddings fail"""
    conn = None
    try:
        conn = await get_db_connection()
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
                'confidence': 0.7,
                'source': 'faq',
                'embedding_type': 'keyword_fallback'
            }
        return None
        
    except Exception as e:
        print(f"Keyword search error: {e}")
        return None
    finally:
        if conn:
            await conn.close()

# ====== AUTHENTICATION ENDPOINTS ======

@app.route('/auth/header', methods=['POST'])
def auth_header():
    """Chainlit authentication endpoint"""
    try:
        # Get token from Authorization header
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Authorization header missing or invalid'}), 401
        
        token = auth_header[7:]  # Remove 'Bearer ' prefix
        
        # Verify token
        payload = verify_token(token)
        if not payload:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        # Get user with company information
        user_info = run_async(get_user_with_company(payload['user_id']))
        if not user_info:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({
            'authenticated': True,
            'user': user_info,
            'embedding_model': 'static-retrieval-mrl-en-v1',
            'performance': '100x-400x faster on CPU',
            'token': token
        })
        
    except Exception as e:
        print(f"Auth header error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# Authentication endpoints (updated to use run_async)
@app.route('/api/signup', methods=['POST'])
def signup():
    async def async_signup():
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
                
            email = data.get('email')
            password = data.get('password')
            company_code = data.get('company_code', 'default')  # Default fallback
            
            if not email or not password:
                return jsonify({'error': 'Email and password are required'}), 400
            
            # Check if company exists
            company = await get_company_by_code(company_code)
            if not company:
                return jsonify({'error': 'Invalid company code'}), 400
            
            # Check if user already exists
            existing_user = await get_user_by_email(email)
            if existing_user:
                return jsonify({'error': 'User already exists'}), 400
            
            # Create user
            password_hash = generate_password_hash(password)
            user_id = await create_user(email, password_hash, company['id'])
            
            # Create Chainlit user for seamless integration
            await create_chainlit_user(email, company['id'], user_id)
            
            # Generate token
            token = generate_token(user_id, company['id'])
            await create_user_session(user_id, token)
            
            return jsonify({
                'message': 'User created successfully',
                'token': token,
                'user': {
                    'id': user_id,
                    'email': email,
                    'company_id': company['id']
                },
                'embedding_model': 'static-retrieval-mrl-en-v1',
                'performance': '100x-400x faster on CPU',
                'chainlit_url': f"{CHAINLIT_SERVER_URL}?token={token}"
            })
            
        except Exception as e:
            print(f"Signup error: {e}")
            return jsonify({'error': str(e)}), 500
    
    return run_async(async_signup())

@app.route('/api/login', methods=['POST'])
def login():
    async def async_login():
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
                
            email = data.get('email')
            password = data.get('password')
            
            if not email or not password:
                return jsonify({'error': 'Email and password are required'}), 400
            
            # Get user
            user = await get_user_by_email(email)
            if not user or not user['is_active']:
                return jsonify({'error': 'Invalid credentials'}), 401
            
            # Check password
            if not check_password_hash(user['password_hash'], password):
                return jsonify({'error': 'Invalid credentials'}), 401
            
            # Create/update Chainlit user for seamless integration
            await create_chainlit_user(email, user['company_id'], user['id'])
            
            # Generate token
            token = generate_token(user['id'], user['company_id'])
            await create_user_session(user['id'], token)
            
            return jsonify({
                'message': 'Login successful',
                'token': token,
                'user': {
                    'id': user['id'],
                    'email': user['email'],
                    'company_id': user['company_id']
                },
                'embedding_model': 'static-retrieval-mrl-en-v1',
                'chainlit_url': f"{CHAINLIT_SERVER_URL}?token={token}"
            })
            
        except Exception as e:
            print(f"Login error: {e}")
            return jsonify({'error': str(e)}), 500
    
    return run_async(async_login())

@app.route('/api/verify-token', methods=['GET'])
def verify_token_route():
    try:
        token = get_token_from_request()
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
            
        payload = verify_token(token)
        if not payload:
            return jsonify({'error': 'Token is invalid or expired'}), 401
        
        return jsonify({
            'valid': True,
            'user': {
                'id': payload['user_id'],
                'company_id': payload['company_id']
            },
            'embedding_model': 'static-retrieval-mrl-en-v1'
        })
        
    except Exception as e:
        print(f"Token verification error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# Enhanced endpoint for Chainlit authentication
@app.route('/api/chainlit-auth', methods=['GET', 'POST'])
def chainlit_auth():
    async def async_chainlit_auth():
        try:
            # Get token from query parameter or Authorization header
            token = get_token_from_request()
            
            if not token:
                return jsonify({'error': 'Access token is required'}), 400
            
            # Verify token
            payload = verify_token(token)
            if not payload:
                return jsonify({'error': 'Invalid or expired token'}), 401
            
            # Get user with company information
            user_info = await get_user_with_company(payload['user_id'])
            if not user_info:
                return jsonify({'error': 'User not found'}), 404
            
            return jsonify({
                'authenticated': True,
                'user': user_info,
                'embedding_model': 'static-retrieval-mrl-en-v1',
                'performance': '100x-400x faster on CPU',
                'token': token  # Return token for Chainlit to use
            })
            
        except Exception as e:
            print(f"Chainlit auth error: {e}")
            return jsonify({'error': str(e)}), 500
    
    return run_async(async_chainlit_auth())

@app.route('/api/search', methods=['POST'])
def search():
    async def async_search():
        try:
            token = get_token_from_request()
            if not token:
                return jsonify({'error': 'Token is missing'}), 401
                
            payload = verify_token(token)
            if not payload:
                return jsonify({'error': 'Token is invalid or expired'}), 401
            
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
                
            query = data.get('query', '').strip()
            if not query:
                return jsonify({'error': 'Query is required'}), 400
            
            # Try embedding-based search first
            result = await search_faqs_by_embedding(query, payload['company_id'])
            
            # Fallback to keyword search if embedding search fails
            if not result:
                result = await search_faqs_by_keywords(query, payload['company_id'])
            
            if result:
                return jsonify(result)
            else:
                return jsonify({
                    'answer': "I couldn't find any relevant information for your question. Please try asking in a different way or contact your administrator.",
                    'source': 'system',
                    'confidence': 0.0,
                    'embedding_type': 'static-retrieval-mrl-en-v1'
                })
                
        except Exception as e:
            print(f"Search error: {e}")
            return jsonify({'error': str(e)}), 500
    
    return run_async(async_search())

# Get company statistics
@app.route('/api/company-stats', methods=['GET'])
def company_stats():
    async def async_stats():
        try:
            token = get_token_from_request()
            if not token:
                return jsonify({'error': 'Token is missing'}), 401
                
            payload = verify_token(token)
            if not payload:
                return jsonify({'error': 'Token is invalid or expired'}), 401
            
            conn = None
            try:
                conn = await get_db_connection()
                # Get company stats
                stats = await conn.fetchrow("""
                    SELECT 
                        c.name as company_name,
                        COUNT(DISTINCT dd.id) as total_documents,
                        COUNT(fd.id) as total_faqs,
                        COUNT(DISTINCT dd.id) FILTER (WHERE dd.doc_status = 'completed') as completed_documents
                    FROM companies c
                    LEFT JOIN doc_data dd ON c.id = dd.company_id
                    LEFT JOIN faq_data fd ON c.id = fd.company_id
                    WHERE c.id = $1
                    GROUP BY c.id, c.name
                """, payload['company_id'])
                
                if stats:
                    return jsonify({
                        'company_name': stats['company_name'],
                        'total_documents': stats['total_documents'] or 0,
                        'total_faqs': stats['total_faqs'] or 0,
                        'completed_documents': stats['completed_documents'] or 0,
                        'embedding_model': 'static-retrieval-mrl-en-v1'
                    })
                else:
                    return jsonify({
                        'company_name': 'Unknown',
                        'total_documents': 0,
                        'total_faqs': 0,
                        'completed_documents': 0,
                        'embedding_model': 'static-retrieval-mrl-en-v1',
                        'performance': '100x-400x faster on CPU'
                    })
                    
            finally:
                if conn:
                    await conn.close()
                
        except Exception as e:
            print(f"Stats error: {e}")
            return jsonify({'error': str(e)}), 500
    
    return run_async(async_stats())

# New endpoint to get embedding model info
@app.route('/api/embedding-info', methods=['GET'])
def embedding_info():
    return jsonify({
        'model': 'static-retrieval-mrl-en-v1',
        'type': 'sentence-transformers',
        'performance': '100x-400x faster on CPU',
        'features': [
            'Ultra-fast CPU inference',
            'No API calls required',
            'Cost-effective',
            'Edge computing friendly',
            'High quality embeddings'
        ],
        'status': 'loaded' if embedding_model else 'error'
    })

# Add a test route for debugging
@app.route('/api/test-auth', methods=['GET'])
def test_auth():
    async def async_test_auth():
        try:
            token = get_token_from_request()
            if not token:
                return jsonify({'error': 'No token provided', 'headers': dict(request.headers), 'args': dict(request.args)}), 401
            
            payload = verify_token(token)
            if not payload:
                return jsonify({'error': 'Invalid token', 'token': token[:20] + '...'}), 401
            
            return jsonify({'success': True, 'payload': payload})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return run_async(async_test_auth())

# Serve the HTML file
@app.route('/')
def serve_html():
    try:
        return send_from_directory('.', 'index.html')
    except Exception as e:
        print(f"Error serving HTML: {e}")
        return jsonify({'error': 'File not found'}), 404

@app.route('/<path:filename>')
def serve_static(filename):
    try:
        return send_from_directory('.', filename)
    except Exception as e:
        print(f"Error serving static file {filename}: {e}")
        return jsonify({'error': 'File not found'}), 404

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Initialize auth tables
    try:
        asyncio.run(init_auth_tables())
        print("✅ Database tables initialized")
    except Exception as e:
        print(f"❌ Database initialization error: {e}")
        print("Please check your DATABASE_URL in .env file")
    
    print("🚀 Flask server starting...")
    print(f"📍 Access the app at: http://localhost:5000")
    print(f"🗄️ Database: {'✅ Connected' if DATABASE_URL else '❌ Missing DATABASE_URL'}")
    print(f"🤖 OpenAI: {'✅ Configured' if OPENAI_API_KEY else '❌ Missing OPENAI_API_KEY'}")
    print(f"⚡ Embedding Model: static-retrieval-mrl-en-v1 (100x-400x faster on CPU!)")
    print(f"🔗 Chainlit: {CHAINLIT_SERVER_URL}")
    print("🔐 Authentication system ready!")
    
    app.run(debug=True, port=5000, host='0.0.0.0')