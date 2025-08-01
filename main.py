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
from contextlib import asynccontextmanager
import traceback
from functools import wraps

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Enhanced CORS configuration
CORS(app, 
     origins=["*"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
     supports_credentials=True)

# Configuration
app.config['SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-this')
DATABASE_URL = os.getenv('DATABASE_URL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
CHAINLIT_SERVER_URL = os.getenv('CHAINLIT_SERVER_URL', 'http://localhost:8001')

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Initialize the static embedding model
print("🚀 Loading static embedding model...")
try:
    embedding_model = SentenceTransformer('sentence-transformers/static-retrieval-mrl-en-v1')
    print("✅ Static embedding model loaded successfully! (100x-400x faster on CPU)")
except Exception as e:
    print(f"❌ Error loading embedding model: {e}")
    embedding_model = None

# Global connection pool
connection_pool = None
pool_lock = asyncio.Lock()

# Global event loop
app_loop = None

async def init_connection_pool():
    """Initialize database connection pool with proper error handling"""
    global connection_pool
    
    async with pool_lock:  # Prevent multiple simultaneous initializations
        if connection_pool is None:
            try:
                connection_pool = await asyncpg.create_pool(
                    DATABASE_URL,
                    min_size=2,
                    max_size=20,
                    max_queries=50000,
                    max_inactive_connection_lifetime=300.0,
                    command_timeout=60,
                    statement_cache_size=0,  # Disable to prevent "another operation in progress" errors
                    server_settings={'jit': 'off'}
                )
                print("✅ Database connection pool initialized")
            except Exception as e:
                print(f"❌ Failed to create database connection pool: {e}")
                raise

async def close_connection_pool():
    """Properly close the connection pool"""
    global connection_pool
    
    async with pool_lock:
        if connection_pool:
            try:
                await connection_pool.close()
                connection_pool = None
                print("✅ Database connection pool closed")
            except Exception as e:
                print(f"Error closing pool: {e}")

async def get_db_connection():
    """Get database connection with proper error handling and retry logic"""
    global connection_pool
    
    # Initialize pool if needed
    if connection_pool is None:
        await init_connection_pool()
    
    # Check if pool is closing or closed
    if connection_pool._closing or connection_pool._closed:
        connection_pool = None
        await init_connection_pool()
    
    max_retries = 3
    retry_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            # Acquire connection with timeout
            conn = await asyncio.wait_for(
                connection_pool.acquire(),
                timeout=15.0
            )
            
            # Test the connection is actually working
            await conn.fetchval('SELECT 1')
            return conn
            
        except asyncio.TimeoutError:
            print(f"Database connection timeout (attempt {attempt + 1}/{max_retries})")
            if attempt == max_retries - 1:
                raise Exception("Database connection timeout after all retries")
            await asyncio.sleep(retry_delay)
            
        except asyncpg.exceptions.ConnectionDoesNotExistError as e:
            print(f"Connection does not exist (attempt {attempt + 1}/{max_retries}): {e}")
            # Try to release the bad connection
            if 'conn' in locals():
                try:
                    await connection_pool.release(conn, timeout=1.0)
                except:
                    pass
            
            if attempt == max_retries - 1:
                # Reinitialize pool on final attempt
                connection_pool = None
                await init_connection_pool()
                raise Exception("Database connection failed after all retries")
            await asyncio.sleep(retry_delay)
            
        except Exception as e:
            print(f"Database connection error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(retry_delay)

async def release_db_connection(conn):
    """Release database connection with proper error handling"""
    global connection_pool
    
    if not connection_pool or not conn:
        return
    
    try:
        # Check if connection is still valid
        if conn.is_closed():
            print("Attempting to release already closed connection")
            return
            
        # Try to release the connection
        await asyncio.wait_for(
            connection_pool.release(conn),
            timeout=5.0
        )
    except asyncio.TimeoutError:
        print("Timeout releasing connection - terminating")
        try:
            await conn.close()
        except:
            pass
    except asyncpg.exceptions.InterfaceError as e:
        if "another operation is in progress" in str(e):
            print("Connection busy - terminating")
            try:
                await conn.close()
            except:
                pass
        else:
            print(f"Interface error releasing connection: {e}")
    except Exception as e:
        print(f"Error releasing connection: {e}")
        try:
            await conn.close()
        except:
            pass

@asynccontextmanager
async def get_db_connection_context():
    """Context manager for database connections - RECOMMENDED APPROACH"""
    conn = None
    try:
        conn = await get_db_connection()
        yield conn
    finally:
        if conn:
            await release_db_connection(conn)

async def init_auth_tables():
    """Initialize authentication tables if they don't exist"""
    async with get_db_connection_context() as conn:
        try:
            # Create companies table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS companies (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    code VARCHAR(255) UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create users table with identifier column for Chainlit compatibility
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    identifier VARCHAR(255) UNIQUE,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    company_id INTEGER REFERENCES companies(id),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    metadata JSONB DEFAULT '{}'::jsonb
                )
            """)
            
            # Create faq_data table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS faq_data (
                    id SERIAL PRIMARY KEY,
                    company_id INTEGER REFERENCES companies(id),
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    embedding FLOAT8[] NULL,
                    doc_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create doc_data table
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
            
            # Create chat_sessions table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) UNIQUE NOT NULL,
                    user_id INTEGER REFERENCES users(id),
                    company_id INTEGER REFERENCES companies(id),
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
                    message_type VARCHAR(50) DEFAULT 'user_message',
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_faq_data_company_id ON faq_data(company_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_data_company_id ON doc_data(company_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_users_identifier ON users(identifier)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_sessions_company_id ON chat_sessions(company_id)")
            
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
            
            print("✅ Authentication tables initialized")
        except Exception as e:
            print(f"Error initializing tables: {e}")
            raise

def generate_token(user_id, company_id, email=None):
    """Generate JWT token with proper payload structure for Chainlit compatibility"""
    payload = {
        'sub': str(user_id),  # Ensure it's a string
        'user_id': user_id,
        'email': email,
        'company_id': company_id,
        'role': 'user',
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24),
        'iat': datetime.datetime.utcnow(),
        'type': 'access',
        'embedding_model': 'static-retrieval-mrl-en-v1'
    }
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')

def verify_token(token):
    """Verify JWT token with better error handling"""
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        print(f"✅ Token verified successfully for user: {payload.get('email', payload.get('sub'))}")
        return payload
    except jwt.ExpiredSignatureError:
        print("❌ Token expired")
        return None
    except jwt.InvalidTokenError as e:
        print(f"❌ Invalid token: {e}")
        return None
    except Exception as e:
        print(f"❌ Token verification error: {e}")
        return None

def get_token_from_request():
    """Extract token from Authorization header or query parameter"""
    # Try Authorization header first
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        return auth_header[7:]
    
    # Try query parameter
    token = request.args.get('token')
    if token:
        return token
    
    return None

# Flask async wrapper that handles event loop properly
def async_flask(f):
    """Decorator for async routes in Flask"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        global app_loop
        # Use the global event loop
        if app_loop is None or app_loop.is_closed():
            app_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(app_loop)
        
        try:
            return app_loop.run_until_complete(f(*args, **kwargs))
        except Exception as e:
            print(f"Error in async route: {e}")
            traceback.print_exc()
            raise
    
    return wrapper

async def get_company_by_code(company_code):
    """Get company by code from database"""
    async with get_db_connection_context() as conn:
        try:
            company = await conn.fetchrow("""
                SELECT id, name, code FROM companies 
                WHERE code = $1
            """, company_code)
            return dict(company) if company else None
        except Exception as e:
            print(f"Error getting company: {e}")
            return None

async def create_user(email, password_hash, company_id):
    """Create new user in database with identifier for Chainlit compatibility"""
    async with get_db_connection_context() as conn:
        try:
            user_id = await conn.fetchval("""
                INSERT INTO users (identifier, email, password_hash, company_id, metadata)
                VALUES ($1, $2, $3, $4, $5) RETURNING id
            """, email, email, password_hash, company_id, json.dumps({}))
            print(f"✅ User created with ID: {user_id}")
            return user_id
        except Exception as e:
            print(f"Error creating user: {e}")
            raise

async def get_user_by_email(email):
    """Get user by email from database"""
    async with get_db_connection_context() as conn:
        try:
            user = await conn.fetchrow("""
                SELECT id, identifier, email, password_hash, company_id, is_active
                FROM users WHERE email = $1
            """, email)
            return dict(user) if user else None
        except Exception as e:
            print(f"Error getting user: {e}")
            return None

async def get_user_with_company(user_id):
    """Get user with company information"""
    async with get_db_connection_context() as conn:
        try:
            user = await conn.fetchrow("""
                SELECT u.id, u.identifier, u.email, u.company_id, 
                       c.name as company_name, c.code as company_code
                FROM users u
                JOIN companies c ON u.company_id = c.id
                WHERE u.id = $1 AND u.is_active = true
            """, int(user_id))
            return dict(user) if user else None
        except Exception as e:
            print(f"Error getting user with company: {e}")
            return None

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
    async with get_db_connection_context() as conn:
        try:
            query_embedding = generate_static_embedding(query)
            
            if not query_embedding:
                return None
            
            faqs = await conn.fetch("""
                SELECT question, answer, embedding
                FROM faq_data
                WHERE company_id = $1 AND embedding IS NOT NULL
            """, company_id)
            
            if not faqs:
                return None
            
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
            
            similarities.sort(key=lambda x: x[0], reverse=True)
            best_match = similarities[0]
            
            if best_match[0] >= 0.6:
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
                    
        except Exception as e:
            print(f"FAQ search error: {e}")
            return None

# ====== AUTHENTICATION ENDPOINTS ======

@app.route('/api/signup', methods=['POST', 'OPTIONS'])
@async_flask
async def signup():
    """Signup endpoint with proper connection handling"""
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        email = data.get('email')
        password = data.get('password')
        company_code = data.get('company_code', 'default')
        
        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400
        
        print(f"🔍 Signup attempt for: {email} (Company: {company_code})")
        
        # Check if company exists
        company = await get_company_by_code(company_code)
        if not company:
            print(f"❌ Invalid company code: {company_code}")
            return jsonify({'error': 'Invalid company code'}), 400
        
        # Check if user already exists
        existing_user = await get_user_by_email(email)
        if existing_user:
            print(f"❌ User already exists: {email}")
            return jsonify({'error': 'User already exists'}), 400
        
        # Create user
        password_hash = generate_password_hash(password)
        user_id = await create_user(email, password_hash, company['id'])
        
        print(f"✅ User created: {email} (ID: {user_id}, Company: {company['id']})")
        
        # Generate token with email included
        token = generate_token(user_id, company['id'], email)
        
        return jsonify({
            'message': 'User created successfully',
            'token': token,
            'user': {
                'id': user_id,
                'email': email,
                'company_id': company['id'],
                'company_name': company['name'],
                'company_code': company['code']
            },
            'embedding_model': 'static-retrieval-mrl-en-v1',
            'chainlit_url': f"{CHAINLIT_SERVER_URL}?token={token}"
        })
        
    except Exception as e:
        print(f"Signup error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/login', methods=['POST', 'OPTIONS'])
@async_flask
async def login():
    """Login endpoint with proper connection handling"""
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400
        
        print(f"🔍 Login attempt for: {email}")
        
        # Get user
        user = await get_user_by_email(email)
        if not user or not user['is_active']:
            print(f"❌ User not found or inactive: {email}")
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Check password
        if not check_password_hash(user['password_hash'], password):
            print(f"❌ Invalid password for: {email}")
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Update last login
        async with get_db_connection_context() as conn:
            await conn.execute(
                "UPDATE users SET last_login = $1 WHERE id = $2",
                datetime.datetime.utcnow(), user['id']
            )
            
            # Get company info
            company = await conn.fetchrow("SELECT name, code FROM companies WHERE id = $1", user['company_id'])
            company = dict(company) if company else {'name': 'Unknown', 'code': 'unknown'}
        
        print(f"✅ Login successful for: {email} (ID: {user['id']}, Company: {user['company_id']})")
        
        # Generate token with email included
        token = generate_token(user['id'], user['company_id'], email)
        
        return jsonify({
            'message': 'Login successful',
            'token': token,
            'user': {
                'id': user['id'],
                'email': user['email'],
                'company_id': user['company_id'],
                'company_name': company['name'],
                'company_code': company['code']
            },
            'embedding_model': 'static-retrieval-mrl-en-v1',
            'chainlit_url': f"{CHAINLIT_SERVER_URL}?token={token}"
        })
        
    except Exception as e:
        print(f"Login error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/verify-token', methods=['GET', 'OPTIONS'])
@async_flask
async def verify_token_route():
    """Verify token endpoint with proper connection handling"""
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        token = get_token_from_request()
        print(f"🔍 Token verification request. Token present: {bool(token)}")
        
        if not token:
            print("❌ No token provided in request")
            return jsonify({'error': 'Token is missing'}), 401
            
        payload = verify_token(token)
        if not payload:
            print("❌ Token verification failed")
            return jsonify({'error': 'Token is invalid or expired'}), 401
        
        user_id = payload.get('user_id') or payload.get('sub')
        print(f"🔍 Getting user info for ID: {user_id}")
        
        # Get fresh user info
        user_info = await get_user_with_company(int(user_id))
        if not user_info:
            print(f"❌ User not found for ID: {user_id}")
            return jsonify({'error': 'User not found'}), 404
        
        print(f"✅ Token verification successful for: {user_info['email']}")
        
        return jsonify({
            'valid': True,
            'user': user_info,
            'embedding_model': 'static-retrieval-mrl-en-v1'
        })
        
    except Exception as e:
        print(f"❌ Token verification error: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/chainlit-auth', methods=['GET', 'POST', 'OPTIONS'])
@async_flask
async def chainlit_auth():
    """Chainlit auth endpoint with proper connection handling"""
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        token = get_token_from_request()
        
        if not token:
            return jsonify({'error': 'Access token is required'}), 400
        
        payload = verify_token(token)
        if not payload:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        user_id = payload.get('user_id') or payload.get('sub')
        user_email = payload.get('email')
        
        print(f"🔍 Chainlit auth for user_id: {user_id}, email: {user_email}")
        
        user_info = await get_user_with_company(int(user_id))
        if not user_info:
            return jsonify({'error': 'User not found'}), 404
        
        actual_email = user_info['email']
        
        print(f"✅ Chainlit auth successful: {actual_email} (Company: {user_info['company_id']})")
        
        return jsonify({
            'authenticated': True,
            'user': {
                'id': user_info['id'],
                'email': actual_email,
                'company_id': user_info['company_id'],
                'company_name': user_info['company_name'],
                'company_code': user_info['company_code']
            },
            'embedding_model': 'static-retrieval-mrl-en-v1',
            'token': token
        })
        
    except Exception as e:
        print(f"Chainlit auth error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['POST', 'OPTIONS'])
@async_flask
async def search():
    """Search endpoint with proper connection handling"""
    if request.method == 'OPTIONS':
        return '', 200
        
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
        
        result = await search_faqs_by_embedding(query, payload['company_id'])
        
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

@app.route('/api/company-stats', methods=['GET', 'OPTIONS'])
@async_flask
async def company_stats():
    """Company stats endpoint with proper connection handling"""
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        token = get_token_from_request()
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
            
        payload = verify_token(token)
        if not payload:
            return jsonify({'error': 'Token is invalid or expired'}), 401
        
        async with get_db_connection_context() as conn:
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
                    'embedding_model': 'static-retrieval-mrl-en-v1'
                })
                
    except Exception as e:
        print(f"Stats error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/embedding-info', methods=['GET'])
def embedding_info():
    """Return information about the embedding model"""
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

# Serve static files
@app.route('/')
def serve_html():
    """Serve the main HTML file"""
    try:
        return send_from_directory('.', 'index.html')
    except Exception as e:
        print(f"Error serving HTML: {e}")
        return jsonify({'error': 'File not found'}), 404

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    try:
        return send_from_directory('.', filename)
    except Exception as e:
        print(f"Error serving static file {filename}: {e}")
        return jsonify({'error': 'File not found'}), 404

# Error handlers
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    print(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# Handle CORS preflight requests
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

# Cleanup function for application shutdown
async def cleanup():
    """Clean up resources on shutdown"""
    await close_connection_pool()
    
    # Clean up any pending tasks
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == '__main__':
    # Initialize global event loop
    app_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(app_loop)
    
    # Initialize auth tables
    try:
        app_loop.run_until_complete(init_auth_tables())
        print("✅ Database tables initialized")
    except Exception as e:
        print(f"❌ Database initialization error: {e}")
        print("Please check your DATABASE_URL in .env file")
        print("Make sure PostgreSQL is running and the database exists")
        exit(1)
    
    print("🚀 Flask server starting...")
    print(f"📍 Access the app at: http://localhost:5000")
    print(f"🗄️ Database: {'✅ Connected' if DATABASE_URL else '❌ Missing DATABASE_URL'}")
    print(f"🤖 OpenAI: {'✅ Configured' if OPENAI_API_KEY else '❌ Missing OPENAI_API_KEY'}")
    print(f"⚡ Embedding Model: static-retrieval-mrl-en-v1 (100x-400x faster on CPU!)")
    print(f"🔗 Chainlit: {CHAINLIT_SERVER_URL}")
    print("🔐 Authentication system ready!")
    print("\n" + "="*60)
    print("SETUP INSTRUCTIONS:")
    print("1. Start this Flask server: python main.py")
    print("2. Start Chainlit server: python your_chainlit_app.py")
    print("3. Open your HTML file in a web server")
    print("4. Use company codes: 'default', 'demo', or 'test'")
    print("="*60 + "\n")
    
    try:
        app.run(debug=True, port=5000, host='0.0.0.0', threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down gracefully...")
    finally:
        # Ensure cleanup on shutdown
        try:
            app_loop.run_until_complete(cleanup())
        except:
            pass
        finally:
            if not app_loop.is_closed():
                app_loop.close()