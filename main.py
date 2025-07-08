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

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
app.config['SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-this')
DATABASE_URL = os.getenv('DATABASE_URL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
CHAINLIT_SERVER_URL = os.getenv('CHAINLIT_SERVER_URL', 'http://localhost:8001')

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Database helper functions
async def get_db_connection():
    """Get database connection"""
    try:
        return await asyncpg.connect(DATABASE_URL)
    except Exception as e:
        print(f"Database connection error: {e}")
        raise

async def init_auth_tables():
    """Initialize authentication tables if they don't exist"""
    conn = await get_db_connection()
    try:
        # Create users table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                company_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
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
        
        print("Authentication tables initialized")
    finally:
        await conn.close()

def generate_token(user_id, company_id):
    payload = {
        'user_id': user_id,
        'company_id': company_id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    }
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')

def verify_token(token):
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

async def get_company_by_code(company_code):
    """Get company by code from database"""
    conn = await get_db_connection()
    try:
        company = await conn.fetchrow("""
            SELECT id, name, code FROM companies 
            WHERE code = $1
        """, company_code)
        return dict(company) if company else None
    finally:
        await conn.close()

async def create_user(email, password_hash, company_id):
    """Create new user in database"""
    conn = await get_db_connection()
    try:
        user_id = await conn.fetchval("""
            INSERT INTO users (email, password_hash, company_id)
            VALUES ($1, $2, $3) RETURNING id
        """, email, password_hash, company_id)
        return user_id
    finally:
        await conn.close()

async def get_user_by_email(email):
    """Get user by email from database"""
    conn = await get_db_connection()
    try:
        user = await conn.fetchrow("""
            SELECT id, email, password_hash, company_id, is_active
            FROM users WHERE email = $1
        """, email)
        return dict(user) if user else None
    finally:
        await conn.close()

async def get_user_with_company(user_id):
    """Get user with company information"""
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

async def create_user_session(user_id, token):
    """Create a new session record in user_sessions table"""
    conn = await get_db_connection()
    try:
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        expires_at = datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        await conn.execute("""
            INSERT INTO user_sessions (user_id, token_hash, expires_at)
            VALUES ($1, $2, $3)
        """, user_id, token_hash, expires_at)
    finally:
        await conn.close()

async def search_faqs_by_embedding(query, company_id):
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

async def search_faqs_by_keywords(query, company_id):
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

@app.route('/api/signup', methods=['POST'])
def signup():
    async def async_signup():
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
                
            email = data.get('email')
            password = data.get('password')
            company_code = data.get('company_code')
            
            if not email or not password or not company_code:
                return jsonify({'error': 'Email, password, and company code are required'}), 400
            
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
                }
            })
            
        except Exception as e:
            print(f"Signup error: {e}")
            return jsonify({'error': 'Internal server error'}), 500
    
    return asyncio.run(async_signup())

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
                }
            })
            
        except Exception as e:
            print(f"Login error: {e}")
            return jsonify({'error': 'Internal server error'}), 500
    
    return asyncio.run(async_login())

@app.route('/api/verify-token', methods=['GET'])
def verify_token_route():
    try:
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        
        if token.startswith('Bearer '):
            token = token[7:]
            
        payload = verify_token(token)
        if not payload:
            return jsonify({'error': 'Token is invalid or expired'}), 401
        
        return jsonify({
            'valid': True,
            'user': {
                'id': payload['user_id'],
                'company_id': payload['company_id']
            }
        })
        
    except Exception as e:
        print(f"Token verification error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/search', methods=['POST'])
def search():
    async def async_search():
        try:
            token = request.headers.get('Authorization')
            if not token:
                return jsonify({'error': 'Token is missing'}), 401
            
            if token.startswith('Bearer '):
                token = token[7:]
                
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
                    'confidence': 0.0
                })
                
        except Exception as e:
            print(f"Search error: {e}")
            return jsonify({'error': 'Internal server error'}), 500
    
    return asyncio.run(async_search())

# New endpoint for Chainlit Copilot authentication
@app.route('/api/copilot-auth', methods=['POST'])
def copilot_auth():
    async def async_copilot_auth():
        try:
            data = request.get_json()
            token = data.get('access_token')
            
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
                'user': user_info
            })
            
        except Exception as e:
            print(f"Copilot auth error: {e}")
            return jsonify({'error': 'Internal server error'}), 500
    
    return asyncio.run(async_copilot_auth())

# Get company statistics
@app.route('/api/company-stats', methods=['GET'])
def company_stats():
    async def async_stats():
        try:
            token = request.headers.get('Authorization')
            if not token:
                return jsonify({'error': 'Token is missing'}), 401
            
            if token.startswith('Bearer '):
                token = token[7:]
                
            payload = verify_token(token)
            if not payload:
                return jsonify({'error': 'Token is invalid or expired'}), 401
            
            conn = await get_db_connection()
            try:
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
                        'completed_documents': stats['completed_documents'] or 0
                    })
                else:
                    return jsonify({
                        'company_name': 'Unknown',
                        'total_documents': 0,
                        'total_faqs': 0,
                        'completed_documents': 0
                    })
                    
            finally:
                await conn.close()
                
        except Exception as e:
            print(f"Stats error: {e}")
            return jsonify({'error': 'Internal server error'}), 500
    
    return asyncio.run(async_stats())

# Serve the HTML file
@app.route('/')
def serve_html():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

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
    print(f"🔗 Chainlit: {CHAINLIT_SERVER_URL}")
    print("📝 Make sure you have companies and FAQs in your database!")
    print("🤖 Start Chainlit copilot with: chainlit run copilot_faq.py")
    
    app.run(debug=True, port=5000, host='0.0.0.0')