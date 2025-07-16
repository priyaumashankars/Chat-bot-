import os
import asyncpg
from openai import OpenAI
import uuid
from datetime import datetime, timedelta
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from dotenv import load_dotenv
import chainlit as cl
from chainlit.input_widget import Select
import asyncio
from chainlit.data.chainlit_data_layer import ChainlitDataLayer
from typing import Any, Dict, List, Optional
import jwt
from fastapi import FastAPI, Response, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
import bcrypt
from sentence_transformers import SentenceTransformer

# Initialize the static embedding model (runs 100x-400x faster on CPU)
print("🚀 Loading static embedding model for Policy Assistant...")
embedding_model = SentenceTransformer('sentence-transformers/static-retrieval-mrl-en-v1')
print("✅ Static embedding model loaded successfully! (100x-400x faster on CPU)")

# FastAPI app for authentication endpoints
app = FastAPI(title="Policy Assistant API", version="1.0.0")
security = HTTPBearer()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8080", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
JWT_SECRET = os.getenv('JWT_SECRET')

# Validate environment variables
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is required")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is required")
if not JWT_SECRET:
    raise ValueError("JWT_SECRET is required for secure authentication")

# Override default Chainlit data layer with DATABASE_URL
cl.data_layer = CustomChainlitDataLayer(database_url=DATABASE_URL)

# Initialize OpenAI client (only for text processing, not embeddings)
client = OpenAI(api_key=OPENAI_API_KEY)

# Global connection pool
connection_pool = None

# JWT settings
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Pydantic models for authentication
class UserSignup(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: Optional[str] = "user"

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    user: dict

class LoginRequest(BaseModel):
    identifier: str
    password: str
    metadata: Optional[dict] = None

# Static embedding helper function
def generate_static_embedding(text: str) -> Optional[list]:
    """Generate embedding using static-retrieval-mrl-en-v1 model (100x-400x faster on CPU)"""
    try:
        # Use the static embedding model instead of OpenAI
        embedding = embedding_model.encode(text, normalize_embeddings=True)
        return embedding.tolist()  # Convert to list for compatibility
    except Exception as e:
        print(f"Error generating static embedding: {e}")
        return None

# Authentication helper functions
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str, token_type: str = "access"):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        if payload.get("type") != token_type:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

def create_jwt(identifier: str, role: str = "user") -> str:
    payload = {
        "sub": identifier,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=24),  # Token expires in 24 hours
        "iat": datetime.utcnow(),
        "embedding_model": "static-retrieval-mrl-en-v1"
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=ALGORITHM)

def validate_jwt_token(token: str) -> Optional[cl.User]:
    try:
        decoded = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        user = cl.User(
            identifier=decoded.get('sub'),
            metadata={
                "role": decoded.get('role', 'user'),
                "authenticated_at": datetime.utcnow().isoformat(),
                "token_validated_locally": True,
                "embedding_model": "static-retrieval-mrl-en-v1",
                "performance": "100x-400x faster on CPU"
            }
        )
        return user
    except jwt.ExpiredSignatureError:
        print("JWT token has expired")
        return None
    except jwt.InvalidTokenError as e:
        print(f"Invalid JWT token: {e}")
        return None

@cl.header_auth_callback
def header_auth_callback(headers: Dict) -> Optional[cl.User]:
    auth_header = headers.get("authorization") or headers.get("Authorization")
    if not auth_header:
        print("No authorization header found")
        return None
    
    try:
        scheme, token = auth_header.split(" ", 1)
        if scheme.lower() != "bearer":
            print("Invalid authorization scheme")
            return None
    except ValueError:
        print("Invalid authorization header format")
        return None
    
    return validate_jwt_token(token)

# FastAPI authentication endpoints
@app.post("/auth/signup", response_model=dict)
async def signup(user_data: UserSignup):
    conn = None
    try:
        conn = await get_db_connection()
        # Check if user exists
        existing_user = conn.execute(
            "SELECT id FROM auth_users WHERE email = $1", 
            user_data.email
        ).fetchone()
        
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create user
        password_hash = hash_password(user_data.password)
        result = await conn.fetchrow("""
            INSERT INTO auth_users (name, email, password_hash, role)
            VALUES ($1, $2, $3, $4)
            RETURNING id
        """, user_data.name, user_data.email, password_hash, user_data.role)
        
        return {
            "message": "User created successfully", 
            "user_id": result['id'],
            "embedding_model": "static-retrieval-mrl-en-v1",
            "performance": "100x-400x faster on CPU"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            await connection_pool.release(conn)

@app.post("/auth/login", response_model=Token)
async def login(user_credentials: UserLogin):
    conn = None
    try:
        conn = await get_db_connection()
        user = await conn.fetchrow(
            "SELECT * FROM auth_users WHERE email = $1 AND is_active = true", 
            user_credentials.email
        )
        
        if not user or not verify_password(user_credentials.password, user["password_hash"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Create tokens
        access_token = create_access_token(data={"sub": user["email"]})
        refresh_token = create_refresh_token(data={"sub": user["email"]})
        
        # Update last login
        await conn.execute(
            "UPDATE auth_users SET last_login = CURRENT_TIMESTAMP WHERE id = $1",
            user["id"]
        )
        
        # Also create/update the chainlit user record
        await db_manager.upsert_user(user["email"], {
            "name": user["name"],
            "role": user["role"],
            "email": user["email"],
            "embedding_model": "static-retrieval-mrl-en-v1"
        })
        
        user_data = {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
            "role": user["role"],
            "embedding_model": "static-retrieval-mrl-en-v1"
        }
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "user": user_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            await connection_pool.release(conn)

@app.post("/auth/refresh")
async def refresh_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    refresh_token = credentials.credentials
    payload = verify_token(refresh_token, "refresh")
    user_email = payload.get("sub")
    
    if not user_email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    # Create new access token
    new_access_token = create_access_token(data={"sub": user_email})
    
    return {
        "access_token": new_access_token, 
        "token_type": "bearer",
        "embedding_model": "static-retrieval-mrl-en-v1"
    }

@app.get("/auth/verify")
async def verify_auth(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = verify_token(token)
    user_email = payload.get("sub")
    
    if not user_email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    conn = None
    try:
        conn = await get_db_connection()
        user = await conn.fetchrow(
            "SELECT * FROM auth_users WHERE email = $1 AND is_active = true", 
            user_email
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        return {
            "message": "Token is valid", 
            "user": dict(user),
            "embedding_model": "static-retrieval-mrl-en-v1",
            "performance": "100x-400x faster on CPU"
        }
    finally:
        if conn:
            await connection_pool.release(conn)

@app.post("/auth/logout")
async def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # In a production environment, you'd want to blacklist the token
    return {
        "message": "Logged out successfully",
        "embedding_model": "static-retrieval-mrl-en-v1"
    }

# Legacy login endpoint for backward compatibility
@app.post("/login")
async def legacy_login(request: LoginRequest, response: Response):
    conn = None
    try:
        conn = await get_db_connection()
        # For backward compatibility, try to find user by identifier or email
        user_record = await conn.fetchrow(
            "SELECT identifier, metadata FROM users WHERE identifier = $1",
            request.identifier
        )
        
        if not user_record:
            # Try to find by email in auth_users table
            auth_user = await conn.fetchrow(
                "SELECT email, name, role FROM auth_users WHERE email = $1",
                request.identifier
            )
            if auth_user and request.password == "password":  # Replace with secure password check
                # Create user record in legacy users table
                metadata = {
                    "name": auth_user["name"],
                    "role": auth_user["role"],
                    "email": auth_user["email"],
                    "embedding_model": "static-retrieval-mrl-en-v1"
                }
                await conn.execute("""
                    INSERT INTO users (identifier, metadata, last_login)
                    VALUES ($1, $2, CURRENT_TIMESTAMP)
                    ON CONFLICT (identifier) DO UPDATE SET
                        metadata = $2,
                        last_login = CURRENT_TIMESTAMP
                """, request.identifier, json.dumps(metadata))
                user_record = {"identifier": request.identifier, "metadata": metadata}
        
        if not user_record or request.password != "password":  # Replace with secure password check
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Generate JWT
        role = "user"
        if user_record['metadata']:
            if isinstance(user_record['metadata'], str):
                metadata = json.loads(user_record['metadata'])
            else:
                metadata = user_record['metadata']
            role = metadata.get('role', 'user')
        
        token = create_jwt(identifier=request.identifier, role=role)
        
        # Set JWT in HTTP-only cookie
        response.set_cookie(
            key="chainlit-jwt",
            value=token,
            httponly=True,
            secure=False,  # Set to True in production (HTTPS)
            samesite="lax",
            max_age=86400  # 24 hours
        )
        return {
            "message": "Login successful", 
            "identifier": request.identifier,
            "embedding_model": "static-retrieval-mrl-en-v1"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            await connection_pool.release(conn)

@app.get("/chainlit/companies")
async def get_companies():
    companies = await db_manager.list_companies()
    return companies

async def get_db_connection():
    global connection_pool
    if connection_pool is None:
        await init_connection_pool()
    try:
        conn = await asyncio.wait_for(connection_pool.acquire(), timeout=15.0)
        return conn
    except asyncio.TimeoutError:
        raise
    except Exception as e:
        raise

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

def safe_strftime(date_obj, format_str='%Y-%m-%d %H:%M', default="Not set"):
    return date_obj.strftime(format_str) if date_obj else default

class DatabaseManager:
    def __init__(self):
        pass

    async def init_database(self):
        conn = None
        try:
            conn = await get_db_connection()
            
            # Create auth_users table for JWT authentication
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS auth_users (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role VARCHAR(50) DEFAULT 'user',
                    is_active BOOLEAN DEFAULT true,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create demo user if it doesn't exist
            demo_password = hash_password("demo123")
            await conn.execute("""
                INSERT INTO auth_users (name, email, password_hash, role)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (email) DO NOTHING
            """, "Demo User", "demo@company.com", demo_password, "admin")
            
            # Legacy users table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    identifier VARCHAR(255) UNIQUE NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS companies (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    code VARCHAR(50) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS faq_data (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    embedding FLOAT8[],
                    doc_id INTEGER,
                    company_id INTEGER,
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
                    company_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) UNIQUE NOT NULL,
                    user_id INTEGER,
                    doc_id INTEGER,
                    company_id INTEGER,
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
            
            # Create indexes
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_faq_data_company_id ON faq_data(company_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_data_company_id ON doc_data(company_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_sessions_company_id ON chat_sessions(company_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_users_identifier ON users(identifier)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_auth_users_email ON auth_users(email)")
            
            # Create trigger for updated_at
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
            
            # Update existing data
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

    async def upsert_user(self, identifier: str, metadata: dict) -> int:
        conn = None
        try:
            conn = await get_db_connection()
            # Add embedding model info to metadata
            metadata["embedding_model"] = "static-retrieval-mrl-en-v1"
            metadata["performance"] = "100x-400x faster on CPU"
            
            result = await conn.fetchrow("""
                INSERT INTO users (identifier, metadata, last_login)
                VALUES ($1, $2, CURRENT_TIMESTAMP)
                ON CONFLICT (identifier) DO UPDATE SET
                    metadata = $2,
                    last_login = CURRENT_TIMESTAMP
                RETURNING id
            """, identifier, json.dumps(metadata))
            return result['id']
        except Exception as e:
            print(f"Error upserting user: {e}")
            return None
        finally:
            if conn:
                await connection_pool.release(conn)

    async def get_company_by_identifier(self, identifier: str):
        conn = None
        try:
            conn = await get_db_connection()
            return await conn.fetchrow("""
                SELECT id, name, code FROM companies
                WHERE LOWER(name) = LOWER($1) OR LOWER(code) = LOWER($1)
            """, identifier.strip())
        except Exception as e:
            return None
        finally:
            if conn:
                await connection_pool.release(conn)

    async def list_companies(self):
        conn = None
        try:
            conn = await get_db_connection()
            companies = await conn.fetch("SELECT name, code FROM companies ORDER BY name")
            return [dict(company) for company in companies]
        except Exception as e:
            return []
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
                user_id = cl.user_session.get("user_id")
                await conn.execute("""
                    INSERT INTO chat_sessions (session_id, user_id, doc_id, company_id, client_type, created_at, last_activity)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (session_id) DO UPDATE SET last_activity = $7
                """, session_id, user_id, None, None, client_type, datetime.now(), datetime.now())
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

    async def create_session(self, session_id: str, user_id: int, doc_id: Optional[int] = None, company_id: Optional[int] = None, client_type: str = 'web') -> bool:
        conn = None
        try:
            conn = await get_db_connection()
            await conn.execute("""
                INSERT INTO chat_sessions (session_id, user_id, doc_id, company_id, client_type, created_at, last_activity)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (session_id) DO UPDATE SET
                    last_activity = $7,
                    user_id = $2,
                    doc_id = COALESCE($3, chat_sessions.doc_id),
                    company_id = COALESCE($4, chat_sessions.company_id),
                    client_type = $5
            """, session_id, user_id, doc_id, company_id, client_type, datetime.now(), datetime.now())
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

    async def get_policy_id_by_name(self, doc_name: str, company_id: Optional[int] = None) -> Optional[int]:
        conn = None
        try:
            conn = await get_db_connection()
            if company_id:
                result = await conn.fetchrow(
                    "SELECT id FROM doc_data WHERE doc_name ILIKE $1 AND company_id = $2",
                    f"%{doc_name}%", company_id
                )
            else:
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

    async def get_policy_details_by_name(self, doc_name: str, company_id: Optional[int] = None) -> Optional[Dict]:
        conn = None
        try:
            conn = await get_db_connection()
            if company_id:
                result = await conn.fetchrow(
                    "SELECT id, doc_name, doc_path FROM doc_data WHERE doc_name ILIKE $1 AND company_id = $2",
                    f"%{doc_name}%", company_id
                )
            else:
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
        self.client = OpenAI(api_key=api_key)

    def generate_embedding(self, text):
        """Generate embedding using static-retrieval-mrl-en-v1 model (100x-400x faster on CPU)"""
        try:
            # Use the static embedding model instead of OpenAI
            return generate_static_embedding(text)
        except Exception as e:
            print(f"Error generating static embedding: {e}")
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

    async def get_available_documents(self, company_id: Optional[int] = None):
        conn = None
        try:
            conn = await get_db_connection()
            if company_id:
                all_docs = await conn.fetch("""
                    SELECT d.id, d.doc_name, d.doc_path, d.faq_count, d.created_at, d.updated_at, d.doc_status, d.file_size, c.name as company_name
                    FROM doc_data d
                    LEFT JOIN companies c ON d.company_id = c.id
                    WHERE d.company_id = $1
                    ORDER BY d.doc_name
                """, company_id)
            else:
                all_docs = await conn.fetch("""
                    SELECT d.id, d.doc_name, d.doc_path, d.faq_count, d.created_at, d.updated_at, d.doc_status, d.file_size, c.name as company_name
                    FROM doc_data d
                    LEFT JOIN companies c ON d.company_id = c.id
                    ORDER BY d.doc_name
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

    async def search_policies(self, query: str, company_id: Optional[int] = None, doc_id: Optional[int] = None, limit: int = 5) -> List[Dict]:
        query_embedding = self.generate_embedding(query)
        if not query_embedding:
            return []
        conn = None
        try:
            conn = await get_db_connection()
            if doc_id:
                faqs = await conn.fetch("""
                    SELECT f.question, f.answer, f.embedding, d.doc_name
                    FROM faq_data f
                    JOIN doc_data d ON f.doc_id = d.id
                    WHERE f.doc_id = $1 AND d.doc_status = 'completed'
                """, doc_id)
            elif company_id:
                faqs = await conn.fetch("""
                    SELECT f.question, f.answer, f.embedding, d.doc_name
                    FROM faq_data f
                    JOIN doc_data d ON f.doc_id = d.id
                    WHERE f.company_id = $1 AND d.doc_status = 'completed'
                """, company_id)
            else:
                faqs = await conn.fetch("""
                    SELECT f.question, f.answer, f.embedding, d.doc_name
                    FROM faq_data f
                    JOIN doc_data d ON f.doc_id = d.id
                    WHERE d.doc_status = 'completed'
                    LIMIT 100
                """)
            if not faqs:
                return []
            similarities = []
            for faq in faqs:
                if faq['embedding']:
                    similarity = cosine_similarity(
                        [query_embedding],
                        [faq['embedding']]
                    )[0][0]
                    similarities.append({
                        'question': faq['question'],
                        'answer': faq['answer'],
                        'doc_name': faq['doc_name'],
                        'similarity': similarity,
                        'embedding_type': 'static-retrieval-mrl-en-v1'
                    })
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:limit]
        except Exception as e:
            print(f"Error searching policies: {e}")
            return []
        finally:
            if conn:
                await connection_pool.release(conn)

    async def generate_response(self, question, conversation_history, doc_id=None, company_id=None, is_copilot=False):
        conn = None
        try:
            conn = await get_db_connection()
            if doc_id:
                faqs = await conn.fetch("""
                    SELECT question, answer, embedding
                    FROM faq_data
                    WHERE doc_id = $1
                """, doc_id)
            elif company_id:
                faqs = await conn.fetch("""
                    SELECT question, answer, embedding
                    FROM faq_data
                    WHERE company_id = $1
                """, company_id)
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
                7. Keep responses under 200 words unless more detail is specifically requested
                8. Powered by ultra-fast static-retrieval-mrl-en-v1 embeddings (100x-400x faster on CPU)"""
            else:
                system_prompt = """You are a helpful AI assistant that answers questions based on uploaded PDF documents.
                Guidelines:
                1. Be conversational and friendly
                2. If you have relevant information from the document, use it to answer
                3. If the question is not covered in the document, politely say so and offer general help
                4. Maintain context from the conversation history
                5. Ask follow-up questions when appropriate
                6. Be concise but informative
                7. Always be helpful and engaging
                8. Powered by ultra-fast static-retrieval-mrl-en-v1 embeddings (100x-400x faster on CPU)"""
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

@cl.action_callback("download_policy")
async def handle_download_action(action):
    policy_name = action.value
    session_id = cl.user_session.get("session_id")
    company_id = cl.user_session.get("company_id")
    if session_id and policy_name and company_id:
        if cl.context.session.client_type == "copilot":
            fn = cl.CopilotFunction(
                name="download_policy",
                args={"policy_name": policy_name, "status": "starting"}
            )
            await fn.acall()
        await download_policy_file(policy_name, session_id, company_id, is_copilot=True)
    else:
        await cl.Message(content="❌ Unable to download policy. Please try again.").send()

@cl.action_callback("select_policy")
async def handle_select_action(action):
    policy_name = action.value
    session_id = cl.user_session.get("session_id")
    company_id = cl.user_session.get("company_id")
    if session_id and policy_name and company_id:
        if cl.context.session.client_type == "copilot":
            fn = cl.CopilotFunction(
                name="select_policy",
                args={"policy_name": policy_name}
            )
            await fn.acall()
        await handle_policy_selection_by_name(policy_name, company_id, is_copilot=True)
    else:
        await cl.Message(content="❌ Unable to select policy. Please try again.").send()

async def handle_system_message(message: cl.Message, session_id: str):
    try:
        content = message.content.strip()
        try:
            data = json.loads(content)
            message_type = data.get('type', 'unknown')
            if message_type == 'demo_interaction':
                response = f"📊 Demo interaction logged: {data.get('user_message', 'N/A')}"
            elif message_type == 'page_focus':
                response = "👁️ Welcome back! I'm here to help with your policy questions."
            elif message_type == 'page_performance':
                load_time = data.get('load_time', 0)
                response = f"⚡ Page loaded in {load_time}ms. Ready to assist with ultra-fast static embeddings!"
            elif message_type == 'network_status':
                status = data.get('status', 'unknown')
                response = f"🌐 Network status: {status}"
            elif message_type == 'javascript_error':
                response = "🐛 I detected a technical issue. Everything should still work normally."
            else:
                response = f"📨 System message received: {message_type}"
        except json.JSONDecodeError:
            response = f"📋 System update: {content}"
        await cl.Message(content=response).send()
        await db_manager.store_message(session_id, 'assistant', response, message_type='system_response')
    except Exception as e:
        print(f"Error handling system message: {e}")

async def handle_list_policies_enhanced(session_id: str, company_id: int, is_copilot: bool = False):
    try:
        available_docs = await chatbot.get_available_documents(company_id=company_id)
        company_name = cl.user_session.get("company_name", "your company")
        if not available_docs:
            response = f"❌ No policies available for {company_name}."
        else:
            if is_copilot:
                fn = cl.CopilotFunction(
                    name="show_notification",
                    args={
                        "message": f"Found {len(available_docs)} policies for {company_name}",
                        "type": "success"
                    }
                )
                await fn.acall()
                response = f"📚 Available Policies for {company_name} ({len(available_docs)}):\n\n"
                actions = []
                for doc in available_docs[:5]:
                    response += f"• {doc['doc_name']} (ID: {doc['id']}, FAQs: {doc.get('faq_count', 0)})\n"
                    actions.extend([
                        cl.Action(
                            name="select_policy",
                            value=doc['doc_name'],
                            label=f"📄 Select {doc['doc_name'][:20]}{'...' if len(doc['doc_name']) > 20 else ''}",
                            description=f"Focus on {doc['doc_name']}"
                        ),
                        cl.Action(
                            name="download_policy",
                            value=doc['doc_name'],
                            label=f"⬇️ Download {doc['doc_name'][:15]}{'...' if len(doc['doc_name']) > 15 else ''}",
                            description=f"Download {doc['doc_name']}"
                        )
                    ])
                if len(available_docs) > 5:
                    response += f"\n... and {len(available_docs) - 5} more policies\n"
                response += "\nQuick Actions (powered by static embeddings):"
                await cl.Message(content=response, actions=actions[:10]).send()
            else:
                response = f"📚 Available Policies for {company_name} ({len(available_docs)}):\n\n"
                for doc in available_docs:
                    response += f"• {doc['doc_name']} (ID: {doc['id']})\n"
                    response += f"  - FAQs: {doc.get('faq_count', 0)}\n"
                    response += f"  - Status: {doc.get('doc_status', 'unknown')}\n"
                    response += f"  - Updated: {safe_strftime(doc.get('updated_at'), default='Never')}\n"
                    response += f"  - Size: {doc.get('file_size', 0)} bytes\n\n"
                response += "Commands:\n"
                response += "• policy name [id] - Get policy name by ID\n"
                response += "• policy id [name] - Get policy ID by name\n"
                response += "• download [policy_name] - Download policy file\n"
                response += "• show [policy_name] - Show policy details\n"
                response += "• admin - Access admin panel\n"
                response += "\n⚡ Powered by static-retrieval-mrl-en-v1 (100x-400x faster!)"
        await cl.Message(content=response).send()
        await db_manager.store_message(session_id, 'assistant', response)
    except Exception as e:
        print(f"Error listing policies: {str(e)}")
        error_response = "❌ Error retrieving policy list. Please try again."
        await cl.Message(content=error_response).send()
        await db_manager.store_message(session_id, 'assistant', error_response)

async def download_policy_with_notification(policy_name: str, session_id: str, company_id: int, is_copilot: bool = False):
    try:
        if is_copilot:
            fn = cl.CopilotFunction(
                name="show_notification",
                args={
                    "message": f"Preparing download for {policy_name}...",
                    "type": "info"
                }
            )
            await fn.acall()
        await download_policy_file(policy_name, session_id, company_id, is_copilot)
        if is_copilot:
            fn = cl.CopilotFunction(
                name="show_notification",
                args={
                    "message": f"Download ready for {policy_name}",
                    "type": "success"
                }
            )
            await fn.acall()
    except Exception as e:
        if is_copilot:
            fn = cl.CopilotFunction(
                name="show_notification",
                args={
                    "message": f"Download failed for {policy_name}",
                    "type": "error"
                }
            )
            await fn.acall()
        raise e

async def handle_admin_command(is_copilot: bool = False):
    if is_copilot:
        await cl.Message(content="🚫 Admin access is not available in copilot mode. Please use the web interface to access administrative functions.").send()
    else:
        await cl.Message(content="🔧 Admin Panel Access: http://localhost:8000\n\nUse the admin panel to upload and manage policy documents.").send()

async def send_user_context_update(user_message: str, session_id: str, company_id: int):
    try:
        fn = cl.CopilotFunction(
            name="get_user_context",
            args={"request_type": "current_state"}
        )
        context_result = await fn.acall()
        print(f"User context: {context_result}")
    except Exception as e:
        print(f"Error getting user context: {e}")

async def enhanced_policy_search(user_message: str, company_id: int, doc_id: int = None, is_copilot: bool = False):
    if is_copilot:
        fn = cl.CopilotFunction(
            name="show_notification",
            args={
                "message": "Searching with ultra-fast static embeddings...",
                "type": "info"
            }
        )
        await fn.acall()
    search_results = await chatbot.search_policies(user_message, company_id=company_id, doc_id=doc_id, limit=3)
    if search_results and is_copilot:
        fn = cl.CopilotFunction(
            name="show_notification",
            args={
                "message": f"Found {len(search_results)} relevant results using static embeddings",
                "type": "success"
            }
        )
        await fn.acall()
    return search_results

@cl.on_chat_start
async def start():
    try:
        await init_connection_pool()
        await db_manager.init_database()
        user = cl.user_session.get("user")
        if not user:
            await cl.Message(content="❌ Authentication required. Please login through the secure web interface to continue.").send()
            return
        
        # Enhanced user validation for JWT
        if not hasattr(user, 'identifier') or not user.identifier:
            await cl.Message(content="❌ Invalid user session. Please login again.").send()
            return
            
        user_metadata = user.metadata or {}
        user_id = await db_manager.upsert_user(user.identifier, user_metadata)
        if not user_id:
            await cl.Message(content="❌ Error setting up user session. Please try again.").send()
            return
            
        session_id = str(uuid.uuid4())
        cl.user_session.set("session_id", session_id)
        cl.user_session.set("user_id", user_id)
        cl.user_session.set("doc_id", None)
        cl.user_session.set("doc_name", None)
        cl.user_session.set("company_id", None)
        cl.user_session.set("company_name", None)
        
        is_copilot = cl.context.session.client_type == "copilot"
        client_type = "copilot" if is_copilot else "web"
        
        companies = await db_manager.list_companies()
        if not companies:
            if is_copilot:
                welcome_message = f"""🤖 Policy Copilot
                👋 Welcome, {user_metadata.get('name', user.identifier)}!
                ⚡ Powered by static-retrieval-mrl-en-v1 (100x-400x faster on CPU!)
                ❌ No companies found in the system. Contact your administrator to add company data."""
            else:
                welcome_message = f"""👋 Welcome to the Policy Assistant, {user_metadata.get('name', user.identifier)}!
                ⚡ Powered by static-retrieval-mrl-en-v1 (100x-400x faster on CPU!)
                ❌ No companies found in the system. Please contact your administrator to add company data."""
            await cl.Message(content=welcome_message).send()
            return
            
        if is_copilot:
            company_list = "\n".join([f"• {comp['name']} (Code: {comp['code']})" for comp in companies])
            company_input = await cl.AskUserMessage(
                content=f"""🏢 Select your company:
                {company_list}
                ⚡ Enhanced with ultra-fast static embeddings!
                Enter company name or code:"""
            ).send()
        else:
            company_list = "\n".join([f"• {comp['name']} (Code: {comp['code']})" for comp in companies])
            company_input = await cl.AskUserMessage(
                content=f"""🏢 Welcome to the Policy Assistant, {user_metadata.get('name', user.identifier)}!
                ⚡ Powered by static-retrieval-mrl-en-v1 (100x-400x faster on CPU!)
                Please select your company to get started:
                Available Companies:
                {company_list}
                Enter your company name or code:"""
            ).send()
            
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
                content="❌ No company identifier provided. Please refresh and try again."
            ).send()
            return
            
        company = await db_manager.get_company_by_identifier(company_identifier.strip())
        if not company:
            await cl.Message(
                content=f"❌ Company '{company_identifier}' not found.\n\nAvailable companies:\n{company_list}\n\nPlease refresh and try again with a valid company name or code."
            ).send()
            return
            
        cl.user_session.set("company_id", company['id'])
        cl.user_session.set("company_name", company['name'])
        
        success = await db_manager.create_session(session_id, user_id, company_id=company['id'], client_type=client_type)
        
        available_docs = await chatbot.get_available_documents(company_id=company['id'])
        
        if is_copilot:
            if available_docs:
                docs_summary = f"{len(available_docs)} policies available for {company['name']}"
                welcome_message = f"""🤖 Policy Copilot Ready
                👋 Welcome, {user_metadata.get('name', user.identifier)}!
                🏢 Company: {company['name']}
                📚 Status: {docs_summary}
                🔐 Authentication: Secure JWT ({user_metadata.get('role', 'user')})
                ⚡ Embedding Model: static-retrieval-mrl-en-v1 (100x-400x faster on CPU!)
                
                Quick Commands:
                • Ask about any policy
                • Type policy names for quick access
                • Request downloads or summaries
                • Use "list policies" to see all available policies
                
                Ready to help with lightning-fast search! 🚀"""
            else:
                welcome_message = f"""🤖 Policy Copilot
                👋 Welcome, {user_metadata.get('name', user.identifier)}!
                🏢 Company: {company['name']}
                🔐 Authentication: Secure JWT ({user_metadata.get('role', 'user')})
                ⚡ Embedding Model: static-retrieval-mrl-en-v1 (100x-400x faster on CPU!)
                ❌ No policies currently available for your company.
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
                    f"• {doc['doc_name']} ({doc.get('faq_count', 0)} FAQs) - Status: {doc.get('doc_status', 'unknown')} - Updated: {safe_strftime(doc.get('updated_at'), default='Never')}"
                    for doc in available_docs
                ])
                welcome_message = f"""👋 Welcome to {company['name']} Policy Assistant, {user_metadata.get('name', user.identifier)}!
                🎯 Current Selection: {selected_policy}
                🔐 Authentication: Secure JWT ({user_metadata.get('role', 'user')})
                ⚡ Embedding Model: static-retrieval-mrl-en-v1 (100x-400x faster on CPU!)
                
                📚 Available Policies:
                {docs_list}
                
                You can:
                • Use the Policy Selector above to switch between policies
                • Type policy name [id] to get the name of a policy by ID
                • Type policy id [name] to get the ID of a policy by name
                • Type download [policy_name] to download a policy document
                • Type list policies to see all available policies
                • Type admin to access the admin panel
                • Ask questions about policies based on your current selection
                
                💬 How can I help you today?
                
                Note: Use the dropdown selector above to change your policy focus, or use text commands for advanced features.
                """
            else:
                welcome_message = f"""👋 Welcome to {company['name']} Policy Assistant, {user_metadata.get('name', user.identifier)}!
                🔐 Authentication: Secure JWT ({user_metadata.get('role', 'user')})
                ⚡ Embedding Model: static-retrieval-mrl-en-v1 (100x-400x faster on CPU!)
                ❌ No policies are currently available for your company.
                Please contact your administrator to upload policies, or check back later.
                💬 Feel free to ask me questions and I'll do my best to help!"""
                
        await cl.Message(content=welcome_message).send()
        await db_manager.store_message(session_id, 'assistant', welcome_message)
    except Exception as e:
        print(f"Error in chat start: {str(e)}")
        error_message = "Sorry, there was an error initializing the chat. Please refresh the page and login again."
        await cl.Message(content=error_message).send()

@cl.on_settings_update
async def setup_agent(settings):
    if cl.context.session.client_type != "copilot":
        selected_policy = settings.get("PolicySelector", "All Policies (Search across all)")
        await handle_policy_widget_selection(selected_policy)

async def handle_policy_widget_selection(selected_policy: str):
    session_id = cl.user_session.get("session_id")
    available_docs = cl.user_session.get("available_docs", [])
    company_name = cl.user_session.get("company_name", "your company")
    if selected_policy == "All Policies (Search across all)":
        cl.user_session.set("doc_id", None)
        cl.user_session.set("doc_name", None)
        response_message = f"🌐 Policy Selection: All Policies for {company_name}\n\n⚡ Powered by static-retrieval-mrl-en-v1 embeddings\nI'll now search across all available policies to answer your questions."
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
                f"📄 Policy Selected: {matching_doc['doc_name']}\n\n"
                f"📊 Available FAQs: {matching_doc.get('faq_count', 0)}\n"
                f"📅 Status: {matching_doc.get('doc_status', 'unknown')}\n"
                f"📅 Last Updated: {safe_strftime(matching_doc.get('updated_at'), default='Never')}\n"
                f"⚡ Search Engine: static-retrieval-mrl-en-v1 (100x-400x faster!)\n\n"
                f"I'll now focus my responses on this specific policy. Use download {matching_doc['doc_name']} to download."
            )
        else:
            response_message = f"❌ Error: Could not find policy '{policy_display_name}'. Please try again."
    await cl.Message(content=response_message).send()
    if session_id:
        await db_manager.store_message(session_id, 'assistant', response_message)

@cl.on_message
async def main(message: cl.Message):
    user = cl.user_session.get("user")
    if not user:
        await cl.Message(content="❌ Authentication required. Please login through the secure web interface to continue.").send()
        return
    
    user_message = message.content.strip()
    session_id = cl.user_session.get("session_id")
    doc_id = cl.user_session.get("doc_id")
    company_id = cl.user_session.get("company_id")
    company_name = cl.user_session.get("company_name")
    is_copilot = cl.context.session.client_type == "copilot"
    
    if not session_id:
        await cl.Message(content="Session error. Please refresh the page.").send()
        return
    if not company_id:
        await cl.Message(content="Company not selected. Please refresh the page.").send()
        return
    
    if is_copilot and message.type == "system_message":
        await handle_system_message(message, session_id)
        return
    
    await db_manager.store_message(session_id, 'user', user_message, message_type=message.type)
    
    if user_message.lower().startswith("list policies"):
        await handle_list_policies_enhanced(session_id, company_id, is_copilot)
        return
    
    if user_message.lower().startswith("download "):
        policy_name = user_message[9:].strip()
        await download_policy_with_notification(policy_name, session_id, company_id, is_copilot)
        return
    
    if user_message.lower().startswith("admin") or user_message.lower() == "open admin":
        await handle_admin_command(is_copilot)
        return
    
    if user_message.lower().startswith("policy name "):
        policy_id_str = user_message[12:].strip()
        await handle_policy_name_lookup(policy_id_str, session_id, is_copilot)
        return
    
    if user_message.lower().startswith("policy id "):
        policy_name = user_message[10:].strip()
        await handle_policy_id_lookup(policy_name, session_id, company_id, is_copilot)
        return
    
    if user_message.lower().startswith("show "):
        policy_name = user_message[5:].strip()
        await handle_show_policy(policy_name, session_id, company_id, is_copilot)
        return
    
    if is_copilot and user_message.lower().startswith("test "):
        msg = user_message[5:].strip()
        response = f"You sent: {msg}\n⚡ Response powered by static-retrieval-mrl-en-v1 embeddings!"
        await cl.Message(content=response).send()
        await db_manager.store_message(session_id, 'assistant', response)
        return
    
    if is_copilot:
        policy_match = await check_policy_mention(user_message, company_id)
        if policy_match:
            await suggest_policy_actions(policy_match, user_message)
            return
    
    try:
        search_results = await enhanced_policy_search(user_message, company_id, doc_id, is_copilot)
        if search_results:
            if is_copilot:
                response = f"Found {len(search_results)} relevant results with static embeddings:\n\n"
                for i, result in enumerate(search_results, 1):
                    confidence = result['similarity'] * 100
                    response += f"{i}. {result['doc_name']} ({confidence:.0f}% match)\n"
                    response += f"{result['answer']}\n\n"
                response += "⚡ Search powered by static-retrieval-mrl-en-v1 (100x-400x faster!)"
            else:
                response = f"""📋 Policy Information for: "{user_message}"
                ⚡ Searched using static-retrieval-mrl-en-v1 embeddings (100x-400x faster!)
                """
                for i, result in enumerate(search_results, 1):
                    confidence = result['similarity'] * 100
                    response += f"""{i}. From: {result['doc_name']} (Confidence: {confidence:.0f}%)
                    Q: {result['question']}
                    A: {result['answer']}
                    ---
                    """
                response += f"""💡 Need more specific information?
                • Type "show [policy_name]" for complete policy details
                • Ask a more specific question
                • Type "list policies" to see all available policies
                
                ⚡ All searches powered by static-retrieval-mrl-en-v1 embeddings!"""
            await cl.Message(content=response).send()
            await db_manager.store_message(session_id, 'assistant', response)
            return
        
        conversation_history = await db_manager.get_conversation_history(session_id, limit=10)
        response = await chatbot.generate_response(
            user_message,
            conversation_history,
            doc_id=doc_id,
            company_id=company_id,
            is_copilot=is_copilot
        )
        await cl.Message(content=response).send()
        await db_manager.store_message(session_id, 'assistant', response)
    except Exception as e:
        print(f"Error in main message handler: {str(e)}")
        error_response = "I'm sorry, I encountered an error while processing your question. Please try again."
        await cl.Message(content=error_response).send()
        await db_manager.store_message(session_id, 'assistant', error_response)

async def handle_show_policy(policy_name: str, session_id: str, company_id: int, is_copilot: bool = False):
    try:
        policy_info = await get_policy_details(policy_name, company_id)
        if not policy_info:
            await cl.Message(
                content=f"""❌ Policy "{policy_name}" not found.
                Type "list policies" to see all available documents."""
            ).send()
            return
        
        policy = policy_info['policy']
        faqs = policy_info['faqs']
        created_date = safe_strftime(policy.get('created_at'), '%Y-%m-%d %H:%M', 'Unknown')
        
        if is_copilot:
            response = f"""📄 {policy['doc_name']}
            📅 Added: {created_date}
            🤖 FAQs: {policy.get('total_faqs', len(faqs))}
            ⚡ Embedding: static-retrieval-mrl-en-v1
            
            Key Information:
            """
            for i, faq in enumerate(faqs[:3], 1):
                response += f"{i}. {faq['answer'][:100]}{'...' if len(faq['answer']) > 100 else ''}\n"
        else:
            response = f"""📄 Policy Document: {policy['doc_name']}
            📅 Added: {created_date}
            🤖 Total FAQs: {policy.get('total_faqs', len(faqs))}
            ⚡ Search Engine: static-retrieval-mrl-en-v1 (100x-400x faster!)
            
            📋 Frequently Asked Questions:
            """
            for i, faq in enumerate(faqs, 1):
                response += f"{i}. {faq['question']}\n"
                response += f"{faq['answer']}\n\n"
            response += "💡 Need more information? Just ask me any specific question about this policy!"
        
        await cl.Message(content=response).send()
        await db_manager.store_message(session_id, 'assistant', response)
    except Exception as e:
        print(f"Error showing policy: {e}")
        await cl.Message(content="❌ Error retrieving policy details.").send()

async def get_policy_details(policy_name: str, company_id: int):
    conn = None
    try:
        conn = await get_db_connection()
        policy = await conn.fetchrow("""
            SELECT doc_name, doc_content, created_at, faq_count as total_faqs
            FROM doc_data
            WHERE LOWER(doc_name) LIKE LOWER($1) AND company_id = $2 AND doc_status = 'completed'
            LIMIT 1
        """, f"%{policy_name}%", company_id)
        
        if not policy:
            return None
        
        faqs = await conn.fetch("""
            SELECT f.question, f.answer
            FROM faq_data f
            JOIN doc_data d ON f.doc_id = d.id
            WHERE LOWER(d.doc_name) LIKE LOWER($1) AND f.company_id = $2
            ORDER BY f.id
        """, f"%{policy_name}%", company_id)
        
        return {
            'policy': dict(policy),
            'faqs': [dict(faq) for faq in faqs]
        }
    except Exception as e:
        print(f"Error getting policy details: {e}")
        return None
    finally:
        if conn:
            await connection_pool.release(conn)

async def download_policy_file(policy_name: str, session_id: str, company_id: int, is_copilot: bool = False):
    try:
        policy_details = await db_manager.get_policy_details_by_name(policy_name, company_id)
        if not policy_details:
            response = f"❌ Policy not found: '{policy_name}'"
            await cl.Message(content=response).send()
            await db_manager.store_message(session_id, 'assistant', response)
            return
        
        policy_path = policy_details['doc_path']
        policy_full_name = policy_details['doc_name']
        
        if not os.path.exists(policy_path):
            response = f"❌ File not found: {policy_full_name}\nPath: {policy_path}"
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
                response = f"📄 {policy_full_name} - Download ready\n⚡ Processed with static embeddings"
            else:
                response = f"📄 Download Ready: {policy_full_name}\n\n📁 File: {os.path.basename(policy_path)}\n⚡ Enhanced with static-retrieval-mrl-en-v1 embeddings"
            
            await cl.Message(
                content=response,
                elements=[file_element]
            ).send()
            await db_manager.store_message(session_id, 'assistant', f"Downloaded policy: {policy_full_name}")
        except Exception as file_error:
            print(f"File creation error: {str(file_error)}")
            response = f"❌ Download Error: Could not prepare file for download.\nPolicy: {policy_full_name}"
            await cl.Message(content=response).send()
            await db_manager.store_message(session_id, 'assistant', response)
    except Exception as e:
        print(f"Error in download_policy_file: {str(e)}")
        error_response = f"❌ Error downloading policy: {policy_name}"
        await cl.Message(content=error_response).send()
        await db_manager.store_message(session_id, 'assistant', error_response)

async def handle_policy_name_lookup(policy_id_str: str, session_id: str, is_copilot: bool = False):
    try:
        policy_id = int(policy_id_str)
        policy_name = await db_manager.get_policy_name_by_id(policy_id)
        if policy_name:
            if is_copilot:
                response = f"📄 ID {policy_id}: {policy_name}\n⚡ Static embedding search ready"
            else:
                response = f"📄 Policy Name for ID {policy_id}:\n\n{policy_name}\n\n⚡ Enhanced with static-retrieval-mrl-en-v1 embeddings"
        else:
            response = f"❌ No policy found with ID: {policy_id}"
        await cl.Message(content=response).send()
        await db_manager.store_message(session_id, 'assistant', response)
    except ValueError:
        response = f"❌ Invalid ID format: '{policy_id_str}' (must be a number)"
        await cl.Message(content=response).send()
        await db_manager.store_message(session_id, 'assistant', response)
    except Exception as e:
        print(f"Error in policy name lookup: {str(e)}")
        error_response = "❌ Error looking up policy name."
        await cl.Message(content=error_response).send()
        await db_manager.store_message(session_id, 'assistant', error_response)

async def handle_policy_id_lookup(policy_name: str, session_id: str, company_id: int, is_copilot: bool = False):
    try:
        policy_id = await db_manager.get_policy_id_by_name(policy_name, company_id)
        if policy_id:
            if is_copilot:
                response = f"🔢 '{policy_name}': ID {policy_id}\n⚡ Ready for ultra-fast search"
            else:
                response = f"🔢 Policy ID for '{policy_name}':\n\nID: {policy_id}\n\n⚡ Enhanced with static-retrieval-mrl-en-v1 embeddings"
        else:
            response = f"❌ No policy found matching: '{policy_name}'"
        await cl.Message(content=response).send()
        await db_manager.store_message(session_id, 'assistant', response)
    except Exception as e:
        print(f"Error in policy ID lookup: {str(e)}")
        error_response = "❌ Error looking up policy ID."
        await cl.Message(content=error_response).send()
        await db_manager.store_message(session_id, 'assistant', error_response)

async def check_policy_mention(user_message: str, company_id: int) -> Optional[Dict]:
    try:
        available_docs = await chatbot.get_available_documents(company_id=company_id)
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
        company_id = cl.user_session.get("company_id")
        
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
        
        response = f"🎯 Detected Policy: {policy_name}\n\n📊 FAQs Available: {faq_count}\n⚡ Search Engine: static-retrieval-mrl-en-v1\n\nQuick Actions:"
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
            company_id=company_id,
            is_copilot=True
        )
        
        await cl.Message(content=f"About {policy_name}:\n\n{policy_response}").send()
        
        if session_id:
            await db_manager.store_message(session_id, 'assistant', response)
            await db_manager.store_message(session_id, 'assistant', f"About {policy_name}: {policy_response}")
    except Exception as e:
        print(f"Error suggesting policy actions: {str(e)}")
        session_id = cl.user_session.get("session_id")
        company_id = cl.user_session.get("company_id")
        conversation_history = await db_manager.get_conversation_history(session_id, limit=5)
        response = await chatbot.generate_response(user_message, conversation_history, company_id=company_id, is_copilot=True)
        await cl.Message(content=response).send()

async def handle_policy_selection_by_name(policy_name: str, company_id: int, is_copilot: bool = False):
    try:
        policy_details = await db_manager.get_policy_details_by_name(policy_name, company_id)
        if not policy_details:
            response = f"❌ Policy not found: '{policy_name}'"
            await cl.Message(content=response).send()
            return
        
        cl.user_session.set("doc_id", policy_details['id'])
        cl.user_session.set("doc_name", policy_details['doc_name'])
        
        if is_copilot:
            response = f"✅ Selected: {policy_details['doc_name']}\n\n🎯 Ready to answer questions about this policy\n⚡ Using static-retrieval-mrl-en-v1 embeddings"
        else:
            response = (
                f"✅ Policy Selected: {policy_details['doc_name']}\n\n"
                f"🆔 ID: {policy_details['id']}\n"
                f"📁 Path: {policy_details['doc_path']}\n"
                f"⚡ Search Engine: static-retrieval-mrl-en-v1 (100x-400x faster!)\n\n"
                f"I'll now focus my responses on this specific policy."
            )
        
        await cl.Message(content=response).send()
        session_id = cl.user_session.get("session_id")
        if session_id:
            await db_manager.store_message(session_id, 'assistant', response)
    except Exception as e:
        print(f"Error in handle_policy_selection_by_name: {str(e)}")
        error_response = f"❌ Error selecting policy: {policy_name}"
        await cl.Message(content=error_response).send()

async def cleanup():
    global connection_pool
    if connection_pool:
        await connection_pool.close()

if __name__ == "__main__":
    import atexit
    atexit.register(lambda: asyncio.run(cleanup()))
    print("🔐 Secure Policy Assistant Starting...")
    print(f"Database URL: {'✅ Configured' if DATABASE_URL else '❌ Missing'}")
    print(f"OpenAI API Key: {'✅ Configured' if OPENAI_API_KEY else '❌ Missing'}")
    print(f"JWT Secret: {'✅ Configured' if JWT_SECRET else '❌ Missing'}")
    print(f"⚡ Embedding Model: static-retrieval-mrl-en-v1 (100x-400x faster on CPU!)")
    print("🚀 Ready for secure authentication with ultra-fast search!")
    cl.run(fastapi_app=app)