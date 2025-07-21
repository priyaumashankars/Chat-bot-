# chainlit_db_setup.py - Run this script to create Chainlit required tables

import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')

async def setup_chainlit_tables():
    """Create all required tables for Chainlit"""
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        print("🔄 Creating Chainlit required tables...")
        
        # # Create User table (with capital U - as Chainlit expects)
        # await conn.execute("""
        #     CREATE TABLE IF NOT EXISTS "User" (
        #         id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        #         identifier VARCHAR(255) UNIQUE NOT NULL,
        #         metadata JSONB,
        #         created_at TIMESTAMPTZ DEFAULT NOW(),
        #         updated_at TIMESTAMPTZ DEFAULT NOW()
        #     )
        # """)
        # print("✅ User table created")
        
        # Create Thread table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS "Thread" (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                created_at TIMESTAMPTZ DEFAULT NOW(),
                name VARCHAR(255),
                user_id UUID REFERENCES "User"(id) ON DELETE CASCADE,
                user_identifier VARCHAR(255),
                tags VARCHAR(255)[],
                metadata JSONB
            )
        """)
        print("✅ Thread table created")
        
        # Create Step table  
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS "Step" (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name VARCHAR(255) NOT NULL,
                type VARCHAR(50) NOT NULL,
                thread_id UUID REFERENCES "Thread"(id) ON DELETE CASCADE,
                parent_id UUID REFERENCES "Step"(id) ON DELETE CASCADE,
                disable_feedback BOOLEAN DEFAULT FALSE,
                streaming BOOLEAN DEFAULT FALSE,
                waiting_for_answer BOOLEAN DEFAULT FALSE,
                is_error BOOLEAN DEFAULT FALSE,
                metadata JSONB,
                tags VARCHAR(255)[],
                input TEXT,
                output TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                start TIMESTAMPTZ,
                "end" TIMESTAMPTZ,
                generation JSONB
            )
        """)
        print("✅ Step table created")
        
        # Create Element table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS "Element" (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                thread_id UUID REFERENCES "Thread"(id) ON DELETE CASCADE,
                step_id UUID REFERENCES "Step"(id) ON DELETE CASCADE,
                name VARCHAR(255) NOT NULL,
                type VARCHAR(50) NOT NULL,
                url TEXT,
                object_key TEXT,
                mime VARCHAR(255),
                page INTEGER,
                language VARCHAR(50),
                display VARCHAR(50),
                size INTEGER,
                auto_play BOOLEAN,
                player_config JSONB,
                for_id VARCHAR(255),
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        print("✅ Element table created")
        
        # Create Feedback table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS "Feedback" (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                for_id UUID NOT NULL,
                thread_id UUID REFERENCES "Thread"(id) ON DELETE CASCADE,
                value INTEGER NOT NULL,
                comment TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        print("✅ Feedback table created")
        
        # Update existing users table to add identifier column if missing
        try:
            await conn.execute("""
                ALTER TABLE users ADD COLUMN IF NOT EXISTS identifier VARCHAR(255) UNIQUE
            """)
            print("✅ Added identifier column to users table")
        except Exception as e:
            print(f"⚠️ Could not add identifier column: {e}")
        
        # Create indexes for performance
        await conn.execute('CREATE INDEX IF NOT EXISTS idx_user_identifier ON "User"(identifier)')
        await conn.execute('CREATE INDEX IF NOT EXISTS idx_thread_user_id ON "Thread"(user_id)')
        await conn.execute('CREATE INDEX IF NOT EXISTS idx_step_thread_id ON "Step"(thread_id)')
        await conn.execute('CREATE INDEX IF NOT EXISTS idx_element_thread_id ON "Element"(thread_id)')
        await conn.execute('CREATE INDEX IF NOT EXISTS idx_feedback_thread_id ON "Feedback"(thread_id)')
        
        print("✅ All indexes created")
        print("🎉 Chainlit database setup completed successfully!")
        
    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        raise
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(setup_chainlit_tables())