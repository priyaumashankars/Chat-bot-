Policy Assistant
A web-based application for managing and querying policy documents using a conversational AI interface powered by OpenAI. Supports PDF uploads, FAQ generation, and semantic search.
Features

User Interface (chat.py):
Conversational chatbot for policy queries.
Policy selection via dropdown or commands (select policy [name], reset).
Download policies (download [policy_name]).
Commands: list policies, policy name [id], policy id [name].


Admin Panel (admin.py):
Upload PDFs and associate with companies.
Auto-generate FAQs using OpenAI.
Manage documents: status, delete [doc_id], reprocess [doc_id], stats.


Tech Stack:
PostgreSQL for data storage.
OpenAI (text-embedding-3-small, gpt-4o-mini) for embeddings and text processing.
Chainlit for UI, asyncpg for database interactions.
PyPDF2 for PDF text extraction.



Prerequisites

Python 3.8+
PostgreSQL
OpenAI API key
Dependencies in requirements.txt

Installation

Clone repo: git clone https://github.com/your-username/policy-assistant.git
Set up virtual environment: python -m venv venv && source venv/bin/activate
Install dependencies: pip install -r requirements.txt
Create .env:DATABASE_URL=postgresql://user:password@localhost:5432/database
OPENAI_API_KEY=your-openai-api-key
UPLOAD_FOLDER=Uploads


Run:
User interface: chainlit run chat.py
Admin panel: chainlit run admin.py



Usage

User Interface: Access at http://localhost:8000, query policies, select policies, or download files.
Admin Panel: Upload PDFs, manage documents, and view stats at http://localhost:8000.

Database Schema

faq_data: Questions, answers, embeddings, document/company IDs.
doc_data: Document metadata, content, status.
chat_sessions/chat_messages: User session and conversation history.
companies: Company names and codes.
document_status: Document processing status.

Contributing

Fork and create a feature branch.
Commit changes and push.
Open a pull request.

License
MIT License (see LICENSE).
Contact
Open an issue on GitHub for support.