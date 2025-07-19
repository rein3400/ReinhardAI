from fastapi import FastAPI, APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

# Import our modules
from models import (
    User, UserCreate, UserLogin, UserResponse,
    ChatRequest, ChatResponse, ChatMessage,
    ChatSession, ChatSessionCreate, ChatSessionResponse,
    AgentTaskCreate, AgentTask, ModelInfo
)
from auth import create_access_token, verify_token
from database import database, init_admin_user
from openrouter_client import openrouter_client
from chatgpt_agent import get_agent, cleanup_agent

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Create the main app
app = FastAPI(title="AI Multi-Agent Platform", version="1.0.0")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Security
security = HTTPBearer()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dependency to get current user
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current authenticated user."""
    username = verify_token(credentials.credentials)
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = await database.get_user_by_username(username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user

# Auth endpoints
@api_router.post("/auth/register", response_model=UserResponse)
async def register(user_data: UserCreate):
    """Register a new user."""
    try:
        user = await database.create_user(user_data)
        return UserResponse(**user.dict())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@api_router.post("/auth/login")
async def login(user_data: UserLogin):
    """Login user."""
    user = await database.authenticate_user(user_data.username, user_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    access_token = create_access_token(data={"sub": user.username})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": UserResponse(**user.dict())
    }

@api_router.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information."""
    return UserResponse(**current_user.dict())

# OpenRouter endpoints
@api_router.get("/models")
async def get_available_models(current_user: User = Depends(get_current_user)):
    """Get available OpenRouter models."""
    try:
        models = await openrouter_client.get_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch models")

@api_router.post("/chat")
async def chat_completion(
    request: ChatRequest, 
    current_user: User = Depends(get_current_user)
):
    """Send chat completion request."""
    try:
        # Check if user has enough credits (unless admin)
        if not current_user.is_admin:
            current_credits = await database.get_user_credits(current_user.id)
            if current_credits <= 0:
                raise HTTPException(status_code=402, detail="Insufficient credits")
        
        # Prepare messages for OpenRouter
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Send request to OpenRouter
        response = await openrouter_client.chat_completion(
            messages=messages,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        # Calculate tokens used
        total_tokens = response.get("usage", {}).get("total_tokens", 1)
        
        # Deduct credits (unless admin)
        if not current_user.is_admin:
            await database.update_user_credits(
                current_user.id, 
                -total_tokens, 
                f"Chat completion using {request.model}"
            )
        
        # Return response
        return {
            "id": response.get("id"),
            "content": response["choices"][0]["message"]["content"],
            "model": response.get("model"),
            "usage": response.get("usage"),
            "created_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        if "Insufficient credits" in str(e):
            raise HTTPException(status_code=402, detail="Insufficient credits")
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")

# Chat session endpoints
@api_router.post("/chat/sessions", response_model=ChatSessionResponse)
async def create_chat_session(
    session_data: ChatSessionCreate,
    current_user: User = Depends(get_current_user)
):
    """Create a new chat session."""
    session = await database.create_chat_session(
        user_id=current_user.id,
        title=session_data.title,
        model=session_data.model
    )
    
    return ChatSessionResponse(
        id=session.id,
        user_id=session.user_id,
        title=session.title,
        model=session.model,
        message_count=len(session.messages),
        created_at=session.created_at,
        updated_at=session.updated_at
    )

@api_router.get("/chat/sessions", response_model=List[ChatSessionResponse])
async def get_chat_sessions(current_user: User = Depends(get_current_user)):
    """Get user's chat sessions."""
    sessions = await database.get_user_chat_sessions(current_user.id)
    return [
        ChatSessionResponse(
            id=session.id,
            user_id=session.user_id,
            title=session.title,
            model=session.model,
            message_count=len(session.messages),
            created_at=session.created_at,
            updated_at=session.updated_at
        )
        for session in sessions
    ]

@api_router.get("/chat/sessions/{session_id}")
async def get_chat_session(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get specific chat session."""
    session = await database.get_chat_session(session_id, current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    return session

@api_router.delete("/chat/sessions/{session_id}")
async def delete_chat_session(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete chat session."""
    await database.delete_chat_session(session_id, current_user.id)
    return {"message": "Chat session deleted"}

# Agent endpoints
@api_router.post("/agents/tasks")
async def create_agent_task(
    task_data: AgentTaskCreate,
    current_user: User = Depends(get_current_user)
):
    """Create a new agent task."""
    task = await database.create_agent_task(
        user_id=current_user.id,
        agent_type=task_data.agent_type,
        task_description=task_data.task_description
    )
    return task

@api_router.get("/agents/tasks")
async def get_agent_tasks(current_user: User = Depends(get_current_user)):
    """Get user's agent tasks."""
    tasks = await database.get_user_agent_tasks(current_user.id)
    return tasks

@api_router.get("/agents/tasks/{task_id}")
async def get_agent_task(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get specific agent task."""
    task = await database.get_agent_task(task_id, current_user.id)
    if not task:
        raise HTTPException(status_code=404, detail="Agent task not found")
    return task

# Admin endpoints
@api_router.get("/admin/users")
async def get_all_users(current_user: User = Depends(get_current_user)):
    """Get all users (admin only)."""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    cursor = database.db.users.find({}, {"password_hash": 0})
    users = []
    async for user_data in cursor:
        users.append(UserResponse(**user_data))
    return users

@api_router.post("/admin/users/{user_id}/credits")
async def update_user_credits_admin(
    user_id: str,
    credits: int,
    current_user: User = Depends(get_current_user)
):
    """Update user credits (admin only)."""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    await database.update_user_credits(
        user_id,
        credits,
        f"Credits updated by admin {current_user.username}"
    )
    return {"message": f"Credits updated for user {user_id}"}

# Health check
@api_router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@api_router.get("/")
async def root():
    return {"message": "AI Multi-Agent Platform API", "version": "1.0.0"}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize the application."""
    logger.info("Starting AI Multi-Agent Platform...")
    await init_admin_user()
    logger.info("Application started successfully!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    await database.close()
    logger.info("Application shutdown complete.")
