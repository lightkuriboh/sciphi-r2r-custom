import os
import psycopg # Main Psycopg 3 import
from psycopg_pool import ConnectionPool # Psycopg 3 connection pooling
# from psycopg.rows import dict_row # Optional: If you want results as dicts directly from cursor
import logging
import json
from datetime import datetime, timezone, date
from uuid import UUID
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)
if not logger.handlers: # Basic logging config if not set up by R2R
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Database Connection Parameters (from Environment Variables) ---
DB_HOST = os.getenv("RAGAI_DB_HOST")
DB_PORT = os.getenv("RAGAI_DB_PORT", "5432")
DB_NAME = os.getenv("RAGAI_DB_NAME")
DB_USER = os.getenv("RAGAI_DB_USER")
DB_PASSWORD_FILE = os.getenv("RAGAI_DB_PASSWORD_FILE")

_db_password: Optional[str] = None
if DB_PASSWORD_FILE:
    try:
        with open(DB_PASSWORD_FILE, 'r') as f:
            _db_password = f.read().strip()
    except IOError as e:
        logger.error(f"Could not read RagAI DB password from file {DB_PASSWORD_FILE}: {e}")
else:
    _db_password = os.getenv("RAGAI_DB_PASSWORD")

_connection_pool: Optional[ConnectionPool] = None

def initialize_ragai_db_connection_pool():
    """
    Initializes the Psycopg 3 connection pool for the RagAI logging database.
    This should be called once when the R2R FastAPI application starts.
    """
    global _connection_pool
    if not all([DB_HOST, DB_NAME, DB_USER, _db_password]):
        logger.error(
            "RagAI logging DB connection parameters (RAGAI_DB_HOST, RAGAI_DB_NAME, "
            "RAGAI_DB_USER, and RAGAI_DB_PASSWORD_FILE or RAGAI_DB_PASSWORD) "
            "not fully configured. Chat logging to main RagAI DB will be disabled."
        )
        _connection_pool = None
        return

    conninfo = f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={_db_password} connect_timeout=10"

    try:
        logger.info(f"Initializing RagAI logging DB connection pool for {DB_NAME} at {DB_HOST}:{DB_PORT}...")
        _connection_pool = ConnectionPool(
            conninfo=conninfo,
            min_size=1,
            max_size=5, # Adjust as needed
            # open=False # Set to False if using with FastAPI lifespan to open it manually
        )
        # Test the pool by getting a connection
        with _connection_pool.connection() as conn:
            logger.info(f"Successfully established initial connection to RagAI logging DB: {conn.info.dbname} on {conn.info.host}")
        logger.info("RagAI logging DB connection pool initialized successfully.")

    except Exception as error:
        logger.error(f"Error while initializing RagAI logging DB connection pool: {error}")
        _connection_pool = None

def shutdown_ragai_db_connection_pool():
    """
    Closes all connections in the RagAI logging DB connection pool.
    This should be called when the R2R application shuts down.
    """
    global _connection_pool
    if _connection_pool:
        try:
            logger.info("Closing RagAI logging DB connection pool...")
            _connection_pool.close() # For psycopg_pool.ConnectionPool (this is a blocking call)
            _connection_pool = None # Clear the global pool variable
            logger.info("RagAI logging DB connection pool closed successfully.")
        except Exception as e:
            logger.error(f"Error closing RagAI logging DB connection pool: {e}", exc_info=True)


def ragai_json_serializer(obj: Any) -> Any:
    """
    JSON serializer for objects not serializable by default json code.
    Handles Pydantic models, datetime, date, and UUID objects.
    """
    if hasattr(obj, 'model_dump') and callable(obj.model_dump): # Pydantic v2
        return obj.model_dump() # Returns a dict
    elif hasattr(obj, 'dict') and callable(obj.dict): # Pydantic v1
        return obj.dict() # Returns a dict
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, UUID):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def log_chat_interaction(
    session_id: str,
    customer_api_key: str,
    model_id_used: str,
    turn: int,
    role: str,
    content: str,
    retrieved_context: Optional[Dict[str, Any]] = None,
    feedback: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Optional[int]:
    """
    Logs a single chat interaction turn. Returns log_id or None.
    """
    if not _connection_pool:
        logger.warning("RagAI logging DB: Pool not available. Skipping chat log for session %s, turn %s.", session_id, turn)
        return None

    try:
        retrieved_context_json = json.dumps(retrieved_context, default=ragai_json_serializer) if retrieved_context is not None else None
        metadata_json = json.dumps(metadata, default=ragai_json_serializer) if metadata is not None else None
            

        log_id = None
        sql = """
            INSERT INTO logs.chat_logs 
            (session_id, customer_api_key, model_id_used, turn, role, content, 
            retrieved_context, feedback, timestamp, metadata)
            VALUES (%(session_id)s, %(customer_api_key)s, %(model_id_used)s, %(turn)s, %(role)s, %(content)s, 
                    %(retrieved_context)s, %(feedback)s, %(timestamp)s, %(metadata)s)
            RETURNING log_id;
        """
        params = {
            "session_id": session_id,
            "customer_api_key": customer_api_key,
            "model_id_used": model_id_used,
            "turn": turn,
            "role": role,
            "content": content,
            "retrieved_context": retrieved_context_json,
            "feedback": feedback,
            "timestamp": datetime.now(timezone.utc),
            "metadata": metadata_json,
        }
        # `with` statement ensures connection is returned to pool
        with _connection_pool.connection() as conn:
            # `with` statement ensures cursor is closed
            with conn.cursor() as cur:
                cur.execute(sql, params)
                result = cur.fetchone()
                if result:
                    log_id = result[0]
                # conn.commit() is called automatically by 'with conn:' block if no exceptions
        
        if log_id:
            logger.info(f"RagAI logging DB: Logged chat interaction for session {session_id}, turn {turn}. Log ID: {log_id}")
        else:
            logger.warning(f"RagAI logging DB: Chat interaction logged for session {session_id}, turn {turn}, but no log_id returned.")
        return log_id
    except psycopg.Error as db_err: # Catch specific Psycopg errors
        logger.error(f"RagAI logging DB (Psycopg Error): Error logging chat interaction for session {session_id}, turn {turn}: {db_err}", exc_info=True)
        # No explicit rollback needed with `with conn:` if an exception occurs, it's handled.
        return None
    except Exception as e: # Catch other potential errors
        logger.error(f"RagAI logging DB (General Error): Error logging chat interaction for session {session_id}, turn {turn}: {e}", exc_info=True)
        return None
    
def accquire_db_connection():
    """Accquires a connection from the pool. Returns None if pool is not initialized or connection fails."""
    if not _connection_pool:
        # This might happen if initialize_ragai_db_connection_pool() failed or was never called.
        # Attempting a one-time re-initialization here could be an option, but
        # it's generally better to ensure initialization happens reliably at app startup.
        logger.error("RagAI logging DB connection pool is not initialized. Cannot get connection.")
        # Consider if re-attempting initialization is desired here or just failing.
        # initialize_ragai_db_connection_pool() # Potentially re-attempt
        # if not _connection_pool: # Check again
        return None
    try:
        # The `with _connection_pool.connection() as conn:` pattern is preferred for individual operations.
        # This raw getconn is for cases where you might manage the connection lifecycle explicitly (less common now).
        # For functions like log_chat_interaction, using the `with` context manager directly is cleaner.
        return _connection_pool.getconn()
    except Exception as e:
        logger.error(f"Error accquiring connection from RagAI logging DB pool: {e}")
        return None

def release_db_connection(conn):
    """Releases a connection back to the pool, closing it if it's in an error state."""
    if _connection_pool and conn:
        try:
            _connection_pool.putconn(conn)
        except Exception as e:
            logger.error(f"Error releasing connection to RagAI logging DB pool: {e}")
            try:
                conn.close() 
            except Exception as close_err:
                logger.error(f"Error closing potentially broken DB connection: {close_err}")

def get_next_turn_numbers(session_id: str) -> tuple[int, int]:
    """
    Calculates the next user and assistant turn numbers for a given session.
    Returns (next_user_turn, next_assistant_turn).
    """
    if not _connection_pool:
        logger.warning("RagAI logging DB: Pool not available for getting turn number.")
        return (1, 2) # Fallback if DB is not available

    current_max_turn = 0
    try:
        # This `with` block correctly gets a connection from the pool
        # and ensures it's returned to the pool when the block exits,
        # whether normally or due to an exception.
        with _connection_pool.connection() as conn:
            # The `if not conn:` check here is technically redundant if _connection_pool.connection()
            # itself raises an exception on failure to get a connection, which it typically would.
            # However, it doesn't hurt as a defensive measure if the pool could return None silently (unlikely for psycopg_pool).
            # For psycopg_pool, if it can't get a connection (e.g., pool exhausted and timeout), it will raise an exception.
            
            with conn.cursor() as cur: # This `with` block ensures the cursor is closed.
                cur.execute(
                    "SELECT MAX(turn) FROM logs.chat_logs WHERE session_id = %s",
                    (session_id,)
                )
                result = cur.fetchone()
                if result and result[0] is not None:
                    current_max_turn = result[0]
        
        # If an exception occurs within the `with conn:` block, conn.commit() is NOT called,
        # and if it's an error that invalidates the connection, the pool handles it when putconn is called.
        # For a SELECT query, no commit is needed.

        user_turn = current_max_turn + 1
        assistant_turn = user_turn + 1
        return user_turn, assistant_turn
        
    except psycopg.Error as db_err: # Catch Psycopg-specific errors
        logger.error(f"RagAI logging DB (Psycopg Error): Error getting max turn for session {session_id}: {db_err}", exc_info=True)
        # The connection is already handled by the `with` block's exit.
        # No explicit conn.rollback() is needed here because SELECTs don't modify,
        # and if it were an INSERT/UPDATE that failed, the `with conn:` block handles rollback on exception.
        return (current_max_turn + 1, current_max_turn + 2) # Best effort on error
    except Exception as e: # Catch other potential errors
        logger.error(f"RagAI logging DB (General Error): Error getting max turn for session {session_id}: {e}", exc_info=True)
        return (current_max_turn + 1, current_max_turn + 2) # Best effort

def update_chat_feedback(
    log_id: int, 
    feedback_value: int, 
    feedback_text: Optional[str] = None
) -> bool:
    """
    Updates the feedback and optional feedback_text for a specific log entry.
    Returns True on success, False on failure.
    """
    if not _connection_pool:
        logger.warning("RagAI logging DB: Pool not available. Skipping feedback update for log_id %s.", log_id)
        return False

    if feedback_value not in [-1, 0, 1]:
        logger.error(f"Invalid feedback_value: {feedback_value} for log_id {log_id}. Must be -1, 0, or 1.")
        return False

    conn = None
    try:
        with _connection_pool.connection() as conn:
            with conn.cursor() as cur:
                if feedback_text is not None and len(feedback_text) > 0:
                    # Update both feedback value and merge feedback_text into metadata
                    # Using jsonb_set to add/update the feedback_text key in the metadata JSONB field.
                    # COALESCE(metadata, '{}'::jsonb) handles cases where metadata might be NULL.
                    sql = """
                        UPDATE logs.chat_logs 
                        SET 
                            feedback = %(feedback_value)s,
                            metadata = jsonb_set(
                                COALESCE(metadata, '{}'::jsonb), 
                                '{feedback_text}', 
                                %(feedback_text_json)s,
                                true -- create_missing: true, create the key if it doesn't exist
                            )
                        WHERE log_id = %(log_id)s;
                    """
                    params = {
                        "feedback_value": feedback_value, 
                        "feedback_text_json": json.dumps(feedback_text), # Store text as a JSON string
                        "log_id": log_id
                    }
                else:
                    # Only update feedback value, and remove feedback_text from metadata if it exists
                    # sql = """
                    #     UPDATE logs.chat_logs 
                    #     SET 
                    #         feedback = %(feedback_value)s,
                    #         metadata = COALESCE(metadata, '{}'::jsonb) - 'feedback_text' 
                    #     WHERE log_id = %(log_id)s;
                    # """
                    sql = """
                        UPDATE logs.chat_logs 
                        SET 
                            feedback = %(feedback_value)s
                        WHERE log_id = %(log_id)s;
                    """
                    params = {"feedback_value": feedback_value, "log_id": log_id}

                cur.execute(sql, params)
                updated_rows = cur.rowcount

        if updated_rows > 0:
            logger.info(f"RagAI logging DB: Updated feedback for log_id {log_id} to {feedback_value}, text: '{feedback_text if feedback_text else ''}'.")
            return True
        else:
            # Check if the log_id exists to give a more specific warning
            with _connection_pool.connection() as conn_check:
                with conn_check.cursor() as cur_check:
                    cur_check.execute("SELECT 1 FROM logs.chat_logs WHERE log_id = %s", (log_id,))
                    if not cur_check.fetchone():
                         logger.warning(f"RagAI logging DB: No log entry found for log_id {log_id} to update feedback.")
                         return False # Explicitly false if log_id not found
            logger.warning(f"RagAI logging DB: Log entry for log_id {log_id} found, but feedback value/text might be unchanged or update did not modify rows.")
            return True # Consider it a "success" if the record exists, even if no change was made (e.g., same feedback value)
    except psycopg.Error as db_err:
        logger.error(f"RagAI logging DB (Psycopg Error): Error updating feedback for log_id {log_id}: {db_err}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"RagAI logging DB (General Error): Error updating feedback for log_id {log_id}: {e}", exc_info=True)
        return False

# --- Lifespan Integration for R2R's FastAPI app ---
# You'll need to integrate this into R2R's main FastAPI app startup/shutdown.
# Example (conceptual, to be adapted into R2R's actual main app file):
#
# from contextlib import asynccontextmanager
# from fastapi import FastAPI
# # Assuming this module is r2r_app.core.ragai_logging_db
# from .ragai_logging_db import initialize_ragai_db_connection_pool, _connection_pool as logging_db_pool
#
# @asynccontextmanager
# async def r2r_app_lifespan(app: FastAPI):
#     print("R2R Main App: Lifespan startup...")
#     # R2R's own startup tasks (e.g., its own DB pools, loading configs)
#     # ...
#
#     # Initialize our RagAI logging DB pool
#     initialize_ragai_db_connection_pool()
#     if logging_db_pool:
#         await logging_db_pool.open() # For psycopg_pool.ConnectionPool, you might need to open it if not done by default or if `open=False` in constructor
#                                      # Check psycopg_pool docs for best practice with FastAPI lifespan
#
#     print("R2R Main App: Startup complete.")
#     yield
#     print("R2R Main App: Lifespan shutdown...")
#     # R2R's own shutdown tasks
#     # ...
#
#     # Close our RagAI logging DB pool
#     if logging_db_pool:
#         await logging_db_pool.close() # Gracefully close all connections in the pool
#         print("RagAI logging DB connection pool closed.")
#     print("R2R Main App: Shutdown complete.")
#
# # For now, a simple attempt to initialize if not already done.
# if _connection_pool is None:
#     initialize_ragai_db_connection_pool()
