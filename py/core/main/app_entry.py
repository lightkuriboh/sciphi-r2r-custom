import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.base import R2RException
from core.utils.logging_config import configure_logging
from prometheus_fastapi_instrumentator import Instrumentator

from .app import R2RApp
from .assembly import R2RBuilder, R2RConfig
from .middleware.project_schema import ProjectSchemaMiddleware

from customization.ragai_logging_db import initialize_ragai_db_connection_pool, shutdown_ragai_db_connection_pool


log_file = configure_logging()

# Global scheduler
scheduler = AsyncIOScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("R2R App Lifespan: Main application startup initiated...")
    r2r_app = await create_r2r_app(
        config_name=config_name,
        config_path=config_path,
    )

    # Copy all routes from r2r_app to app
    app.router.routes = r2r_app.app.routes

    # Copy middleware and exception handlers
    app.middleware = r2r_app.app.middleware  # type: ignore
    app.exception_handlers = r2r_app.app.exception_handlers

    # --- RAGAI CUSTOMIZATION: Initialize RagAI Logging DB Connection Pool ---
    logging.info("R2R App Lifespan: Initializing RagAI logging database connection pool...")
    initialize_ragai_db_connection_pool()
    # --- END RAGAI CUSTOMIZATION ---

    # Start the scheduler
    logging.info("R2R App Lifespan: Starting R2R scheduler...")
    scheduler.start()
    logging.info("R2R App Lifespan: R2R scheduler started.")

    # Start the Hatchet worker
    logging.info("R2R App Lifespan: Starting R2R Hatchet worker...")
    await r2r_app.orchestration_provider.start_worker()
    logging.info("R2R App Lifespan: R2R Hatchet worker started.")

    logging.info("R2R App Lifespan: Startup complete. Application is ready.")
    yield # <--- R2R APP IS RUNNING HERE

    # Shutdown
    logging.info("R2R App Lifespan: Shutdown initiated...")
    # --- RAGAI CUSTOMIZATION: Shutdown RagAI Logging DB Connection Pool ---
    logging.info("R2R App Lifespan: Shutting down RagAI logging database connection pool...")
    shutdown_ragai_db_connection_pool() # This is a synchronous call
    # If shutdown_ragai_db_connection_pool itself involved lengthy blocking I/O
    # and this lifespan was very sensitive to blocking, you might consider
    # asyncio.to_thread for it, but for closing a pool, it's usually fine.

    Instrumentator().instrument(app).expose(app)
    logging.info("FastAPI app instrumented with Prometheus.")
    # --- END RAGAI CUSTOMIZATION ---

    logging.info("R2R App Lifespan: Shutting down R2R scheduler...")
    scheduler.shutdown()
    logging.info("R2R App Lifespan: R2R scheduler shut down.")

    # if hasattr(r2r_app.orchestration_provider, 'stop_worker') and callable(getattr(r2r_app.orchestration_provider, 'stop_worker')):
    #     logging.info("R2R App Lifespan: Stopping R2R Hatchet worker...")
    #     await r2r_app.orchestration_provider.stop_worker() # Assuming an async stop method
    #     logging.info("R2R App Lifespan: R2R Hatchet worker stopped.")
    # elif hasattr(r2r_app.orchestration_provider, 'close') and callable(getattr(r2r_app.orchestration_provider, 'close')): # Or a close method
    #      logging.info("R2R App Lifespan: Closing R2R Hatchet worker...")
    #      await r2r_app.orchestration_provider.close() # Assuming an async close method
    #      logging.info("R2R App Lifespan: R2R Hatchet worker closed.")


    logging.info("R2R App Lifespan: Shutdown complete.")


async def create_r2r_app(
    config_name: Optional[str] = "default",
    config_path: Optional[str] = None,
) -> R2RApp:
    config = R2RConfig.load(config_name=config_name, config_path=config_path)

    if (
        config.embedding.provider == "openai"
        and "OPENAI_API_KEY" not in os.environ
    ):
        raise ValueError(
            "Must set OPENAI_API_KEY in order to initialize OpenAIEmbeddingProvider."
        )

    # Build the R2RApp
    builder = R2RBuilder(config=config)
    return await builder.build()


config_name = os.getenv("R2R_CONFIG_NAME", None)
config_path = os.getenv("R2R_CONFIG_PATH", None)

if not config_path and not config_name:
    config_name = "default"
host = os.getenv("R2R_HOST", os.getenv("HOST", "0.0.0.0"))
port = int(os.getenv("R2R_PORT", "7272"))

config = R2RConfig.load(config_name=config_name, config_path=config_path)

project_name = (
    os.getenv("R2R_PROJECT_NAME") or config.app.project_name or "r2r_default"
)

logging.info(
    f"Environment R2R_IMAGE: {os.getenv('R2R_IMAGE')}",
)
logging.info(
    f"Environment R2R_CONFIG_NAME: {'None' if config_name is None else config_name}"
)
logging.info(
    f"Environment R2R_CONFIG_PATH: {'None' if config_path is None else config_path}"
)
logging.info(f"Environment R2R_PROJECT_NAME: {os.getenv('R2R_PROJECT_NAME')}")
logging.info(f"Using project name: {project_name}")
logging.info(
    f"Environment R2R_POSTGRES_HOST: {os.getenv('R2R_POSTGRES_HOST')}"
)
logging.info(
    f"Environment R2R_POSTGRES_DBNAME: {os.getenv('R2R_POSTGRES_DBNAME')}"
)
logging.info(
    f"Environment R2R_POSTGRES_PORT: {os.getenv('R2R_POSTGRES_PORT')}"
)
logging.info(
    f"Environment R2R_POSTGRES_PASSWORD: {os.getenv('R2R_POSTGRES_PASSWORD')}"
)

# Create the FastAPI app
app = FastAPI(
    lifespan=lifespan,
    log_config=None,
)


@app.exception_handler(R2RException)
async def r2r_exception_handler(request: Request, exc: R2RException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "message": exc.message,
            "error_type": type(exc).__name__,
        },
    )


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.add_middleware(
    ProjectSchemaMiddleware,
    default_schema=project_name,
)
