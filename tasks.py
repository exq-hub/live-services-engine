"""Task automation script using Invoke for LSE server management.

This module provides task automation for starting and stopping the Live Services
Engine using Uvicorn. It allows for easy command-line management of the server
with configurable host, port, and reload settings.
"""

import os

import uvicorn
from invoke import task

# Set the default values for Uvicorn server
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_RELOAD = True


@task
def run(c, host=DEFAULT_HOST, port=DEFAULT_PORT, reload=DEFAULT_RELOAD):
    """
    Launch all Exquisitor backend services.

    Args:
    c -- context (required by invoke for executing shell commands)
    host -- the host to bind the server to (default: 127.0.0.1)
    port -- the port to bind the server to (default: 8000)
    reload -- whether to enable auto-reload on file changes (default: True)
    """
    uvicorn.run("main:app", host=host, port=port, reload=reload)


@task
def stop(c):
    """
    Stop the Uvicorn server.
    """
    c.run("pkill -f 'uvicorn'")


@task
def deploy(c):
    """
    Automate deployment: Stop running instance and launch a new one.
    """
    print("Stopping existing instance...")
    stop(c)
    print("Starting new instance...")
    run(c)
