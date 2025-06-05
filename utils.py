"""Utility functions for the Atari Pixels project."""

import torch
import logging
import sys

def setup_device_logging(requested_device: str = None, run_name: str = "DefaultRun", log_level=logging.INFO):
    """
    Sets up the device (CUDA, MPS, or CPU) and basic logging.

    Args:
        requested_device (str, optional): Specific device to use ('cuda', 'mps', 'cpu'). 
                                          If None, auto-detects.
        run_name (str): A name for the current run/script, used in log messages.
        log_level: The logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        torch.device: The selected torch device.
    """
    # Basic logging setup
    # Remove any existing handlers for the root logger to avoid duplicate messages
    # if running in an interactive environment that might pre-configure logging.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(level=log_level,
                        format=f'%(asctime)s - {run_name} - %(levelname)s - %(message)s',
                        stream=sys.stdout) # Ensure logs go to stdout

    if requested_device:
        if requested_device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        elif requested_device == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif requested_device == "cpu":
            device = torch.device("cpu")
        else:
            logging.warning(f"Requested device '{requested_device}' not available or not recognized. Auto-detecting.")
            requested_device = None # Fallback to auto-detection
    
    if not requested_device: # Auto-detection if no specific device requested or if requested was invalid
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    
    logging.info(f"Using device: {device}")
    return device

# Example of how AtariEnv was set up, can be used for consistency if needed elsewhere
# Might be better in its own environment file if it grows more complex.
# For now, just keeping setup_device_logging here. 