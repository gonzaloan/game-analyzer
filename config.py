"""
Configuration file for gameplay highlight detector.
All adjustable parameters for the system.
"""

import os
from typing import Literal
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# AWS Configuration
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
AWS_PROFILE = os.getenv('AWS_PROFILE', 'gmunoz-admin')

# AI Model Selection
MODEL_CHOICE: Literal["claude-sonnet-4.5", "claude-3.7-sonnet", "claude-haiku-4.5", "claude-3.5-haiku", "claude-3-haiku", "nova-lite"] = os.getenv('MODEL_CHOICE', 'nova-lite')

MODEL_IDS = {
    "claude-sonnet-4.5": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claude-3.7-sonnet": "anthropic.claude-3-7-sonnet-20250219-v1:0",
    "claude-haiku-4.5": "anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude-3.5-haiku": "anthropic.claude-3-5-haiku-20241022-v1:0",
    "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    "nova-lite": "amazon.nova-lite-v1:0"
}

MODEL_ID = MODEL_IDS[MODEL_CHOICE]

# Model Parameters
MAX_TOKENS = 150
TEMPERATURE = 0.3

# Video Analysis Configuration
FRAME_INTERVAL_SECONDS = float(os.getenv('FRAME_INTERVAL_SECONDS', '3'))
SCORE_THRESHOLD = int(os.getenv('SCORE_THRESHOLD', '7'))

# Frame Processing
FRAME_WIDTH = int(os.getenv('FRAME_WIDTH', '640'))
FRAME_HEIGHT = int(os.getenv('FRAME_HEIGHT', '360'))
FRAME_JPEG_QUALITY = int(os.getenv('FRAME_JPEG_QUALITY', '70'))

# Clip Generation Configuration
CLIP_DURATION = int(os.getenv('CLIP_DURATION', '8'))
CLIP_BUFFER_BEFORE = int(os.getenv('CLIP_BUFFER_BEFORE', '3'))
CLIP_BUFFER_AFTER = CLIP_DURATION - CLIP_BUFFER_BEFORE
CLIP_VIDEO_CODEC = "libx264"
CLIP_ENCODING_PRESET = "veryfast"
CLIP_CRF = 23

# Cost Estimation
TOKENS_PER_IMAGE_ESTIMATE = 1200

COSTS_PER_1M_TOKENS = {
    "claude-sonnet-4.5": {"input": 3.00, "output": 15.00},
    "claude-3.7-sonnet": {"input": 3.00, "output": 15.00},
    "claude-haiku-4.5": {"input": 0.80, "output": 4.00},
    "claude-3.5-haiku": {"input": 0.80, "output": 4.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "nova-lite": {"input": 0.06, "output": 0.24}
}

BUDGET_LIMIT = float(os.getenv('BUDGET_LIMIT', '3.00'))

# Directory Configuration
OUTPUT_DIR = "highlights"
TEMP_DIR = ".temp_frames"
REPORTS_DIR = "reports"

# Logging and Debug
LOG_LEVEL = "INFO"
SAVE_DEBUG_FRAMES = False
SHOW_DETAILED_PROGRESS = True

# Analysis Prompt
def _load_prompt() -> str:
    """Load analysis prompt from file or use fallback."""
    prompt_file = os.path.join(os.path.dirname(__file__), 'analysis_prompt.txt')
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        # Fallback if file doesn't exist
        return "You are an expert gameplay highlight detector. Analyze this frame and respond with JSON only."

ANALYSIS_PROMPT = _load_prompt()


def get_model_cost(model_choice: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost for using a specific model."""
    costs = COSTS_PER_1M_TOKENS.get(model_choice, COSTS_PER_1M_TOKENS["nova-lite"])
    input_cost = (input_tokens / 1_000_000) * costs["input"]
    output_cost = (output_tokens / 1_000_000) * costs["output"]
    return input_cost + output_cost


def estimate_total_cost(video_duration_seconds: float, model_choice: str = MODEL_CHOICE) -> dict:
    """Estimate total cost for analyzing a video."""
    num_frames = int(video_duration_seconds / FRAME_INTERVAL_SECONDS)
    input_tokens = num_frames * TOKENS_PER_IMAGE_ESTIMATE
    output_tokens = num_frames * MAX_TOKENS
    total_cost = get_model_cost(model_choice, input_tokens, output_tokens)

    return {
        "video_duration": video_duration_seconds,
        "frames_to_analyze": num_frames,
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "estimated_cost_usd": round(total_cost, 2),
        "model_used": model_choice,
        "within_budget": total_cost <= BUDGET_LIMIT
    }


def print_config_summary():
    """Print current configuration summary."""
    print("\n" + "="*50)
    print("SYSTEM CONFIGURATION")
    print("="*50)
    print(f"Model: {MODEL_CHOICE}")
    print(f"Frame interval: {FRAME_INTERVAL_SECONDS}s")
    print(f"Score threshold: {SCORE_THRESHOLD}/10")
    print(f"Clip duration: {CLIP_DURATION}s ({CLIP_BUFFER_BEFORE}s before + {CLIP_BUFFER_AFTER}s after)")
    print(f"Analysis resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"Budget limit: ${BUDGET_LIMIT}")
    print("="*50 + "\n")


if __name__ == "__main__":
    print_config_summary()

    print("EXAMPLE: 5-minute video")
    estimate = estimate_total_cost(300)
    print(f"  Frames to analyze: {estimate['frames_to_analyze']}")
    print(f"  Estimated tokens: {estimate['estimated_input_tokens']:,} input + {estimate['estimated_output_tokens']:,} output")
    print(f"  Estimated cost: ${estimate['estimated_cost_usd']:.2f}")
    print(f"  Within budget? {'Yes' if estimate['within_budget'] else 'No'}")
