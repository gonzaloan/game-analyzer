# Gameplay Highlight Detector

Automatic detection of highlight moments in gameplay videos using AWS Bedrock AI and computer vision.

Automatically detects kills, victories, achievements, epic fails, and clutch moments in any game without prior training.

---

## Features

- Universal detection: Works with any game (Call of Duty, Fortnite, Mario, etc.)
- AI analysis: Uses Amazon Bedrock models (Claude or Nova)
- Cost-effective: From $0.01 per 5-minute video with Nova Lite
- Automatic clip generation: 8-second clips ready for social media
- Detailed reports: JSON with all analysis and statistics
- Highly configurable: Adjust thresholds, models, intervals, etc.

---

## Requirements

### 1. Python 3.8+
```bash
python3 --version
```

### 2. FFmpeg
```bash
brew install ffmpeg

```

## Installation

### 1. Navigate to project directory
```bash
cd ~/game-analyzer
```

### 2. Create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate  
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

**Recommended: Use .env file**

```bash
# Copy the example file
cp .env.example .env

# Edit .env with your credentials
nano .env
```

Add your AWS credentials to `.env`:
```bash
AWS_ACCESS_KEY_ID=your-access-key-here
AWS_SECRET_ACCESS_KEY=your-secret-key-here
AWS_REGION=us-east-1
AWS_PROFILE=gmunoz-admin

# Optional: Override default settings
MODEL_CHOICE=nova-lite
SCORE_THRESHOLD=7
FRAME_INTERVAL_SECONDS=3
```

**Alternative: Export environment variables**
```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_REGION="us-east-1"
```

**Important:** Never commit `.env` to git. It's already in `.gitignore`.

---

## Usage

### Basic Commands

```bash
# Full analysis
python main.py video.mp4

# Cost estimation only (no processing)
python main.py video.mp4 --dry-run

# Custom threshold
python main.py video.mp4 --threshold 8

# Analysis without generating clips
python main.py video.mp4 --no-clips
```

### Advanced Options

```bash
# Change AI model
python main.py video.mp4 --model nova-lite
python main.py video.mp4 --model claude-3-haiku
python main.py video.mp4 --model claude-3-5-sonnet

# Adjust detection threshold
python main.py video.mp4 --threshold 9  # Only epic moments (9-10)
python main.py video.mp4 --threshold 5  # More inclusive (5+)

# Change frame analysis interval
python main.py video.mp4 --interval 5   # Every 5 seconds (cheaper)
python main.py video.mp4 --interval 1   # Every 1 second (more precise)

# Save analyzed frames for debugging
python main.py video.mp4 --save-frames
```

### Targeted Search

Find specific types of highlights by describing what you want:

```bash
# Find specific moments
python main.py video.mp4 --find "mario stomping enemies"
python main.py video.mp4 --find "headshots"
python main.py video.mp4 --find "funny fails"
python main.py video.mp4 --find "clutch moments with low health"
python main.py video.mp4 --find "power-ups"

# Combine with other options
python main.py video.mp4 --find "triple kills" --model claude-haiku-4.5 --max-clips 5
```

**How it works:**
- The AI only scores high (7+) frames that match your search query
- Regular highlights that don't match your query get low scores (0-3)
- More precise results but requires re-analyzing the video

**Examples:**
- `--find "mario growing from small to big"` - Only power-up transformations
- `--find "enemy deaths"` - Only combat moments
- `--find "victory screens"` - Only level completions
- `--find "near misses"` - Only close calls

### Help

```bash
python main.py --help
```

---

## Process Flow

1. Shows cost estimation
2. Asks for confirmation
3. Extracts frames every 3 seconds
4. Analyzes each frame with Bedrock
5. Generates clips of detected highlights
6. Saves complete JSON report

---

## Cost Estimation

### Cost per model (per 1M tokens)

| Model | Input | Output | Est. 5 min video |
|-------|-------|--------|-----------------|
| Nova Lite | $0.06 | $0.24 | $0.02 |
| Claude 3 Haiku | $0.25 | $1.25 | $0.05 |
| Claude 3.5 Sonnet | $3.00 | $15.00 | $0.50 |

**Cost factors:**
- Video duration (longer = more expensive)
- Frame interval (smaller interval = more expensive)
- Model used (Claude Sonnet is most expensive)

**Example:**
A 10-minute video with Nova Lite analyzing every 3 seconds costs approximately $0.03-0.04.

---

## Output Structure

```
game-analyzer/
├── highlights/              # Generated clips
│   ├── clip_001_score9_kill_0-15.mp4
│   ├── clip_002_score8_victory_2-45.mp4
│   └── ...
├── reports/                 # JSON reports
│   └── report_gameplay_20250101_120000.json
└── .temp_frames/           # Temporary frames (only with --save-frames)
    └── frame_0001_3.0s.jpg
```

### JSON Report Example

```json
{
  "metadata": {
    "generated_at": "2025-01-01T12:00:00",
    "model_used": "nova-lite"
  },
  "video": {
    "duration_seconds": 300,
    "resolution": "1920x1080"
  },
  "results": {
    "total_frames_analyzed": 100,
    "highlights_found": 8,
    "clips_generated": 8
  },
  "costs": {
    "total_usd": 0.0142,
    "total_input_tokens": 120000,
    "total_output_tokens": 8000
  },
  "highlights": [
    {
      "timestamp": 15.5,
      "score": 9,
      "moment_type": "kill",
      "reason": "Triple kill with headshot indicators visible",
      "visual_cues": "Kill feed showing 3 eliminations"
    }
  ]
}
```

---

## Detected Moment Types

| Type | Description | Examples |
|------|-------------|----------|
| victory | Match victory/win | "Victory", "You Win", trophies |
| kill | Eliminations | Kill feed, multi-kills, headshots |
| achievement | Unlocked achievements | Achievement popups, level up |
| epic_action | Epic action | Explosions, clutch plays, low HP |
| fail | Funny fails | Silly deaths, glitches, falls |
| other | Other moments | Interesting unclassified moments |

---

## Configuration

Edit `config.py` to customize:

### Analysis parameters
```python
FRAME_INTERVAL_SECONDS = 3    # Analysis frequency
SCORE_THRESHOLD = 7           # Minimum highlight threshold
MODEL_CHOICE = "nova-lite"    # AI model
```

### Clip parameters
```python
CLIP_DURATION = 8             # Total clip duration
CLIP_BUFFER_BEFORE = 3        # Seconds before moment
CLIP_VIDEO_CODEC = "libx264"  # Video codec
CLIP_CRF = 23                 # Quality (18-28 recommended)
```

### Frame resolution
```python
FRAME_WIDTH = 640             # Width for analysis
FRAME_HEIGHT = 360            # Height for analysis
FRAME_JPEG_QUALITY = 70       # JPEG quality (1-100)
```

### Custom Analysis Prompt

The AI prompt used for analyzing frames is stored in `analysis_prompt.txt`. You can edit this file to customize the detection behavior:

```bash
vi analysis_prompt.txt
```

This makes it easy to:
- Fine-tune detection for specific game types
- Adjust scoring criteria
- Add new highlight categories
- Modify visual cues to look for

The prompt is loaded automatically when the system starts. No code changes needed.

---

## Benchmarks

### Processing times (approximate)

| Video Duration | Frames | Nova Lite | Claude Haiku | Claude Sonnet |
|----------------|--------|-----------|--------------|---------------|
| 3 min | 60 | 45s | 60s | 90s |
| 5 min | 100 | 75s | 100s | 150s |
| 10 min | 200 | 150s | 200s | 300s |

Times include frame extraction and analysis (not clip generation).

---
