#!/usr/bin/env python3
"""
Gameplay Highlight Detector
Automatic detection of highlight-worthy moments in gameplay videos using AWS Bedrock.
"""

import os
import sys
import json
import base64
import cv2
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import subprocess

import config

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Warning: tqdm not installed. Progress bars disabled.")

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False
    print("Error: boto3 not installed. Run: pip install boto3")
    sys.exit(1)


def get_video_info(video_path: str) -> Dict:
    """Get basic video information using OpenCV."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0

        return {
            "path": video_path,
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_seconds": duration,
            "duration_formatted": f"{int(duration // 60)}:{int(duration % 60):02d}"
        }
    finally:
        cap.release()


def extract_frames(video_path: str, interval_seconds: float = None) -> List[Dict]:
    """Extract frames from video at specified intervals."""
    if interval_seconds is None:
        interval_seconds = config.FRAME_INTERVAL_SECONDS

    print(f"\nExtracting frames every {interval_seconds}s...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_seconds)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    extracted_count = 0

    iterator = range(0, total_frames, frame_interval)
    if HAS_TQDM:
        iterator = tqdm(iterator, desc="Extracting frames", unit="frame")

    try:
        for target_frame in iterator:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()

            if not ret:
                break

            timestamp = target_frame / fps

            resized_frame = cv2.resize(
                frame,
                (config.FRAME_WIDTH, config.FRAME_HEIGHT),
                interpolation=cv2.INTER_AREA
            )

            _, buffer = cv2.imencode(
                '.jpg',
                resized_frame,
                [cv2.IMWRITE_JPEG_QUALITY, config.FRAME_JPEG_QUALITY]
            )
            base64_image = base64.b64encode(buffer).decode('utf-8')

            frames.append({
                "frame_number": extracted_count,
                "video_frame_number": target_frame,
                "timestamp": timestamp,
                "timestamp_formatted": f"{int(timestamp // 60)}:{int(timestamp % 60):02d}",
                "image_data": resized_frame,
                "base64": base64_image
            })

            extracted_count += 1

            if config.SAVE_DEBUG_FRAMES:
                debug_dir = Path(config.TEMP_DIR)
                debug_dir.mkdir(exist_ok=True)
                debug_path = debug_dir / f"frame_{extracted_count:04d}_{timestamp:.1f}s.jpg"
                cv2.imwrite(str(debug_path), resized_frame)

    finally:
        cap.release()

    if not HAS_TQDM:
        print(f"Extracted {extracted_count} frames")

    return frames


def create_bedrock_client():
    """Create Bedrock Runtime client using configured credentials."""
    try:
        session = boto3.Session(
            profile_name=config.AWS_PROFILE,
            region_name=config.AWS_REGION
        )
        client = session.client('bedrock-runtime')
        return client

    except NoCredentialsError:
        raise NoCredentialsError(
            f"AWS credentials not found. Configure with: aws configure --profile {config.AWS_PROFILE}"
        )
    except Exception as e:
        raise Exception(f"Error creating Bedrock client: {str(e)}")


def build_analysis_prompt(query: str = None) -> str:
    """Build the analysis prompt, optionally including a user query."""
    base_prompt = config.ANALYSIS_PROMPT

    if query:
        custom_instruction = f"""

IMPORTANT - USER IS SEARCHING FOR SPECIFIC HIGHLIGHTS:
The user is looking for: "{query}"

YOU MUST:
1. Only score high (7+) if the frame matches what the user is looking for
2. If the frame doesn't match the user's query, score it low (0-3) even if it would normally be a highlight
3. In your 'reason' field, explicitly explain if/how the frame matches the user's search

Examples:
- User searches for "mario stomping enemies" → Only score high if you see Mario stomping an enemy
- User searches for "headshots" → Only score high if you see headshot indicators
- User searches for "funny moments" → Only score high if you see something comedic/unexpected

BE STRICT: If you're not sure it matches the user's query, score it low.
"""
        return base_prompt + custom_instruction

    return base_prompt


def analyze_frame_with_bedrock(
    client,
    frame_base64: str,
    timestamp: float,
    frame_number: int,
    query: str = None
) -> Dict:
    """Analyze frame using AWS Bedrock to determine if it's a highlight.

    Args:
        query: Optional search query to find specific highlights
    """
    try:
        if config.MODEL_CHOICE.startswith("claude"):
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": config.MAX_TOKENS,
                "temperature": config.TEMPERATURE,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": frame_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": build_analysis_prompt(query)
                            }
                        ]
                    }
                ]
            }
        else:
            body = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "image": {
                                    "format": "jpeg",
                                    "source": {
                                        "bytes": frame_base64
                                    }
                                }
                            },
                            {
                                "text": build_analysis_prompt(query)
                            }
                        ]
                    }
                ],
                "inferenceConfig": {
                    "max_new_tokens": config.MAX_TOKENS,
                    "temperature": config.TEMPERATURE
                }
            }

        response = client.invoke_model(
            modelId=config.MODEL_ID,
            body=json.dumps(body)
        )

        response_body = json.loads(response['body'].read())

        if config.MODEL_CHOICE.startswith("claude"):
            content_text = response_body['content'][0]['text']
            input_tokens = response_body['usage']['input_tokens']
            output_tokens = response_body['usage']['output_tokens']
        else:
            content_text = response_body['output']['message']['content'][0]['text']
            input_tokens = response_body['usage']['inputTokens']
            output_tokens = response_body['usage']['outputTokens']

        try:
            content_text = content_text.strip()
            if content_text.startswith("```"):
                lines = content_text.split('\n')
                content_text = '\n'.join(lines[1:-1])

            analysis = json.loads(content_text)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse JSON for frame {frame_number}")
            analysis = {
                "score": 0,
                "moment_type": "error",
                "reason": "Failed to parse AI response",
                "visual_cues": content_text[:200]
            }

        cost = config.get_model_cost(
            config.MODEL_CHOICE,
            input_tokens,
            output_tokens
        )

        result = {
            "frame_number": frame_number,
            "timestamp": timestamp,
            "timestamp_formatted": f"{int(timestamp // 60)}:{int(timestamp % 60):02d}",
            "score": analysis.get("score", 0),
            "moment_type": analysis.get("moment_type", "unknown"),
            "reason": analysis.get("reason", "No reason provided"),
            "visual_cues": analysis.get("visual_cues", ""),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost
        }

        return result

    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'AccessDeniedException':
            raise Exception(
                f"Access denied to model {config.MODEL_ID}. "
                "Enable it at: https://console.aws.amazon.com/bedrock/home#/modelaccess"
            )
        else:
            raise Exception(f"AWS Error: {error_code} - {str(e)}")

    except Exception as e:
        raise Exception(f"Error analyzing frame {frame_number}: {str(e)}")


def deduplicate_highlights(highlights: List[Dict], min_distance_seconds: float = 15.0, max_clips: int = None) -> List[Dict]:
    """
    Remove duplicate highlights that are too close together in time.
    Keeps the highest-scoring highlight within each time window.
    Optionally limits total number of clips to best N.
    """
    if not highlights:
        return []

    # Sort by timestamp
    sorted_highlights = sorted(highlights, key=lambda x: x['timestamp'])

    # Group highlights that are close together
    groups = []
    current_group = [sorted_highlights[0]]

    for highlight in sorted_highlights[1:]:
        time_diff = highlight['timestamp'] - current_group[-1]['timestamp']

        if time_diff <= min_distance_seconds:
            # Add to current group
            current_group.append(highlight)
        else:
            # Start new group
            groups.append(current_group)
            current_group = [highlight]

    # Don't forget the last group
    groups.append(current_group)

    # Keep only the best from each group (highest score)
    deduplicated = []
    for group in groups:
        best = max(group, key=lambda x: x['score'])
        deduplicated.append(best)

    # Sort by score (highest first) and optionally limit to top N
    deduplicated.sort(key=lambda x: x['score'], reverse=True)

    if max_clips and len(deduplicated) > max_clips:
        deduplicated = deduplicated[:max_clips]
        # Re-sort by timestamp for cleaner output
        deduplicated.sort(key=lambda x: x['timestamp'])

    return deduplicated


def find_highlights(
    video_path: str,
    threshold: int = None,
    dry_run: bool = False,
    min_distance: float = 15.0,
    max_clips: int = None,
    query: str = None
) -> Tuple[List[Dict], List[Dict]]:
    """Find highlights in a video by analyzing frames with Bedrock.

    Args:
        query: Optional search query to find specific types of highlights
               (e.g., "mario stomping enemies", "headshots", "funny moments")
    """
    if threshold is None:
        threshold = config.SCORE_THRESHOLD

    video_info = get_video_info(video_path)

    print("\n" + "="*60)
    print("GAMEPLAY HIGHLIGHT DETECTOR")
    print("="*60)
    print(f"Video: {Path(video_path).name}")
    print(f"Duration: {video_info['duration_formatted']} ({video_info['duration_seconds']:.1f}s)")
    print(f"Resolution: {video_info['width']}x{video_info['height']}")
    print("="*60)

    cost_estimate = config.estimate_total_cost(video_info['duration_seconds'])
    print(f"\nCOST ESTIMATION")
    print(f"  Model: {config.MODEL_CHOICE}")
    print(f"  Frames to analyze: {cost_estimate['frames_to_analyze']}")
    print(f"  Estimated tokens: {cost_estimate['estimated_input_tokens']:,} input + {cost_estimate['estimated_output_tokens']:,} output")
    print(f"  Estimated cost: ${cost_estimate['estimated_cost_usd']:.2f}")

    if query:
        print(f"\nSEARCH MODE ENABLED")
        print(f"  Looking for: \"{query}\"")
        print(f"  Only frames matching this query will score high (7+)")

    if not cost_estimate['within_budget']:
        print(f"  WARNING: Exceeds budget of ${config.BUDGET_LIMIT}")

    if dry_run:
        print("\nDRY RUN mode. Video will not be processed.")
        return [], []

    print("\n" + "="*60)
    proceed = input("Continue? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Analysis cancelled.")
        return [], []

    frames = extract_frames(video_path)

    print(f"\nConnecting to AWS Bedrock...")
    try:
        client = create_bedrock_client()
        print(f"Connected to Bedrock in region {config.AWS_REGION}")
    except Exception as e:
        print(f"\n{str(e)}")
        print(f"\nSolution: Configure AWS credentials with: aws configure --profile {config.AWS_PROFILE}")
        sys.exit(1)

    print(f"\nAnalyzing frames with {config.MODEL_CHOICE}...")
    print("="*60)

    all_analyses = []
    highlights = []
    total_cost = 0.0

    iterator = enumerate(frames)
    if HAS_TQDM:
        iterator = tqdm(iterator, total=len(frames), desc="Analyzing", unit="frame")

    for i, frame_data in iterator:
        analysis = analyze_frame_with_bedrock(
            client,
            frame_data['base64'],
            frame_data['timestamp'],
            frame_data['frame_number'],
            query=query
        )

        all_analyses.append(analysis)
        total_cost += analysis['cost_usd']

        if config.SHOW_DETAILED_PROGRESS and not HAS_TQDM:
            symbol = "*" if analysis['score'] >= threshold else " "
            print(f"{symbol} [{i+1}/{len(frames)}] @ {analysis['timestamp_formatted']} "
                  f"- Score: {analysis['score']}/10 ({analysis['moment_type']})")

        if analysis['score'] >= threshold:
            highlights.append(analysis)
            if HAS_TQDM:
                iterator.set_postfix({"highlights": len(highlights), "cost": f"${total_cost:.2f}"})

    print("\n" + "="*60)
    print("ANALYSIS COMPLETED")
    print("="*60)
    print(f"Highlights found (raw): {len(highlights)}/{len(frames)} frames")

    # Deduplicate highlights
    if highlights:
        deduplicated = deduplicate_highlights(highlights, min_distance, max_clips)
        removed = len(highlights) - len(deduplicated)
        if removed > 0:
            print(f"Duplicates removed: {removed} (keeping best from each time window)")
        if max_clips and len(highlights) > max_clips:
            print(f"Limited to top {max_clips} clips by score")
        print(f"Final highlights: {len(deduplicated)}")
        highlights = deduplicated
    else:
        if query:
            print(f"\nNo highlights found matching: \"{query}\"")
            print("Try:")
            print(f"  - Using a less specific search (e.g., 'enemies' instead of 'stomping enemies')")
            print(f"  - Lowering --threshold (current: {threshold})")
            print(f"  - Using a better model (--model claude-haiku-4.5)")
            print(f"  - Running without --find to see all highlights first")
        else:
            print("\nNo highlights found above threshold.")
            print(f"Try lowering --threshold (current: {threshold})")

    print(f"\nReal cost: ${total_cost:.4f}")
    print(f"Tokens processed: {sum(a['input_tokens'] for a in all_analyses):,} input + {sum(a['output_tokens'] for a in all_analyses):,} output")
    print("="*60)

    return highlights, all_analyses


def check_ffmpeg():
    """Verify FFmpeg is installed."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise Exception()
    except:
        raise Exception(
            "FFmpeg not installed. Install it:\n"
            "  macOS: brew install ffmpeg\n"
            "  Ubuntu: sudo apt-get install ffmpeg\n"
            "  Windows: https://ffmpeg.org/download.html"
        )


def create_clip(
    video_path: str,
    highlight: Dict,
    output_dir: Path
) -> Optional[str]:
    """Create a video clip for a specific highlight using FFmpeg."""
    try:
        start_time = max(0, highlight['timestamp'] - config.CLIP_BUFFER_BEFORE)
        duration = config.CLIP_DURATION

        timestamp_str = highlight['timestamp_formatted'].replace(':', '-')
        score = highlight['score']
        moment_type = highlight['moment_type'].replace(' ', '_')
        filename = f"clip_{highlight['frame_number']:03d}_score{score}_{moment_type}_{timestamp_str}.mp4"
        output_path = output_dir / filename

        cmd = [
            'ffmpeg',
            '-ss', str(start_time),
            '-i', video_path,
            '-t', str(duration),
            '-c:v', config.CLIP_VIDEO_CODEC,
            '-preset', config.CLIP_ENCODING_PRESET,
            '-crf', str(config.CLIP_CRF),
            '-c:a', 'aac',
            '-b:a', '128k',
            '-y',
            str(output_path)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            return str(output_path)
        else:
            print(f"Warning: Error generating clip for frame {highlight['frame_number']}")
            return None

    except Exception as e:
        print(f"Error creating clip: {str(e)}")
        return None


def create_clips(video_path: str, highlights: List[Dict]) -> List[str]:
    """Generate video clips for all detected highlights."""
    if not highlights:
        print("\nNo highlights to generate clips.")
        return []

    try:
        check_ffmpeg()
    except Exception as e:
        print(f"\n{str(e)}")
        return []

    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    print(f"\nGenerating {len(highlights)} clips...")
    print(f"Output directory: {output_dir.absolute()}")
    print("="*60)

    generated_clips = []

    iterator = highlights
    if HAS_TQDM:
        iterator = tqdm(highlights, desc="Generating clips", unit="clip")

    for highlight in iterator:
        clip_path = create_clip(video_path, highlight, output_dir)

        if clip_path:
            generated_clips.append(clip_path)
            if not HAS_TQDM:
                print(f"  Created: {Path(clip_path).name}")

    print(f"\n{len(generated_clips)} clips generated successfully")

    return generated_clips


def save_report(
    video_path: str,
    video_info: Dict,
    highlights: List[Dict],
    all_analyses: List[Dict],
    generated_clips: List[str],
    total_cost: float
) -> str:
    """Save complete analysis report in JSON format."""
    reports_dir = Path(config.REPORTS_DIR)
    reports_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = Path(video_path).stem
    report_filename = f"report_{video_name}_{timestamp}.json"
    report_path = reports_dir / report_filename

    moment_types = {}
    for analysis in all_analyses:
        mtype = analysis['moment_type']
        moment_types[mtype] = moment_types.get(mtype, 0) + 1

    report = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "detector_version": "1.0.0",
            "model_used": config.MODEL_CHOICE,
            "model_id": config.MODEL_ID
        },
        "video": {
            "path": str(video_path),
            "name": Path(video_path).name,
            "duration_seconds": video_info['duration_seconds'],
            "duration_formatted": video_info['duration_formatted'],
            "resolution": f"{video_info['width']}x{video_info['height']}",
            "fps": video_info['fps']
        },
        "analysis_config": {
            "frame_interval_seconds": config.FRAME_INTERVAL_SECONDS,
            "score_threshold": config.SCORE_THRESHOLD,
            "frame_resolution": f"{config.FRAME_WIDTH}x{config.FRAME_HEIGHT}",
            "clip_duration": config.CLIP_DURATION
        },
        "results": {
            "total_frames_analyzed": len(all_analyses),
            "highlights_found": len(highlights),
            "highlight_rate": round(len(highlights) / len(all_analyses) * 100, 2) if all_analyses else 0,
            "clips_generated": len(generated_clips)
        },
        "costs": {
            "total_usd": round(total_cost, 4),
            "total_input_tokens": sum(a['input_tokens'] for a in all_analyses),
            "total_output_tokens": sum(a['output_tokens'] for a in all_analyses),
            "average_cost_per_frame": round(total_cost / len(all_analyses), 6) if all_analyses else 0
        },
        "statistics": {
            "moment_types": moment_types,
            "score_distribution": {
                "0-3": sum(1 for a in all_analyses if a['score'] <= 3),
                "4-6": sum(1 for a in all_analyses if 4 <= a['score'] <= 6),
                "7-9": sum(1 for a in all_analyses if 7 <= a['score'] <= 9),
                "10": sum(1 for a in all_analyses if a['score'] == 10)
            },
            "average_score": round(sum(a['score'] for a in all_analyses) / len(all_analyses), 2) if all_analyses else 0,
            "max_score": max((a['score'] for a in all_analyses), default=0),
            "min_score": min((a['score'] for a in all_analyses), default=0)
        },
        "highlights": highlights,
        "all_analyses": all_analyses,
        "generated_clips": generated_clips
    }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return str(report_path)


def main():
    """Main program function."""
    parser = argparse.ArgumentParser(
        description='Automatic gameplay highlight detector',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py video.mp4
  python main.py video.mp4 --threshold 8 --model claude-haiku-4.5
  python main.py video.mp4 --dry-run
  python main.py video.mp4 --max-clips 5
  python main.py video.mp4 --find "mario stomping enemies"
  python main.py video.mp4 --find "headshots" --model claude-haiku-4.5
  python main.py video.mp4 --find "funny fails" --max-clips 10
  python main.py video.mp4 --no-clips
        """
    )

    parser.add_argument(
        'video',
        help='Path to gameplay video to analyze'
    )

    parser.add_argument(
        '-t', '--threshold',
        type=int,
        default=config.SCORE_THRESHOLD,
        help=f'Minimum score for highlight (0-10). Default: {config.SCORE_THRESHOLD}'
    )

    parser.add_argument(
        '-i', '--interval',
        type=float,
        default=config.FRAME_INTERVAL_SECONDS,
        help=f'Interval between frames in seconds. Default: {config.FRAME_INTERVAL_SECONDS}'
    )

    parser.add_argument(
        '-m', '--model',
        choices=['claude-sonnet-4.5', 'claude-3.7-sonnet', 'claude-haiku-4.5', 'claude-3.5-haiku', 'claude-3-haiku', 'nova-lite'],
        default=config.MODEL_CHOICE,
        help=f'AI model to use. Default: {config.MODEL_CHOICE}'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show cost estimation without processing'
    )

    parser.add_argument(
        '--no-clips',
        action='store_true',
        help='Skip clip generation, analysis only'
    )

    parser.add_argument(
        '--save-frames',
        action='store_true',
        help='Save analyzed frames for debugging'
    )

    parser.add_argument(
        '--min-distance',
        type=float,
        default=15.0,
        help='Minimum seconds between clips (deduplication). Default: 15'
    )

    parser.add_argument(
        '--max-clips',
        type=int,
        default=None,
        help='Maximum number of clips to generate (best N). Default: unlimited'
    )

    parser.add_argument(
        '--find',
        type=str,
        default=None,
        help='Search query: only detect highlights matching this description (e.g., "mario stomping enemies", "headshots", "funny fails")'
    )

    args = parser.parse_args()

    if args.model:
        config.MODEL_CHOICE = args.model
        config.MODEL_ID = config.MODEL_IDS[args.model]

    config.FRAME_INTERVAL_SECONDS = args.interval
    config.SCORE_THRESHOLD = args.threshold
    config.SAVE_DEBUG_FRAMES = args.save_frames

    config.print_config_summary()

    try:
        highlights, all_analyses = find_highlights(
            args.video,
            threshold=args.threshold,
            dry_run=args.dry_run,
            min_distance=args.min_distance,
            max_clips=args.max_clips,
            query=args.find
        )

        if args.dry_run:
            return

        generated_clips = []
        if highlights and not args.no_clips:
            generated_clips = create_clips(args.video, highlights)
        elif not highlights:
            print("\nNo clips generated: No highlights found")
            if args.find:
                print(f"Search query '{args.find}' didn't match any frames")
        elif args.no_clips:
            print("\nClips generation skipped (--no-clips flag)")

        total_cost = sum(a['cost_usd'] for a in all_analyses)

        video_info = get_video_info(args.video)
        report_path = save_report(
            args.video,
            video_info,
            highlights,
            all_analyses,
            generated_clips,
            total_cost
        )

        print(f"\nReport saved: {report_path}")
        print("\n" + "="*60)
        print("PROCESS COMPLETED")
        print("="*60)

    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
