#!/usr/bin/env python3
"""
Auto-Discovery Helper for Video+Annotation Pairs
=================================================

Automatically finds matching video and annotation files and generates
configuration for batch processing.

Usage:
    python scripts/generate_video_config.py --video_dir data/raw/videos --annotation_dir data/raw/annotations
"""

import os
import argparse
from pathlib import Path
import json
from typing import List, Dict, Tuple


def find_video_files(directory: str) -> List[str]:
    """Find all video files in directory (recursively)."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv'}
    video_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in video_extensions:
                video_files.append(os.path.join(root, file))
    
    return sorted(video_files)


def find_annotation_files(directory: str) -> List[str]:
    """Find all XML annotation files in directory (recursively)."""
    annotation_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.xml'):
                annotation_files.append(os.path.join(root, file))
    
    return sorted(annotation_files)


def match_videos_to_annotations(
    video_files: List[str],
    annotation_files: List[str],
    matching_strategy: str = 'basename'
) -> List[Tuple[str, str, str]]:
    """
    Match video files to annotation files.
    
    Returns:
        List of (video_path, annotation_path, suggested_name) tuples
    """
    matches = []
    
    if matching_strategy == 'basename':
        # Match based on base filename (without extension)
        video_map = {Path(v).stem: v for v in video_files}
        annotation_map = {Path(a).stem.replace('-annotations', '').replace('_annotations', ''): a 
                         for a in annotation_files}
        
        for base_name in video_map.keys():
            if base_name in annotation_map:
                matches.append((
                    video_map[base_name],
                    annotation_map[base_name],
                    base_name
                ))
    
    elif matching_strategy == 'directory':
        # Match based on parent directory
        for video in video_files:
            video_parent = Path(video).parent.name
            
            for annotation in annotation_files:
                annotation_parent = Path(annotation).parent.name
                
                if video_parent == annotation_parent:
                    suggested_name = f"{video_parent}-{Path(video).stem}"
                    matches.append((video, annotation, suggested_name))
                    break
    
    elif matching_strategy == 'manual':
        # Pair them in order (assumes same number and order)
        for i, (video, annotation) in enumerate(zip(video_files, annotation_files)):
            suggested_name = f"video{i+1:02d}"
            matches.append((video, annotation, suggested_name))
    
    return matches


def generate_config_dict(matches: List[Tuple[str, str, str]]) -> List[Dict]:
    """Generate configuration dictionary."""
    configs = []
    
    for video_path, annotation_path, name in matches:
        configs.append({
            "video": video_path,
            "annotations": annotation_path,
            "name": name
        })
    
    return configs


def generate_python_config(configs: List[Dict], output_path: str):
    """Generate Python configuration file."""
    with open(output_path, 'w') as f:
        f.write('"""Auto-generated video configuration"""\n\n')
        f.write('VIDEO_CONFIGS = [\n')
        
        for config in configs:
            f.write('    {\n')
            f.write(f'        "video": "{config["video"]}",\n')
            f.write(f'        "annotations": "{config["annotations"]}",\n')
            f.write(f'        "name": "{config["name"]}"\n')
            f.write('    },\n')
        
        f.write(']\n')


def generate_json_config(configs: List[Dict], output_path: str):
    """Generate JSON configuration file."""
    with open(output_path, 'w') as f:
        json.dump(configs, f, indent=2)


def print_config_summary(configs: List[Dict]):
    """Print summary of found configurations."""
    print("\n" + "=" * 70)
    print("📋 FOUND VIDEO+ANNOTATION PAIRS")
    print("=" * 70)
    print(f"\nTotal pairs found: {len(configs)}\n")
    
    for i, config in enumerate(configs, 1):
        print(f"{i}. {config['name']}")
        print(f"   Video: {config['video']}")
        print(f"   Annotations: {config['annotations']}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Auto-discover video and annotation pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-discover using basename matching
  python generate_video_config.py --video_dir data/raw/videos --annotation_dir data/raw/annotations

  # Use directory-based matching
  python generate_video_config.py --video_dir data/raw/videos --annotation_dir data/raw/annotations --strategy directory

  # Generate JSON instead of Python
  python generate_video_config.py --video_dir data/raw/videos --annotation_dir data/raw/annotations --format json
        """
    )
    
    parser.add_argument('--video_dir', type=str, required=True,
                       help='Directory containing video files (searched recursively)')
    parser.add_argument('--annotation_dir', type=str, required=True,
                       help='Directory containing annotation XML files (searched recursively)')
    parser.add_argument('--strategy', type=str, default='basename',
                       choices=['basename', 'directory', 'manual'],
                       help='Matching strategy (default: basename)')
    parser.add_argument('--format', type=str, default='python',
                       choices=['python', 'json'],
                       help='Output format (default: python)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (default: auto-generated)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show matches without generating config file')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔍 AUTO-DISCOVERING VIDEO+ANNOTATION PAIRS")
    print("=" * 70)
    print(f"\nVideo directory: {args.video_dir}")
    print(f"Annotation directory: {args.annotation_dir}")
    print(f"Matching strategy: {args.strategy}")
    
    # Find files
    print("\n📂 Searching for files...")
    video_files = find_video_files(args.video_dir)
    annotation_files = find_annotation_files(args.annotation_dir)
    
    print(f"   Found {len(video_files)} video files")
    print(f"   Found {len(annotation_files)} annotation files")
    
    # Match files
    print(f"\n🔗 Matching using '{args.strategy}' strategy...")
    matches = match_videos_to_annotations(video_files, annotation_files, args.strategy)
    
    if not matches:
        print("\n❌ No matches found!")
        print("\nTry a different matching strategy:")
        print("  --strategy basename  : Match by filename (default)")
        print("  --strategy directory : Match by parent directory")
        print("  --strategy manual    : Pair in order")
        return
    
    # Generate config
    configs = generate_config_dict(matches)
    
    # Print summary
    print_config_summary(configs)
    
    # Generate output file
    if not args.dry_run:
        if args.output is None:
            if args.format == 'python':
                output_path = 'video_configs_auto.py'
            else:
                output_path = 'video_configs_auto.json'
        else:
            output_path = args.output
        
        print(f"💾 Generating configuration file: {output_path}")
        
        if args.format == 'python':
            generate_python_config(configs, output_path)
            print(f"\n✅ Python config generated!")
            print(f"\n📝 To use this configuration:")
            print(f"   1. Review the generated file: {output_path}")
            print(f"   2. Update 00_batch_extract_frames.py to import it:")
            print(f"      from video_configs_auto import VIDEO_CONFIGS")
            print(f"   3. Run: python scripts/00_batch_extract_frames.py")
        else:
            generate_json_config(configs, output_path)
            print(f"\n✅ JSON config generated!")
            print(f"\n📝 To use this configuration:")
            print(f"   1. Review the generated file: {output_path}")
            print(f"   2. Load it in your script with json.load()")
    else:
        print("\n🔍 DRY RUN - No files generated")
        print("Remove --dry-run flag to generate configuration file")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
