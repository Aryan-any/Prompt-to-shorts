import argparse
import os
import yaml
from dotenv import load_dotenv
from mm_story_agent import MMStoryAgent
from pathlib import Path
import json

def load_config(config_path: str):
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("--llm", type=str, default=None)
    parser.add_argument("--tts", type=str, default=None)
    parser.add_argument("--low-memory", action="store_true", default=True)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if args.llm:
        config["story_writer"]["cfg"]["llm"] = args.llm
    if args.tts:
        config["speech_generation"]["tool"] = args.tts

    agent = MMStoryAgent(low_memory_mode=args.low_memory)
    agent.call(config)

if __name__ == "__main__":
    main()
