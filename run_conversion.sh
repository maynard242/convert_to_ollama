#!/usr/bin/env bash
set -eo pipefail

# Colors for output logging
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}      SEA-LION v4.5 Conversion Pipeline Orchestrator${NC}"
echo -e "${BLUE}==================================================${NC}"

# 1. Setup python virtual environment
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}[1/3] Creating python virtual environment (.venv)...${NC}"
    python3 -m venv .venv
else
    echo -e "${GREEN}[1/3] Virtual environment (.venv) already exists.${NC}"
fi

echo -e "${YELLOW}Activating virtual environment and upgrading package manager...${NC}"
source .venv/bin/activate
pip install --upgrade pip

echo -e "${YELLOW}Installing python dependencies from requirements.txt...${NC}"
pip install -r requirements.txt

# 2. Run pipeline smoke-test to verify llama.cpp compilation and conversion
echo -e "\n${BLUE}[2/3] Running pipeline smoke-test with Qwen2.5-0.5B-Instruct...${NC}"
python convert_to_ollama.py Qwen/Qwen2.5-0.5B-Instruct \
  --quant Q4_K_M

echo -e "${GREEN}Smoke-test successful! Conversion pipeline verified.${NC}"

# 3. Execute conversion for target models
echo -e "\n${BLUE}[3/3] Starting download & batch conversion of target models...${NC}"
python convert_to_ollama.py \
  aisingapore/Qwen-SEA-LION-v4.5-27B-IT \
  aisingapore/Gemma-SEA-LION-v4.5-E2B \
  --quant Q4_K_M

echo -e "\n${GREEN}==================================================${NC}"
echo -e "${GREEN}  All conversions and registration steps complete!${NC}"
echo -e "${GREEN}==================================================${NC}"
