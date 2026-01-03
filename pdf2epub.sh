#!/bin/bash
# pdf2epub - Script helper pour simplifier l'utilisation
# Usage: ./pdf2epub.sh input.pdf [options]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI_PATH="$SCRIPT_DIR/src/pdf2epub/cli.py"
GEMINI_KEY_FILE="$HOME/gemini.key"

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Vérifier que Python 3 est disponible
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 n'est pas installé${NC}"
    exit 1
fi

# Vérifier que Tesseract est disponible
if ! command -v tesseract &> /dev/null; then
    echo -e "${YELLOW}⚠️  Tesseract OCR n'est pas installé${NC}"
    echo "Installez-le avec: sudo apt-get install tesseract-ocr tesseract-ocr-fra"
    exit 1
fi

# Charger la clé Gemini si elle existe
if [ -f "$GEMINI_KEY_FILE" ]; then
    export GEMINI_API_KEY=$(cat "$GEMINI_KEY_FILE")
    echo -e "${GREEN}✓ Clé Gemini chargée${NC}"
fi

# Exécuter pdf2epub avec PYTHONPATH
cd "$SCRIPT_DIR"
PYTHONPATH=src python3 "$CLI_PATH" "$@"
