#!/bin/bash
#
# Script de test de non-régression pour pdf2epub
# Compare les sorties avec différentes configurations
#

set -e

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Compteurs
TOTAL=0
PASSED=0
FAILED=0

# Répertoires
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PDFS_DIR="$(dirname "$PROJECT_DIR")"
SRC_DIR="$PROJECT_DIR/src"

echo -e "${BLUE}======================================================================"
echo -e "  Tests de non-régression pdf2epub"
echo -e "======================================================================${NC}"
echo ""
echo "Répertoire projet: $PROJECT_DIR"
echo "Répertoire PDFs: $PDFS_DIR"
echo ""

# Fonction pour exécuter un test
run_test() {
    local test_name="$1"
    local pdf_file="$2"
    shift 2
    local extra_args="$@"
    
    TOTAL=$((TOTAL + 1))
    
    echo -e "\n${YELLOW}[${TOTAL}] Test: ${test_name}${NC}"
    echo "  PDF: $pdf_file"
    echo "  Args: $extra_args"
    
    # Nettoyer les fichiers précédents
    rm -f "$PDFS_DIR/${pdf_file%.pdf}.epub"
    rm -f "$PDFS_DIR/${pdf_file%.pdf}_tesseract.txt"
    rm -rf "$PROJECT_DIR/tmp"
    
    # Exécuter la conversion
    cd "$PROJECT_DIR"
    if PYTHONPATH="$SRC_DIR" python3 -m pdf2epub.cli \
        -i "$PDFS_DIR/$pdf_file" \
        $extra_args > /tmp/test_output.log 2>&1; then
        
        # Vérifier que les fichiers sont créés
        if [ -f "$PDFS_DIR/${pdf_file%.pdf}.epub" ] && [ -f "$PDFS_DIR/${pdf_file%.pdf}_tesseract.txt" ]; then
            epub_size=$(du -h "$PDFS_DIR/${pdf_file%.pdf}.epub" | cut -f1)
            txt_lines=$(wc -l < "$PDFS_DIR/${pdf_file%.pdf}_tesseract.txt")
            txt_words=$(wc -w < "$PDFS_DIR/${pdf_file%.pdf}_tesseract.txt")
            chapters=$(grep -c "@@@ " "$PDFS_DIR/${pdf_file%.pdf}_tesseract.txt" || echo "0")
            
            echo -e "  ${GREEN}✓ Conversion réussie${NC}"
            echo "    - EPUB: $epub_size"
            echo "    - Texte: $txt_lines lignes, $txt_words mots"
            echo "    - Chapitres: $chapters"
            
            PASSED=$((PASSED + 1))
            
            # Nettoyer
            rm -f "$PDFS_DIR/${pdf_file%.pdf}.epub"
            rm -f "$PDFS_DIR/${pdf_file%.pdf}_tesseract.txt"
        else
            echo -e "  ${RED}✗ Fichiers de sortie manquants${NC}"
            FAILED=$((FAILED + 1))
        fi
    else
        echo -e "  ${RED}✗ Échec de la conversion${NC}"
        FAILED=$((FAILED + 1))
    fi
}

# Créer un fichier de config de base
cat > "$PROJECT_DIR/test_regression_config.yaml" <<EOF
ocr:
  engine: tesseract
  language: fra
  confidence_threshold: 80

chapter_detection:
  enabled: true
  threshold_percentage: 25
  detect_only_on_header: false

image_processing:
  dpi: 200
  x_margin: 30
  y_margin: 50
  use_page_dewarp: true

text_processing:
  fix_hyphenation: true
  fix_dialogs: true
  fix_lettrine: true

ai_proofreading:
  enabled: false

performance:
  parallel_processing: true
  max_workers: 4

output:
  include_cover: true
  temp_directory: ./tmp

debug: false
log_level: INFO
EOF

# Test 1: Configuration par défaut avec couverture
run_test "default_with_cover" \
    "ext_cigales.pdf" \
    -a "Auteur Test" -t "Les Cigales" \
    --config "$PROJECT_DIR/test_regression_config.yaml"

# Test 2: Sans couverture, avec détection chapitres
sed 's/include_cover: true/include_cover: false/' "$PROJECT_DIR/test_regression_config.yaml" > "$PROJECT_DIR/test_regression_config_nocover.yaml"
run_test "no_cover_with_chapters" \
    "ext_cigales.pdf" \
    -a "Auteur Test" -t "Les Cigales" \
    --config "$PROJECT_DIR/test_regression_config_nocover.yaml"

# Test 3: Avec couverture, sans détection chapitres
sed 's/enabled: true/enabled: false/' "$PROJECT_DIR/test_regression_config.yaml" > "$PROJECT_DIR/test_regression_config_nochap.yaml"
run_test "with_cover_no_chapters" \
    "ext_cigales.pdf" \
    -a "Auteur Test" -t "Les Cigales" \
    --config "$PROJECT_DIR/test_regression_config_nochap.yaml"

# Test 4: Mode séquentiel
sed 's/parallel_processing: true/parallel_processing: false/' "$PROJECT_DIR/test_regression_config_nocover.yaml" > "$PROJECT_DIR/test_regression_config_seq.yaml"
run_test "sequential_processing" \
    "ext_cigales.pdf" \
    -a "Auteur Test" -t "Les Cigales" \
    --config "$PROJECT_DIR/test_regression_config_seq.yaml"

# Test 5: PDF plus long avec chapitres
if [ -f "$PDFS_DIR/ext_rs.pdf" ]; then
    run_test "longer_pdf_with_chapters" \
        "ext_rs.pdf" \
        -a "Anne-Marie Desplat-Duc" -t "Retrouvailles" \
        --config "$PROJECT_DIR/test_regression_config.yaml"
else
    echo -e "${YELLOW}⊘ Test skipped: ext_rs.pdf non trouvé${NC}"
    TOTAL=$((TOTAL + 1))
fi

# Test 6: Marges personnalisées
if [ -f "$PDFS_DIR/ext_2_pour_une.pdf" ]; then
    sed -e 's/x_margin: 30/x_margin: 40/' -e 's/y_margin: 50/y_margin: 60/' \
        "$PROJECT_DIR/test_regression_config.yaml" > "$PROJECT_DIR/test_regression_config_margins.yaml"
    run_test "custom_margins" \
        "ext_2_pour_une.pdf" \
        -a "Anne-Marie Desplat-Duc" -t "Deux pour une" \
        --config "$PROJECT_DIR/test_regression_config_margins.yaml"
else
    echo -e "${YELLOW}⊘ Test skipped: ext_2_pour_une.pdf non trouvé${NC}"
    TOTAL=$((TOTAL + 1))
fi

# Test 7: Détection chapitres par mot-clé uniquement
if [ -f "$PDFS_DIR/ext_rs.pdf" ]; then
    sed -e 's/include_cover: true/include_cover: false/' -e 's/detect_only_on_header: false/detect_only_on_header: true/' \
        "$PROJECT_DIR/test_regression_config.yaml" > "$PROJECT_DIR/test_regression_config_keyword.yaml"
    run_test "keyword_only_chapter_detection" \
        "ext_rs.pdf" \
        -a "Anne-Marie Desplat-Duc" -t "Retrouvailles" \
        --config "$PROJECT_DIR/test_regression_config_keyword.yaml"
else
    echo -e "${YELLOW}⊘ Test skipped: ext_rs.pdf non trouvé${NC}"
    TOTAL=$((TOTAL + 1))
fi

# Nettoyage final
rm -f "$PROJECT_DIR"/test_regression_config*.yaml
rm -rf "$PROJECT_DIR/tmp"

# Résumé
echo ""
echo -e "${BLUE}======================================================================"
echo -e "  Résumé des tests"
echo -e "======================================================================${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ Tous les tests ont réussi ($PASSED/$TOTAL)${NC}"
    exit 0
else
    echo -e "${RED}✗ $FAILED test(s) échoué(s) sur $TOTAL${NC}"
    echo -e "${GREEN}✓ $PASSED test(s) réussi(s)${NC}"
    exit 1
fi
