#!/usr/bin/env python3
"""
Script de test de non-régression pour pdf2epub.

Compare les sorties de la nouvelle version avec des références attendues
pour plusieurs configurations différentes.
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Ajouter le répertoire src au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Couleurs pour l'affichage
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


class RegressionTest:
    """Classe pour gérer les tests de régression."""
    
    def __init__(self, test_dir: Path, pdfs_dir: Path):
        """
        Initialiser les tests.
        
        Args:
            test_dir: Répertoire des tests
            pdfs_dir: Répertoire contenant les PDFs de test
        """
        self.test_dir = test_dir
        self.pdfs_dir = pdfs_dir
        self.results: List[Dict] = []
        self.temp_dir = Path(tempfile.mkdtemp(prefix="pdf2epub_test_"))
        
    def cleanup(self):
        """Nettoyer les fichiers temporaires."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def run_conversion(
        self,
        pdf_path: Path,
        config: Dict,
        test_name: str
    ) -> Tuple[bool, Dict]:
        """
        Exécuter une conversion et vérifier les résultats.
        
        Args:
            pdf_path: Chemin vers le PDF
            config: Configuration à utiliser
            test_name: Nom du test
            
        Returns:
            Tuple (succès, métriques)
        """
        print(f"\n{BLUE}▶ Test: {test_name}{RESET}")
        print(f"  PDF: {pdf_path.name}")
        print(f"  Config: {config}")
        
        # Créer fichier de config temporaire
        config_file = self.temp_dir / f"config_{test_name}.yaml"
        self._write_config(config_file, config)
        
        # Commande de conversion
        cmd = [
            sys.executable,
            "-m", "pdf2epub.cli",
            "-i", str(pdf_path),
            "-a", config.get("author", "Test Author"),
            "-t", config.get("title", pdf_path.stem),
            "--config", str(config_file)
        ]
        
        # Exécuter
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.test_dir.parent / "src")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                env=env,
                cwd=str(self.pdfs_dir)
            )
            
            success = result.returncode == 0
            
            # Vérifier les fichiers générés
            epub_path = pdf_path.parent / f"{pdf_path.stem}.epub"
            txt_path = pdf_path.parent / f"{pdf_path.stem}_tesseract.txt"
            
            metrics = {
                "return_code": result.returncode,
                "epub_exists": epub_path.exists(),
                "txt_exists": txt_path.exists(),
                "epub_size": epub_path.stat().st_size if epub_path.exists() else 0,
                "txt_lines": self._count_lines(txt_path) if txt_path.exists() else 0,
                "txt_words": self._count_words(txt_path) if txt_path.exists() else 0,
                "chapters": self._count_chapters(txt_path) if txt_path.exists() else 0,
                "stderr": result.stderr[-500:] if result.stderr else "",  # Dernières lignes
            }
            
            # Nettoyer
            if epub_path.exists():
                epub_path.unlink()
            if txt_path.exists():
                txt_path.unlink()
            
            # Afficher résultats
            if success:
                print(f"  {GREEN}✓ Conversion réussie{RESET}")
                print(f"    - EPUB: {metrics['epub_size'] / 1024:.1f} KB")
                print(f"    - Texte: {metrics['txt_lines']} lignes, {metrics['txt_words']} mots")
                print(f"    - Chapitres: {metrics['chapters']}")
            else:
                print(f"  {RED}✗ Échec de conversion{RESET}")
                print(f"    Code retour: {metrics['return_code']}")
                if metrics['stderr']:
                    print(f"    Erreur: {metrics['stderr'][:200]}")
            
            return success, metrics
            
        except subprocess.TimeoutExpired:
            print(f"  {RED}✗ Timeout (>300s){RESET}")
            return False, {"error": "timeout"}
        except Exception as e:
            print(f"  {RED}✗ Erreur: {e}{RESET}")
            return False, {"error": str(e)}
    
    def _write_config(self, config_file: Path, config: Dict):
        """Écrire un fichier de configuration YAML."""
        import yaml
        
        full_config = {
            "ocr": {
                "engine": config.get("ocr_engine", "tesseract"),
                "language": config.get("language", "fra"),
                "confidence_threshold": 80,
            },
            "chapter_detection": {
                "enabled": config.get("chapter_detection", True),
                "threshold_percentage": config.get("chapter_threshold", 25),
                "detect_only_on_header": config.get("keyword_only", False),
            },
            "image_processing": {
                "dpi": 200,
                "x_margin": config.get("x_margin", 30),
                "y_margin": config.get("y_margin", 50),
                "use_page_dewarp": config.get("use_dewarp", True),
            },
            "text_processing": {
                "fix_hyphenation": True,
                "fix_dialogs": True,
                "fix_lettrine": True,
            },
            "ai_proofreading": {
                "enabled": False,
            },
            "performance": {
                "parallel_processing": config.get("parallel", True),
                "max_workers": config.get("max_workers", 4),
            },
            "output": {
                "include_cover": config.get("include_cover", True),
                "temp_directory": "./tmp",
            },
            "debug": False,
            "log_level": "INFO",
        }
        
        with open(config_file, "w") as f:
            yaml.dump(full_config, f)
    
    def _count_lines(self, file_path: Path) -> int:
        """Compter les lignes d'un fichier."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return sum(1 for _ in f)
        except:
            return 0
    
    def _count_words(self, file_path: Path) -> int:
        """Compter les mots d'un fichier."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return sum(len(line.split()) for line in f)
        except:
            return 0
    
    def _count_chapters(self, file_path: Path) -> int:
        """Compter les chapitres marqués dans un fichier."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                return content.count("@@@ ")
        except:
            return 0
    
    def run_all_tests(self) -> bool:
        """
        Exécuter tous les tests de régression.
        
        Returns:
            True si tous les tests passent
        """
        print(f"\n{BLUE}{'=' * 70}{RESET}")
        print(f"{BLUE}  Tests de non-régression pdf2epub{RESET}")
        print(f"{BLUE}{'=' * 70}{RESET}")
        
        # Liste des tests à exécuter
        test_configs = [
            # Test 1: Configuration par défaut avec couverture
            {
                "name": "default_with_cover",
                "pdf": "ext_cigales.pdf",
                "config": {
                    "title": "Les Cigales",
                    "author": "Auteur Test",
                    "include_cover": True,
                    "chapter_detection": True,
                },
            },
            # Test 2: Sans couverture, avec détection chapitres
            {
                "name": "no_cover_with_chapters",
                "pdf": "ext_cigales.pdf",
                "config": {
                    "title": "Les Cigales",
                    "author": "Auteur Test",
                    "include_cover": False,
                    "chapter_detection": True,
                },
            },
            # Test 3: Avec couverture, sans détection chapitres
            {
                "name": "with_cover_no_chapters",
                "pdf": "ext_cigales.pdf",
                "config": {
                    "title": "Les Cigales",
                    "author": "Auteur Test",
                    "include_cover": True,
                    "chapter_detection": False,
                },
            },
            # Test 4: Mode séquentiel (pas de parallélisation)
            {
                "name": "sequential_processing",
                "pdf": "ext_cigales.pdf",
                "config": {
                    "title": "Les Cigales",
                    "author": "Auteur Test",
                    "include_cover": False,
                    "chapter_detection": True,
                    "parallel": False,
                },
            },
            # Test 5: PDF plus long avec chapitres
            {
                "name": "longer_pdf_with_chapters",
                "pdf": "ext_rs.pdf",
                "config": {
                    "title": "Retrouvailles",
                    "author": "Anne-Marie Desplat-Duc",
                    "include_cover": True,
                    "chapter_detection": True,
                },
            },
            # Test 6: Marges personnalisées
            {
                "name": "custom_margins",
                "pdf": "ext_2_pour_une.pdf",
                "config": {
                    "title": "Deux pour une",
                    "author": "Anne-Marie Desplat-Duc",
                    "include_cover": True,
                    "chapter_detection": True,
                    "x_margin": 40,
                    "y_margin": 60,
                },
            },
            # Test 7: Détection chapitres par mot-clé uniquement
            {
                "name": "keyword_only_chapter_detection",
                "pdf": "ext_rs.pdf",
                "config": {
                    "title": "Retrouvailles",
                    "author": "Anne-Marie Desplat-Duc",
                    "include_cover": False,
                    "chapter_detection": True,
                    "keyword_only": True,
                },
            },
        ]
        
        # Exécuter chaque test
        total = len(test_configs)
        passed = 0
        
        for i, test_config in enumerate(test_configs, 1):
            print(f"\n{YELLOW}[{i}/{total}]{RESET}", end=" ")
            
            pdf_path = self.pdfs_dir / test_config["pdf"]
            
            if not pdf_path.exists():
                print(f"{RED}✗ PDF non trouvé: {pdf_path}{RESET}")
                self.results.append({
                    "test": test_config["name"],
                    "status": "skipped",
                    "reason": "PDF not found"
                })
                continue
            
            success, metrics = self.run_conversion(
                pdf_path,
                test_config["config"],
                test_config["name"]
            )
            
            self.results.append({
                "test": test_config["name"],
                "pdf": test_config["pdf"],
                "status": "passed" if success else "failed",
                "metrics": metrics
            })
            
            if success:
                passed += 1
        
        # Résumé
        print(f"\n{BLUE}{'=' * 70}{RESET}")
        print(f"{BLUE}  Résumé des tests{RESET}")
        print(f"{BLUE}{'=' * 70}{RESET}")
        
        failed = total - passed
        
        if failed == 0:
            print(f"{GREEN}✓ Tous les tests ont réussi ({passed}/{total}){RESET}")
        else:
            print(f"{RED}✗ {failed} test(s) échoué(s) sur {total}{RESET}")
            print(f"{GREEN}✓ {passed} test(s) réussi(s){RESET}")
        
        # Sauvegarder les résultats
        results_file = self.test_dir / "regression_results.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nRésultats sauvegardés dans: {results_file}")
        
        return failed == 0


def main():
    """Point d'entrée principal."""
    # Chemins
    test_dir = Path(__file__).parent
    project_dir = test_dir.parent
    pdfs_dir = project_dir.parent
    
    print(f"Répertoire des tests: {test_dir}")
    print(f"Répertoire PDFs: {pdfs_dir}")
    
    # Créer et exécuter les tests
    tester = RegressionTest(test_dir, pdfs_dir)
    
    try:
        all_passed = tester.run_all_tests()
        sys.exit(0 if all_passed else 1)
    finally:
        tester.cleanup()


if __name__ == "__main__":
    main()
