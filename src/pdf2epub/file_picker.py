"""
Simple terminal-based file picker using tkinter.
Falls back to manual input if GUI is not available.
"""
import os
from pathlib import Path
from typing import Optional


def pick_file(
    title: str = "Sélectionner un fichier PDF",
    initial_dir: Optional[str] = None,
    filetypes: Optional[list[tuple[str, str]]] = None
) -> Optional[str]:
    """
    Open a GUI file picker dialog.
    
    Args:
        title: Dialog title
        initial_dir: Initial directory to open
        filetypes: List of (description, pattern) tuples
        
    Returns:
        Selected file path or None if cancelled
    """
    if filetypes is None:
        filetypes = [("Fichiers PDF", "*.pdf"), ("Tous les fichiers", "*.*")]
    
    if initial_dir is None:
        initial_dir = os.getcwd()
    
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        # Create root window (hidden)
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        # Open file dialog
        file_path = filedialog.askopenfilename(
            title=title,
            initialdir=initial_dir,
            filetypes=filetypes
        )
        
        # Cleanup
        root.destroy()
        
        return file_path if file_path else None
        
    except ImportError:
        print("⚠️  tkinter not available, please enter path manually")
        return None
    except Exception as e:
        print(f"⚠️  Error opening file selector: {e}")
        return None


def pick_file_terminal(
    prompt: str = "PDF file path",
    initial_dir: Optional[str] = None,
    file_extension: str = ".pdf"
) -> str:
    """
    Fallback text-based file picker with file listing.
    
    Args:
        prompt: Prompt message
        initial_dir: Initial directory
        file_extension: Extension to filter (e.g., '.pdf')
        
    Returns:
        Selected file path
    """
    if initial_dir is None:
        initial_dir = os.getcwd()
    
    current_dir = Path(initial_dir).resolve()
    
    # List matching files in current directory AND parent
    print(f"\n📂 Found {file_extension} files:")
    print("─" * 75)
    
    all_files = []
    
    # Check current directory
    current_files = sorted([f for f in current_dir.glob(f"*{file_extension}")])
    if current_files:
        print(f"\n  In {current_dir}:")
        for f in current_files:
            all_files.append(f)
    
    # Check parent directory
    parent_dir = current_dir.parent
    parent_files = sorted([f for f in parent_dir.glob(f"*{file_extension}")])
    if parent_files:
        print(f"\n  In {parent_dir}:")
        for f in parent_files:
            all_files.append(f)
    
    # Check subdirectories one level deep
    for subdir in sorted(current_dir.iterdir()):
        if subdir.is_dir() and not subdir.name.startswith('.'):
            sub_files = sorted(list(subdir.glob(f"*{file_extension}"))[:3])  # Limit to 3 per dir
            if sub_files:
                print(f"\n  In {subdir.name}/:")
                for f in sub_files:
                    all_files.append(f)
    
    if all_files:
        print()
        for idx, file in enumerate(all_files, 1):
            # Show relative path if possible
            try:
                rel_path = file.relative_to(current_dir)
                print(f"  {idx}. {rel_path}")
            except ValueError:
                print(f"  {idx}. {file}")
        print()
        
        # Try numeric selection
        choice = input(f"File number (or full path): ").strip()
        
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(all_files):
                return str(all_files[idx])
    else:
        print("  No files found in nearby directories.\n")
    
    # Manual input fallback
    while True:
        path_input = input(f"{prompt}: ").strip()
        if path_input:
            path_obj = Path(path_input).expanduser().resolve()
            if path_obj.exists() and path_obj.suffix.lower() == file_extension.lower():
                return str(path_obj)
            else:
                print(f"❌ File not found or not a {file_extension}. Try again.")
        else:
            print("❌ Please enter a file path.")


def pick_pdf_interactive() -> str:
    """
    Interactive PDF file picker with GUI fallback to terminal.
    
    Returns:
        Path to selected PDF file
    """
    print("📂 PDF file selection...")
    print("─" * 75)
    
    # Try GUI first
    gui_path = pick_file(
        title="Select PDF to convert",
        initial_dir=os.getcwd(),
        filetypes=[("PDF Files", "*.pdf"), ("All files", "*.*")]
    )
    
    if gui_path:
        print(f"✓ File selected: {Path(gui_path).name}")
        return gui_path
    
    # Fallback to terminal picker
    return pick_file_terminal(
        prompt="PDF file path",
        initial_dir=os.getcwd(),
        file_extension=".pdf"
    )
