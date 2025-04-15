#!/usr/bin/env python3
"""
GlossPreprocess - Dataset Analysis and Problematic File Identification

This script serves as the first step in the GlossAPI pipeline, combining:
1. Dataset analysis - Analyzes files and collects metadata
2. Problematic file identification - Detects files that may cause issues in processing

The script analyzes files in a directory, extracts metadata, validates file structure,
and identifies potentially problematic files based on specific criteria:
- File extension issues (missing, multiple, or unusual extensions)
- Suspiciously small file sizes
- PDF-specific issues (font embedding, subsetting problems)

Usage:
    python gloss_preprocess.py --input_dir /path/to/input [--output_dir /path/to/output]

Outputs:
    - preprocessing_analysis_results.csv: Detailed analysis of all files
    - problematic_files.txt: List of files that may cause issues in processing
    - preprocessing_report.md: Comprehensive report with statistics and findings
"""

import os
import sys
import argparse
import csv
import re
import subprocess
import datetime
import logging
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Union, Tuple, Any

# Import libraries for specific file type analysis
import chardet
try:
    from PyPDF2 import PdfReader
except ImportError:
    print("Warning: PyPDF2 not installed. PDF validation will be limited.")
    PdfReader = None

try:
    from lxml import etree
except ImportError:
    print("Warning: lxml not installed. XML/HTML validation will be limited.")
    etree = None

try:
    import zipfile
except ImportError:
    print("Warning: zipfile module not available. DOCX validation will be limited.")
    zipfile = None

class GlossPreprocess:
    """
    A class for analyzing datasets and identifying problematic files.
    
    This class provides functionality to:
    1. Analyze files and collect metadata
    2. Identify potentially problematic files based on specific criteria
    3. Generate comprehensive reports on the dataset
    
    It serves as the first step in the GlossAPI pipeline to ensure that only
    high-quality files proceed to the extraction and processing stages.
    """
    
    def __init__(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        log_level: int = logging.INFO
    ):
        """
        Initialize the GlossPreprocess processor.
        
        Args:
            input_dir: Directory containing input files to analyze
            output_dir: Directory for output files (default: creates 'reports/preprocessing' in parent of input_dir)
            log_level: Logging level (default: logging.INFO)
        """
        # Setup logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Store paths
        self.input_dir = Path(input_dir)
        
        # Set up output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Create centralized reports directory with preprocessing subdirectory
            self.output_dir = self.input_dir.parent / "reports" / "preprocessing"
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up output file paths
        self.analysis_csv = self.output_dir / "preprocessing_analysis_results.csv"
        self.problematic_files_txt = self.output_dir / "problematic_files.txt"
        self.report_md = self.output_dir / "preprocessing_report.md"
        
        # Initialize results storage
        self.analysis_results = []
        self.problematic_files = []
        
        self.logger.info(f"Initialized GlossPreprocess with input directory: {self.input_dir}")
        self.logger.info(f"Output files will be saved to: {self.output_dir}")

    # ===== FILE COLLECTION METHODS =====
    
    def collect_files(self) -> List[Path]:
        """
        Collect all files from the input directory and its subdirectories.
        
        Returns:
            List of Path objects for files to process
        """
        files_to_process = []
        
        # Check if there are subdirectories
        has_subdirectories = any(p.is_dir() for p in self.input_dir.iterdir())
        
        if has_subdirectories:
            self.logger.info(f"Subdirectories detected in {self.input_dir}. Processing recursively...")
        
        # Walk the directory tree
        for path in self.input_dir.rglob('*'):
            if path.is_file():
                files_to_process.append(path)
        
        self.logger.info(f"Found {len(files_to_process)} files to process")
        return files_to_process

    # ===== FILE METADATA EXTRACTION METHODS =====
    
    def get_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Get basic file metadata.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing file metadata
        """
        # Get file stats
        stats = file_path.stat()
        
        # Extract metadata
        metadata = {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "file_extension": file_path.suffix.lower().lstrip('.') if file_path.suffix else "",
            "file_size_bytes": stats.st_size,
            "last_character": file_path.name[-1] if file_path.name else "",
            "extension_count": len([ext for ext in file_path.name.split('.')[1:] if ext])
        }
        
        return metadata

    # ===== FILE CONTENT ANALYSIS METHODS =====
    
    def analyze_file_encoding(self, file_path: Path) -> Dict[str, Any]:
        """
        Analyze file encoding.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing encoding information
        """
        encoding_info = {"detected_encoding": None}
        
        try:
            # Read first 4KB of file to detect encoding
            with open(file_path, "rb") as f:
                raw_data = f.read(4096)
            
            # Detect encoding
            detection_result = chardet.detect(raw_data)
            encoding_info["detected_encoding"] = detection_result['encoding']
            
        except Exception as e:
            encoding_info["error"] = str(e)
            self.logger.warning(f"Error analyzing encoding of {file_path}: {e}")
        
        return encoding_info

    def validate_file_structure(self, file_path: Path, file_extension: str) -> Dict[str, Any]:
        """
        Validate file structure for specific file types.
        
        Args:
            file_path: Path to the file
            file_extension: File extension
            
        Returns:
            Dictionary containing validation results
        """
        validation_info = {"structure_validation_error": None}
        
        try:
            # PDF validation
            if file_extension == "pdf" and PdfReader:
                try:
                    with open(file_path, "rb") as f:
                        PdfReader(f)
                except Exception as e:
                    validation_info["structure_validation_error"] = f"PDF validation error: {str(e)}"
                    
            # XML validation
            elif file_extension == "xml" and etree:
                try:
                    with open(file_path, "rb") as f:
                        etree.parse(f)
                except Exception as e:
                    validation_info["structure_validation_error"] = f"XML validation error: {str(e)}"
                    
            # HTML validation
            elif file_extension in ["html", "htm"] and etree:
                try:
                    with open(file_path, "rb") as f:
                        etree.parse(f)
                except Exception as e:
                    validation_info["structure_validation_error"] = f"HTML validation error: {str(e)}"
                    
            # DOCX validation
            elif file_extension == "docx" and zipfile:
                if not zipfile.is_zipfile(file_path):
                    validation_info["structure_validation_error"] = "Invalid DOCX file (not a ZIP archive)"
        
        except Exception as e:
            validation_info["structure_validation_error"] = f"Validation error: {str(e)}"
            self.logger.warning(f"Error validating structure of {file_path}: {e}")
        
        return validation_info

    # ===== PDF-SPECIFIC ANALYSIS METHODS =====
    
    def analyze_pdf_fonts(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Analyze PDF fonts using pdffonts command-line tool.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing font information
        """
        pdf_info = {
            "total_font_count": 0,
            "identity_h_font_count": 0,
            "embedded_font_count": 0,
            "subsetted_font_count": 0,
            "unicode_mapping_count": 0,
            "pdffonts_output": ""
        }
        
        try:
            # Run pdffonts to get font information
            result = subprocess.run(['pdffonts', str(pdf_path)], 
                                   capture_output=True, 
                                   text=True, 
                                   check=True)
            
            output = result.stdout
            pdf_info["pdffonts_output"] = output
            
            # Parse the pdffonts output
            lines = output.strip().split('\n')
            if len(lines) < 3:  # Header + separator + at least one font
                return pdf_info
            
            # Skip the first two lines (header and separator)
            font_lines = lines[2:]
            pdf_info["total_font_count"] = len(font_lines)
            
            for line in font_lines:
                # Split the line into columns
                cols = re.split(r'\s+', line.strip(), maxsplit=7)
                if len(cols) >= 7:
                    if cols[-5] == "Identity-H":
                        pdf_info["identity_h_font_count"] += 1
                    
                    if cols[-4].lower() == "yes":
                        pdf_info["embedded_font_count"] += 1
                    
                    if cols[-3].lower() == "yes":
                        pdf_info["subsetted_font_count"] += 1
                    
                    if cols[-2].lower() == "yes":
                        pdf_info["unicode_mapping_count"] += 1
        
        except subprocess.CalledProcessError:
            pdf_info["error"] = "Error executing pdffonts"
            self.logger.warning(f"Error executing pdffonts on {pdf_path}")
        except Exception as e:
            pdf_info["error"] = str(e)
            self.logger.warning(f"Error analyzing PDF fonts in {pdf_path}: {e}")
        
        return pdf_info

    # ===== PROBLEMATIC FILE IDENTIFICATION METHODS =====
    
    def check_pdf_criteria(self, row: Dict[str, Any]) -> bool:
        """
        Check if a PDF file meets criteria for being problematic.
        
        Extremely strict criteria targeting specific problematic files:
        1. Very low font embedding percentage (below 15%)
        2. Has Identity-H fonts but almost no subsetting (below 10%)
        3. Combination of these issues
        
        Args:
            row: A dictionary representing a row from the analysis results
            
        Returns:
            bool: True if problematic, False otherwise
        """
        try:
            total_font_count = int(row['total_font_count']) if row['total_font_count'] else 0
            embedded_font_count = int(row['embedded_font_count']) if row['embedded_font_count'] else 0
            identity_h_font_count = int(row['identity_h_font_count']) if row['identity_h_font_count'] else 0
            subsetted_font_count = int(row['subsetted_font_count']) if row['subsetted_font_count'] else 0
        except (ValueError, KeyError):
            return False
        
        if total_font_count == 0:
            return False
        
        embedding_percentage = (embedded_font_count / total_font_count * 100) if total_font_count > 0 else 100
        subsetting_percentage = (subsetted_font_count / total_font_count * 100) if total_font_count > 0 else 100
        
        return (
            identity_h_font_count > 0 and 
            subsetting_percentage < 10 and 
            embedding_percentage < 15
        )

    def check_file_criteria(self, row: Dict[str, Any]) -> bool:
        """
        Check if a file meets criteria for being problematic based on its file type.
        
        Args:
            row: A dictionary representing a row from the analysis results
            
        Returns:
            bool: True if problematic, False otherwise
        """
        file_extension = row['file_extension'].lower()
        file_size_bytes = int(row['file_size_bytes']) if row['file_size_bytes'] else 0
        extension_count = int(row['extension_count']) if row['extension_count'] else 0
        
        # Check for extension issues
        if extension_count == 0 or extension_count > 1:
            return True
        
        # Check for unusual file extensions (containing non-alphanumeric characters)
        if any(c in file_extension for c in "[](){}'\",;:"):
            return True
        
        # Check for suspiciously small files (potentially empty or corrupted)
        recognized_extensions = ['html', 'docx', 'pdf', 'xml', 'doc', 'txt', 'md', 'csv', 'json']
        
        if file_extension in recognized_extensions:
            if file_extension == 'html' and file_size_bytes < 100:
                return True
            elif file_extension in ['docx', 'doc'] and file_size_bytes < 2500:
                return True
            elif file_extension == 'pdf' and file_size_bytes < 800:
                return True
            elif file_extension == 'xml' and file_size_bytes < 50:
                return True
            elif file_extension == 'txt' and file_size_bytes < 10:
                return True
            elif file_extension == 'md' and file_size_bytes < 20:
                return True
            elif file_extension in ['csv', 'json'] and file_size_bytes < 30:
                return True
        
        # PDF-specific criteria
        if file_extension == 'pdf':
            return self.check_pdf_criteria(row)
        
        return False
    
    # ===== FILE PROCESSING METHODS =====
    
    def process_file(self, file_path: Path, verbose: bool = False) -> Dict[str, Any]:
        """
        Process a single file and collect all measurements.
        
        Args:
            file_path: Path to the file
            verbose: Whether to print detailed progress information
            
        Returns:
            Dictionary containing all collected measurements
        """
        if verbose:
            self.logger.debug(f"Processing {file_path}...")
        
        # Get file metadata
        results = self.get_file_metadata(file_path)
        file_extension = results["file_extension"]
        
        # Analyze file encoding
        results.update(self.analyze_file_encoding(file_path))
        
        # Validate file structure
        results.update(self.validate_file_structure(file_path, file_extension))
        
        # PDF-specific analysis
        if file_extension == "pdf":
            results.update(self.analyze_pdf_fonts(file_path))
        
        return results

    def process_files(self, files_to_process: List[Path], verbose: bool = False) -> List[Dict[str, Any]]:
        """
        Process all files and collect results.
        
        Args:
            files_to_process: List of file paths to process
            verbose: Whether to print detailed progress information
            
        Returns:
            List of dictionaries containing results for each file
        """
        results = []
        total_files = len(files_to_process)
        
        self.logger.info(f"Processing {total_files} files...")
        
        for i, file_path in enumerate(files_to_process):
            # Calculate and display progress percentage
            progress_pct = (i + 1) / total_files * 100
            
            # Update progress every 5% or for every file if verbose
            if verbose or i == 0 or i == total_files - 1 or int(progress_pct) % 5 == 0:
                self.logger.info(f"Progress: {progress_pct:.1f}% ({i+1}/{total_files}) - Processing: {file_path.name}")
            
            file_results = self.process_file(file_path, verbose)
            results.append(file_results)
        
        return results
    
    def identify_problematic_files(self, analysis_results: List[Dict[str, Any]]) -> List[str]:
        """
        Identify problematic files based on file-specific criteria.
        
        Args:
            analysis_results: List of dictionaries containing analysis results for each file
            
        Returns:
            List of problematic file paths
        """
        problematic_files = []
        
        self.logger.info("Identifying problematic files...")
        
        for row in analysis_results:
            file_path = row['file_path']
            
            # Check if file meets criteria for being problematic
            if self.check_file_criteria(row):
                problematic_files.append(file_path)
        
        self.logger.info(f"Found {len(problematic_files)} problematic files")
        return problematic_files
    
    # ===== RESULTS WRITING METHODS =====
    
    def write_analysis_to_csv(self, results: List[Dict[str, Any]], output_csv: Path) -> None:
        """
        Write analysis results to CSV file.
        
        Args:
            results: List of dictionaries containing results for each file
            output_csv: Path to output CSV file
        """
        # Define the order of columns (first columns in the desired order)
        priority_fields = [
            "file_name", 
            "file_path", 
            "file_extension", 
            "file_size_bytes"
        ]
        
        # Get all possible field names from results
        all_fields = set()
        for result in results:
            all_fields.update(result.keys())
        
        # Create fieldnames with priority fields first, then the rest alphabetically
        fieldnames = priority_fields + sorted(list(all_fields - set(priority_fields)))
        
        # Write results to CSV
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        self.logger.info(f"Analysis results saved to {output_csv}")
    
    def write_problematic_files_to_txt(self, problematic_files: List[str], output_txt: Path) -> None:
        """
        Write problematic file paths to a text file.
        
        Args:
            problematic_files: List of problematic file paths
            output_txt: Path to output text file
        """
        with open(output_txt, 'w', encoding='utf-8') as f:
            for file_path in problematic_files:
                f.write(f"{file_path}\n")
        
        self.logger.info(f"Problematic file paths saved to {output_txt}")
    
    # ===== REPORT GENERATION METHODS =====
    
    def generate_report(self, analysis_results: List[Dict[str, Any]], problematic_files: List[str], report_file: Union[str, Path]) -> bool:
        """
        Generate a comprehensive report with statistics and findings.
        
        Args:
            analysis_results: List of dictionaries with analysis results for each file
            problematic_files: List of problematic file paths
            report_file: Path to save the report
            
        Returns:
            True if report was generated successfully, False otherwise
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(report_file), exist_ok=True)
            
            # Process analysis results to extract statistics
            total_files = len(analysis_results)
            
            # Count file extensions
            file_extensions = Counter()
            
            # Track problematic files by criteria
            problematic_by_criteria = defaultdict(int)
            
            # PDF-specific statistics
            pdf_stats = {
                'total': 0,
                'with_identity_h': 0,
                'with_embedding': 0,
                'with_subsetting': 0,
                'with_unicode': 0,
                'embedding_percentages': [],
                'subsetting_percentages': [],
                'identity_h_percentages': []
            }
            
            problematic_set = set(problematic_files)
            
            for row in analysis_results:
                total_files += 1
                file_extension = row['file_extension'].lower()
                file_size_bytes = int(row['file_size_bytes']) if row['file_size_bytes'] else 0
                file_path = row['file_path']
                extension_count = int(row['extension_count']) if row['extension_count'] else 0
                file_extensions[file_extension] += 1
                
                # Count problematic files by criteria
                if file_path in problematic_set:
                    # Extension issues
                    if extension_count == 0:
                        problematic_by_criteria['missing_extension'] += 1
                    elif extension_count > 1:
                        problematic_by_criteria['multiple_extensions'] += 1
                    elif any(c in file_extension for c in "[](){}'\",;:"):
                        problematic_by_criteria['unusual_extension'] += 1
                    # Size issues
                    elif file_extension == 'html' and file_size_bytes < 100:
                        problematic_by_criteria['small_html'] += 1
                    elif file_extension == 'docx' and file_size_bytes < 2500:
                        problematic_by_criteria['small_docx'] += 1
                    elif file_extension == 'xml' and file_size_bytes < 50:
                        problematic_by_criteria['small_xml'] += 1
                    elif file_extension == 'pdf':
                        if file_size_bytes < 800:
                            problematic_by_criteria['small_pdf'] += 1
                        else:
                            # Must be problematic due to font issues
                            problematic_by_criteria['pdf_font_issues'] += 1
                
                # Collect PDF-specific statistics
                if file_extension == 'pdf':
                    pdf_stats['total'] += 1
                    
                    try:
                        total_font_count = int(row['total_font_count']) if row.get('total_font_count') else 0
                        embedded_font_count = int(row['embedded_font_count']) if row.get('embedded_font_count') else 0
                        identity_h_font_count = int(row['identity_h_font_count']) if row.get('identity_h_font_count') else 0
                        subsetted_font_count = int(row['subsetted_font_count']) if row.get('subsetted_font_count') else 0
                        unicode_mapping_count = int(row['unicode_mapping_count']) if row.get('unicode_mapping_count') else 0
                        
                        if identity_h_font_count > 0:
                            pdf_stats['with_identity_h'] += 1
                        
                        if embedded_font_count > 0:
                            pdf_stats['with_embedding'] += 1
                        
                        if subsetted_font_count > 0:
                            pdf_stats['with_subsetting'] += 1
                            
                        if unicode_mapping_count > 0:
                            pdf_stats['with_unicode'] += 1
                            
                        # Calculate percentages for each PDF
                        if total_font_count > 0:
                            embedding_pct = embedded_font_count / total_font_count * 100
                            subsetting_pct = subsetted_font_count / total_font_count * 100
                            identity_h_pct = identity_h_font_count / total_font_count * 100
                            
                            pdf_stats['embedding_percentages'].append(embedding_pct)
                            pdf_stats['subsetting_percentages'].append(subsetting_pct)
                            pdf_stats['identity_h_percentages'].append(identity_h_pct)
                    except (ValueError, KeyError):
                        # Skip if conversion fails
                        pass
            
            # Generate the report
            with open(report_file, 'w', encoding='utf-8') as f:
                # Report header
                f.write("# Preprocessing Analysis Report\n\n")
                f.write(f"*Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
                
                # Overall statistics
                f.write("## Overall Statistics\n\n")
                f.write(f"- **Total Files Analyzed**: {total_files}\n")
                f.write(f"- **Problematic Files Identified**: {len(problematic_files)} ({(len(problematic_files)/total_files*100 if total_files > 0 else 0):.2f}%)\n\n")
                
                # Problematic files breakdown
                f.write("### Problematic Files Breakdown\n\n")
                
                criteria_labels = [
                    "Files with missing extension",
                    "Files with multiple extensions",
                    "Files with unusual extensions",
                    "Suspiciously small HTML files",
                    "Suspiciously small DOCX files",
                    "Suspiciously small PDF files",
                    "Suspiciously small XML files",
                    "PDF files with font issues"
                ]
                criteria_width = max(len("Criteria"), max(len(label) for label in criteria_labels))
                count_width = len("Count")
                
                f.write("| " + "Criteria".ljust(criteria_width) + " | " + "Count".ljust(count_width) + " |\n")
                f.write("| " + "-" * criteria_width + " | " + "-" * count_width + " |\n")
                
                criteria_values = [
                    ("Files with missing extension", problematic_by_criteria['missing_extension']),
                    ("Files with multiple extensions", problematic_by_criteria['multiple_extensions']),
                    ("Files with unusual extensions", problematic_by_criteria['unusual_extension']),
                    ("Suspiciously small HTML files", problematic_by_criteria['small_html']),
                    ("Suspiciously small DOCX files", problematic_by_criteria['small_docx']),
                    ("Suspiciously small PDF files", problematic_by_criteria['small_pdf']),
                    ("Suspiciously small XML files", problematic_by_criteria['small_xml']),
                    ("PDF files with font issues", problematic_by_criteria['pdf_font_issues'])
                ]
                
                for criteria, count in criteria_values:
                    f.write("| " + criteria.ljust(criteria_width) + " | " + str(count).ljust(count_width) + " |\n")
                f.write("\n")
                
                # File type distribution
                f.write("## File Type Distribution\n\n")
                
                ext_width = max(len("File Extension"), max(len(ext) for ext in file_extensions.keys()))
                count_width = max(len("Count"), max(len(str(count)) for count in file_extensions.values()))
                pct_width = len("Percentage")
                
                f.write("| " + "File Extension".ljust(ext_width) + " | " + "Count".ljust(count_width) + " | " + "Percentage".ljust(pct_width) + " |\n")
                f.write("| " + "-" * ext_width + " | " + "-" * count_width + " | " + "-" * pct_width + " |\n")
                
                for ext, count in file_extensions.most_common():
                    percentage = count / total_files * 100 if total_files > 0 else 0
                    f.write("| " + ext.ljust(ext_width) + " | " + str(count).ljust(count_width) + " | " + f"{percentage:.2f}%".ljust(pct_width) + " |\n")
                
                # PDF-specific statistics
                if pdf_stats['total'] > 0:
                    f.write("## PDF-Specific Statistics\n\n")
                    f.write(f"- **Total PDF Files**: {pdf_stats['total']}\n")
                    f.write(f"- **PDFs with Identity-H Fonts**: {pdf_stats['with_identity_h']} ({pdf_stats['with_identity_h']/pdf_stats['total']*100:.2f}%)\n")
                    f.write(f"- **PDFs with Embedded Fonts**: {pdf_stats['with_embedding']} ({pdf_stats['with_embedding']/pdf_stats['total']*100:.2f}%)\n")
                    f.write(f"- **PDFs with Subsetted Fonts**: {pdf_stats['with_subsetting']} ({pdf_stats['with_subsetting']/pdf_stats['total']*100:.2f}%)\n")
                    f.write(f"- **PDFs with Unicode Mapping**: {pdf_stats['with_unicode']} ({pdf_stats['with_unicode']/pdf_stats['total']*100:.2f}%)\n\n")
                    
                    if pdf_stats['embedding_percentages']:
                        avg_embedding = sum(pdf_stats['embedding_percentages']) / len(pdf_stats['embedding_percentages'])
                        avg_subsetting = sum(pdf_stats['subsetting_percentages']) / len(pdf_stats['subsetting_percentages'])
                        avg_identity_h = sum(pdf_stats['identity_h_percentages']) / len(pdf_stats['identity_h_percentages'])
                        
                        f.write("### Font Statistics Averages\n\n")
                        f.write(f"- **Average Font Embedding Percentage**: {avg_embedding:.2f}%\n")
                        f.write(f"- **Average Font Subsetting Percentage**: {avg_subsetting:.2f}%\n")
                        f.write(f"- **Average Identity-H Font Percentage**: {avg_identity_h:.2f}%\n\n")
                
                # Applied criteria explanation
                f.write("## Applied Criteria\n\n")
                
                f.write("### File Extension Criteria\n\n")
                f.write("Files are considered problematic if they have:\n\n")
                f.write("- Missing file extension (extension count = 0)\n")
                f.write("- Multiple file extensions (extension count > 1)\n")
                f.write("- Unusual file extensions (containing non-alphanumeric characters like brackets, quotes, etc.)\n\n")
                
                f.write("### File Size Criteria\n\n")
                f.write("Files are considered problematic if they are suspiciously small:\n\n")
                f.write("- HTML files: < 100 bytes\n")
                f.write("- DOCX files: < 2,500 bytes\n")
                f.write("- PDF files: < 800 bytes\n")
                f.write("- XML files: < 50 bytes\n\n")
                
                f.write("### PDF Criteria\n\n")
                f.write("A PDF file is considered problematic if it meets ALL of the following criteria:\n\n")
                f.write("1. Has Identity-H fonts\n")
                f.write("2. Font subsetting percentage is below 10%\n")
                f.write("3. Font embedding percentage is below 15%\n\n")
                
                # List of problematic files
                if problematic_files:
                    f.write("## Problematic Files\n\n")
                    f.write("The following files were identified as problematic based on the specified criteria:\n\n")
                    for file_path in problematic_files:
                        f.write(f"- `{file_path}`\n")
            
            self.logger.info(f"Comprehensive report generated at {report_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
    
    # ===== MAIN EXECUTION METHODS =====
    
    def run(self, verbose: bool = False) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            verbose: Whether to print detailed progress information
            
        Returns:
            Tuple containing:
                - List of dictionaries with analysis results for each file
                - List of problematic file paths
        """
        # Step 1: Collect files
        files_to_process = self.collect_files()
        
        # Step 2: Process files and collect results
        self.analysis_results = self.process_files(files_to_process, verbose)
        
        # Step 3: Identify problematic files
        self.problematic_files = self.identify_problematic_files(self.analysis_results)
        
        # Step 4: Write results to output files
        self.write_analysis_to_csv(self.analysis_results, self.analysis_csv)
        self.write_problematic_files_to_txt(self.problematic_files, self.problematic_files_txt)
        
        # Step 5: Generate comprehensive report
        self.generate_report(self.analysis_results, self.problematic_files, self.report_md)
        
        return self.analysis_results, self.problematic_files
    
    def get_clean_files(self) -> List[str]:
        """
        Get a list of files that are not identified as problematic.
        
        This method can be used to filter out problematic files before
        proceeding with the rest of the GlossAPI pipeline.
        
        Returns:
            List of clean file paths
        """
        if not self.analysis_results:
            self.logger.warning("No analysis results available. Run the preprocessing pipeline first.")
            return []
        
        if not self.problematic_files:
            # If no problematic files were identified, return all files
            return [row['file_path'] for row in self.analysis_results]
        
        # Create a set of problematic files for faster lookup
        problematic_set = set(self.problematic_files)
        
        # Return only files that are not in the problematic set
        return [row['file_path'] for row in self.analysis_results if row['file_path'] not in problematic_set]


# ===== COMMAND LINE INTERFACE =====

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='GlossPreprocess - Dataset Analysis and Problematic File Identification'
    )
    parser.add_argument('--input_dir', required=True, help='Directory containing files to analyze')
    parser.add_argument('--output_dir', required=False, help='Directory for output files')
    parser.add_argument('--verbose', action='store_true', help='Print detailed progress information')
    return parser.parse_args()

def main():
    """Main function to run the script."""
    # Parse arguments
    args = parse_arguments()
    
    # Ensure input directory exists
    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)
    
    # Initialize preprocessor
    preprocessor = GlossPreprocess(input_dir=input_dir, output_dir=args.output_dir)
    
    # Run preprocessing pipeline
    analysis_results, problematic_files = preprocessor.run(verbose=args.verbose)
    
    # Print summary
    print(f"\nPreprocessing complete!")
    print(f"Total files analyzed: {len(analysis_results)}")
    print(f"Problematic files identified: {len(problematic_files)}")
    print(f"\nOutput files:")
    print(f"- Analysis results: {preprocessor.analysis_csv}")
    print(f"- Problematic files: {preprocessor.problematic_files_txt}")
    print(f"- Comprehensive report: {preprocessor.report_md}")

if __name__ == "__main__":
    main()
