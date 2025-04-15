#!/usr/bin/env python3
"""
GlossClean - Document Cleaning Module for GlossAPI

This module provides a unified, modular cleaning system for the GlossAPI pipeline.
It combines document cleaning, tag handling, and quality enhancement into a single
component that can be optionally integrated into the pipeline.

Features:
- HTML/XML tag cleaning with proper Greek replacements
- Text quality enhancement (HTML entities, character normalization)
- Document quality assessment (optional)
- Memory-efficient processing
- Code example preservation (for programming books)
- Parquet file cleaning and processing

Usage:
    # As a standalone component
    cleaner = GlossClean()
    cleaner.process_directory(input_dir, output_dir)
    
    # Or integrated into the Corpus pipeline
    corpus.clean(verify_tags=True, assess_quality=False)
    
    # For processing programming books
    cleaner = GlossClean(preserve_code_examples=True)
    cleaner.process_directory(input_dir, output_dir)
    
    # For cleaning parquet files
    cleaner = GlossClean()
    cleaner.clean_parquet_tags(parquet_file)
    
    # For exploring parquet files
    GlossClean.explore_parquet(parquet_file)
"""

import os
import re
import html
import unicodedata
import json
import logging
import shutil
from pathlib import Path
from collections import Counter
from typing import Dict, List, Set, Tuple, Union, Optional
import datetime
import pandas as pd
import pyarrow.parquet as pq
import sys

# Default mapping of tag names to their descriptive labels
DEFAULT_TAG_MAPPING = {
    'NoDocSe': 'Αριθμός Εγγράφου:',
    'Date': 'Ημερομηνία:',
    'TitreType': 'Ειδικός Τύπος Εγγράφου:',
    'TitreRecueil': 'Νομική Αναφορά:',
    'Titre': 'Τίτλος:',
    'TitreSuite': 'Υπότιτλος:',
    'Depute': 'Συντάκτης:',
    'RefProcLect': 'Αναφορά Διαδικασίας:',
    'DocRef': 'Αναφορά Εγγράφου:',
    'Commission': 'Επιτροπή:',
    'CommissionResp': 'Γνωμοδότηση Επιτροπής:',
    'RepeatBlock-By': 'Υποβλήθηκε Από:',
    'RepeatBlock-NoDocSe': 'Αριθμός Εγγράφου (Επαναλαμβανόμενο):',
    'Replacing': 'Αντικαθιστά:',
    'TablingGroups': 'Ομάδες Κατάθεσης:'
}


class GlossClean:
    """
    A unified class for cleaning documents in the GlossAPI pipeline.
    
    This class combines document cleaning, tag handling, and quality enhancement
    into a single component that can be optionally integrated into the pipeline.
    It also provides functionality for cleaning and exploring parquet files.
    """
    
    def __init__(self, 
                 log_level=logging.INFO, 
                 custom_tag_mapping=None, 
                 logger: Optional[logging.Logger] = None,
                 preserve_markdown_headings=True,
                 verify_tags=True,
                 preserve_code_examples=False):
        """
        Initialize the GlossClean class.
        
        Args:
            log_level: Logging level (default: logging.INFO)
            custom_tag_mapping: Optional dictionary mapping tag names to heading strings
                               If provided, this will override the default mapping
            logger (logging.Logger, optional): Logger instance
            preserve_markdown_headings: Whether to preserve markdown headings (default: True)
            verify_tags: Whether to verify and clean any remaining tags in the entire document
            preserve_code_examples: Whether to preserve HTML/XML tags in code examples (default: False)
        """
        # Setup logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logger or logging.getLogger(__name__)
        
        # Use custom mapping if provided, otherwise use default
        self.tag_mapping = custom_tag_mapping if custom_tag_mapping else DEFAULT_TAG_MAPPING
        
        # Format-preserving options
        self.preserve_markdown_headings = preserve_markdown_headings
        self.verify_tags = verify_tags
        self.preserve_code_examples = preserve_code_examples
        
        # Define common HTML/XML tags to clean
        self.common_tags = [
            'p', 'div', 'span', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'ul', 'ol', 'li', 'table', 'tr', 'td', 'th', 'thead', 'tbody',
            'a', 'img', 'br', 'hr', 'em', 'strong', 'b', 'i', 'u', 'sub', 'sup',
            'blockquote', 'code', 'pre', 'section', 'article', 'header', 'footer'
        ]
    
    def load_custom_mapping(self, mapping_file):
        """
        Load a custom tag mapping from a JSON file.
        
        Args:
            mapping_file (str): Path to the JSON file containing tag mappings
            
        Returns:
            dict: Dictionary mapping tag names to their descriptive labels
        """
        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                custom_mapping = json.load(f)
            self.tag_mapping = custom_mapping
            return custom_mapping
        except Exception as e:
            self.logger.error(f"Error loading custom mapping: {e}")
            return {}
    
    def extract_tags(self, text: str) -> List[str]:
        """
        Extract HTML/XML tags from text.
        
        Args:
            text (str): The text to extract tags from
            
        Returns:
            list: List of extracted tags
        """
        # Extract all tags using regex
        tag_pattern = r'</?([a-zA-Z0-9_:-]+)[^>]*?>'
        tags = re.findall(tag_pattern, text)
        return tags
    
    def enhance_text(self, text: str) -> str:
        """
        Enhance text quality by applying various cleaning steps.
        
        Args:
            text (str): Text to enhance
            
        Returns:
            str: Enhanced text
        """
        # Create a copy of the text to work with
        enhanced_text = text
        
        # 1. Decode HTML entities
        enhanced_text = html.unescape(enhanced_text)
        
        # 2. Normalize Unicode characters
        enhanced_text = unicodedata.normalize('NFKD', enhanced_text)
        
        # 3. Remove HTML comments
        enhanced_text = self._clean_html_comments(enhanced_text)
        
        # 4. Fix common issues in Greek text
        enhanced_text = self._fix_greek_text(enhanced_text)
        
        # 5. Normalize whitespace
        enhanced_text = self._normalize_whitespace(enhanced_text)
        
        return enhanced_text
    
    def _clean_html_comments(self, text: str) -> str:
        """
        Remove HTML comments from text.
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Text with HTML comments removed
        """
        # Pattern to match HTML comments
        comment_pattern = r'<!--.*?-->'
        
        # Remove HTML comments
        cleaned_text = re.sub(comment_pattern, '', text, flags=re.DOTALL)
        
        return cleaned_text
    
    def _fix_greek_text(self, text: str) -> str:
        """
        Fix common Unicode issues in Greek text.
        
        Args:
            text (str): The text to fix
            
        Returns:
            str: The fixed text
        """
        # Replace common problematic character sequences in Greek text
        replacements = {
            # Common OCR errors and encoding issues in Greek text
            'ηνπ': 'του',
            'εηα': 'εια',
            'ηαη': 'ται',
            'ηεο': 'της',
            'ηελ': 'την',
            'ηηθ': 'τικ',
            'ζηε': 'στη',
            'ζην': 'στο',
            'ηαζ': 'τασ',
            'πνπ': 'που',
            'ησλ': 'των',
            'καη': 'ματ',
            'ζεη': 'σει',
            'δηα': 'δια',
            'πξν': 'προ',
            'λαη': 'ναι',
            'ζεο': 'σης',
            'νπν': 'οπο',
            'ηεξ': 'τερ',
            'αηα': 'ατα',
            # Additional replacements for common issues in bad quality files
            '&lt;': '<',
            '&gt;': '>',
            '&amp;': '&',
            '&quot;': '"',
            '&apos;': "'",
            '&nbsp;': ' ',
        }
        
        # Apply replacements
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove all backslashes before underscores in the entire text
        text = re.sub(r'\\(_)', r'\1', text)
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace in text.
        
        Args:
            text (str): Text to normalize
            
        Returns:
            str: Text with normalized whitespace
        """
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove spaces at the beginning and end of lines
        text = re.sub(r'^ +| +$', '', text, flags=re.MULTILINE)
        
        # Ensure single newline after headings
        text = re.sub(r'(#+.*?)\n+', r'\1\n\n', text)
        
        return text
    
    def clean_text(self, text: str) -> str:
        """
        Clean HTML/XML tags from text while preserving formatting.
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Cleaned text
        """
        # First enhance the text quality
        enhanced_text = self.enhance_text(text)
        
        # If we need to preserve code examples, identify and protect them
        if self.preserve_code_examples:
            # Store code blocks to restore later
            code_blocks = {}
            
            # Find and replace markdown code blocks with placeholders
            markdown_code_pattern = r'```(?:\w+)?\n(.*?)```'
            markdown_code_matches = re.finditer(markdown_code_pattern, enhanced_text, re.DOTALL)
            for i, match in enumerate(markdown_code_matches):
                placeholder = f"__CODE_BLOCK_PLACEHOLDER_{i}__"
                code_blocks[placeholder] = match.group(0)
                enhanced_text = enhanced_text.replace(match.group(0), placeholder)
            
            # Find and replace HTML code blocks with placeholders
            html_code_pattern = r'<pre[^>]*>.*?</pre>|<code[^>]*>.*?</code>'
            html_code_matches = re.finditer(html_code_pattern, enhanced_text, re.DOTALL)
            for i, match in enumerate(html_code_matches):
                placeholder = f"__HTML_CODE_BLOCK_PLACEHOLDER_{i}__"
                code_blocks[placeholder] = match.group(0)
                enhanced_text = enhanced_text.replace(match.group(0), placeholder)
        
        # Extract tags for processing
        tags = self.extract_tags(enhanced_text)
        tag_counter = Counter(tags)
        
        # Process the text with our tag mappings
        cleaned_text = enhanced_text
        
        # Replace known tags with their Greek equivalents
        for tag, label in self.tag_mapping.items():
            # Create patterns for opening and closing tags
            open_pattern = f'<{tag}[^>]*?>'
            close_pattern = f'</{tag}>'
            
            # Replace opening tags with the label
            if self.preserve_markdown_headings:
                replacement = f'{label} '
            else:
                replacement = f'## {label}\n'
            
            cleaned_text = re.sub(open_pattern, replacement, cleaned_text)
            
            # Remove closing tags
            cleaned_text = re.sub(close_pattern, '', cleaned_text)
        
        # If verification is enabled, clean any remaining HTML/XML tags
        if self.verify_tags:
            # Clean remaining common HTML tags
            for tag in self.common_tags:
                open_pattern = f'<{tag}[^>]*?>'
                close_pattern = f'</{tag}>'
                
                cleaned_text = re.sub(open_pattern, '', cleaned_text)
                cleaned_text = re.sub(close_pattern, '', cleaned_text)
            
            # Clean any remaining tags with a generic pattern
            cleaned_text = re.sub(r'<[^>]+>', '', cleaned_text)
        
        # If we preserved code examples, restore them now
        if self.preserve_code_examples and code_blocks:
            for placeholder, code_block in code_blocks.items():
                cleaned_text = cleaned_text.replace(placeholder, code_block)
        
        return cleaned_text
    
    def process_file(self, input_file: Union[str, Path], output_file: Union[str, Path]) -> None:
        """
        Process a single markdown file.
        
        Args:
            input_file (Path): Path to the input file
            output_file (Path): Path to save the output file
        """
        try:
            # Read file
            with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Clean text
            cleaned_content = self.clean_text(content)
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Save cleaned file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            
            self.logger.debug(f"Processed {input_file} -> {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error processing {input_file}: {e}")
    
    def process_directory(self, input_dir: Union[str, Path], output_dir: Union[str, Path]) -> None:
        """
        Process all markdown files in a directory.
        
        Args:
            input_dir (Path): Directory containing input files
            output_dir (Path): Directory to save output files
        """
        # Convert to Path objects
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all markdown files
        input_files = list(input_dir.glob("**/*.md"))
        self.logger.info(f"Found {len(input_files)} markdown files to process")
        
        # Process each file
        for input_file in input_files:
            # Get relative path to preserve directory structure
            rel_path = input_file.relative_to(input_dir)
            output_file = output_dir / rel_path
            
            # Process file
            self.process_file(input_file, output_file)
        
        self.logger.info(f"Processed {len(input_files)} files from {input_dir} to {output_dir}")

    def clean_parquet_tags(self, parquet_file: Union[str, Path]) -> None:
        """
        Clean HTML/XML tags from a parquet file.
        
        Args:
            parquet_file (Union[str, Path]): Path to the parquet file
        """
        # Convert to Path object
        parquet_file = Path(parquet_file)
        
        # Create output file name
        output_file = parquet_file.parent / f"cleaned_{parquet_file.name}"
        
        # Log
        self.logger.info(f"Reading parquet file: {parquet_file}")
        
        try:
            # Read the parquet file
            df = pd.read_parquet(parquet_file)
            
            # Check if 'text' column exists, otherwise look for other text columns
            text_columns = []
            if 'text' in df.columns:
                text_columns.append('text')
            
            # Check for other potential text columns in GlossAPI output
            for col in ['header', 'section']:
                if col in df.columns:
                    text_columns.append(col)
            
            if not text_columns:
                self.logger.error(f"Parquet file does not have any text columns to clean")
                return
                
            # Clean each text column
            for col in text_columns:
                self.logger.info(f"Cleaning column: {col}")
                df[col] = df[col].apply(lambda x: self.clean_text(x) if isinstance(x, str) else x)
            
            # Save the cleaned parquet file
            df.to_parquet(output_file, index=False)
            self.logger.info(f"Saved cleaned parquet file to: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error cleaning parquet file: {e}")
    
    @staticmethod
    def explore_parquet(parquet_file: Union[str, Path], show_rows: int = 5) -> None:
        """
        Explore a parquet file and print its schema and contents.
        
        Args:
            parquet_file (Path): Path to the parquet file
            show_rows (int): Number of rows to show (default: 5)
        """
        try:
            # Read the parquet file schema
            table = pq.read_table(parquet_file)
            print("Schema:")
            print(table.schema)
            
            # Read the first few rows
            df = pd.read_parquet(parquet_file)
            print(f"\nFirst {show_rows} rows:")
            print(df.head(show_rows))
            
            # Get basic statistics
            print("\nDataFrame Info:")
            print(df.info())
            
            print("\nColumn names:")
            print(df.columns.tolist())
            
            print("\nNumber of rows:", len(df))
            
            # If there are specific columns of interest, examine their unique values
            if 'section_type' in df.columns:
                print("\nUnique section types:")
                print(df['section_type'].unique())
            
            if 'doc_id' in df.columns:
                print("\nNumber of unique documents:")
                print(df['doc_id'].nunique())
                
        except Exception as e:
            print(f"Error exploring parquet file: {e}")


# Standalone usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='GlossClean - Document Cleaning Module for GlossAPI')
    
    # Input/output options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input_dir', help='Directory containing input files')
    input_group.add_argument('--clean_parquet', help='Path to parquet file to clean')
    input_group.add_argument('--explore_parquet', help='Path to parquet file to explore')
    
    parser.add_argument('--output_dir', help='Directory to save outputs (optional, defaults to auto-generated)')
    parser.add_argument('--custom_mapping', help='Path to custom tag mapping JSON file')
    parser.add_argument('--no_verify_tags', action='store_false', dest='verify_tags',
                        help='Disable verification and cleaning of remaining tags')
    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                        help='Logging level (default: INFO)')
    parser.add_argument('--preserve_code_examples', action='store_true',
                        help='Preserve HTML/XML tags in code examples (default: False)')
    
    args = parser.parse_args()
    
    # Set log level
    log_level = getattr(logging, args.log_level)
    
    # Initialize GlossClean
    cleaner = GlossClean(log_level=log_level, verify_tags=args.verify_tags, preserve_code_examples=args.preserve_code_examples)
    
    # Load custom mapping if provided
    if args.custom_mapping:
        try:
            custom_mapping = cleaner.load_custom_mapping(args.custom_mapping)
            cleaner.tag_mapping = custom_mapping
        except Exception as e:
            print(f"Error loading custom mapping: {e}")
    
    # Handle parquet file exploration
    if args.explore_parquet:
        print(f"Exploring parquet file: {args.explore_parquet}")
        GlossClean.explore_parquet(args.explore_parquet)
        sys.exit(0)
    
    # Handle parquet file cleaning
    if args.clean_parquet:
        print(f"Cleaning parquet file: {args.clean_parquet}")
        cleaner.clean_parquet_tags(args.clean_parquet)
        sys.exit(0)
    
    # Handle directory processing
    if args.input_dir:
        input_dir = Path(args.input_dir)
        
        # Generate output directory if not specified
        output_dir = args.output_dir
        if not output_dir:
            # Create output directory at the same level as input directory
            parent_dir = input_dir.parent
            output_dir = parent_dir / "cleaned"
            print(f"Auto-generated output directory: {output_dir}")
        
        # Process directory
        cleaner.process_directory(input_dir, output_dir)
        print(f"Cleaned documents saved to: {output_dir}")
