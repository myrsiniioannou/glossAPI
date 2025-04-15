#!/usr/bin/env python3
"""
TagDetector - HTML/XML Tag Detection Tool for GlossAPI

This module provides a standalone tool for detecting HTML/XML tags in markdown files.
It scans a directory of files, identifies all tags, counts occurrences, and generates
comprehensive reports for analysis.

Usage:
    # As a standalone script
    python tag_detector.py --input_dir /path/to/markdown --output_dir /path/to/output [--known_mappings /path/to/mappings.json]
    
    # Or as an imported module
    detector = TagDetector()
    detector.analyze_directory(input_dir, output_dir)
"""

import os
import re
import json
import logging
import argparse
import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any, Union


class TagDetector:
    """
    A class for detecting and analyzing HTML/XML tags in markdown files.
    
    This class scans markdown files, identifies all HTML/XML tags, counts occurrences,
    and generates reports for analysis. It can compare detected tags against known
    mappings to identify new tags that may need translations.
    """
    
    def __init__(self, known_mappings_file: Optional[str] = None, log_level: int = logging.INFO):
        """
        Initialize the TagDetector.
        
        Args:
            known_mappings_file: Path to a JSON file containing known tag mappings (optional)
            log_level: Logging level (default: logging.INFO)
        """
        # Setup logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Regex patterns for tag detection
        self.opening_tag_pattern = re.compile(r'<([a-zA-Z][a-zA-Z0-9_:-]*)(?:\s+[^>]*)?>')
        self.closing_tag_pattern = re.compile(r'</([a-zA-Z][a-zA-Z0-9_:-]*)>')
        self.self_closing_tag_pattern = re.compile(r'<([a-zA-Z][a-zA-Z0-9_:-]*)(?:\s+[^>]*)?\s*/>')
        self.complete_tag_pattern = re.compile(r'<([a-zA-Z][a-zA-Z0-9_:-]*)(?:\s+[^>]*)?>.*?</\1>', re.DOTALL)
        
        # Load known mappings
        self.known_mappings = {}
        
        # Try to use provided mappings file, or fall back to default
        if known_mappings_file is None:
            # Look for default mappings file
            default_mappings_file = Path(__file__).parent.parent.parent / "data" / "tag_mappings" / "default_tag_mappings.json"
            if default_mappings_file.exists():
                known_mappings_file = str(default_mappings_file)
                self.logger.info(f"Using default tag mappings file: {default_mappings_file}")
            else:
                self.logger.warning(f"Default tag mappings file not found at {default_mappings_file}")
        
        # Load mappings if available
        if known_mappings_file:
            try:
                with open(known_mappings_file, 'r', encoding='utf-8') as f:
                    mappings_data = json.load(f)
                    self.known_mappings = mappings_data.get('known_tags', {})
                self.logger.info(f"Loaded {len(self.known_mappings)} known tag mappings from {known_mappings_file}")
            except Exception as e:
                self.logger.warning(f"Failed to load known mappings from {known_mappings_file}: {e}")
        
        # Initialize tag storage
        self.reset_analysis()
    
    def reset_analysis(self):
        """Reset the analysis results."""
        self.tag_data = defaultdict(lambda: {
            'count': 0,
            'files': set(),
            'examples': []
        })
        self.files_analyzed = 0
        self.files_with_tags = 0
        self.total_tags_found = 0
    
    def analyze_directory(self, input_dir: Union[str, Path], output_dir: Union[str, Path]) -> Dict[str, Any]:
        """
        Analyze all markdown files in a directory for HTML/XML tags.
        
        Args:
            input_dir: Directory containing markdown files to analyze
            output_dir: Directory to save analysis reports
            
        Returns:
            Dictionary containing analysis results
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # Reset analysis results
        self.reset_analysis()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create data directory structure if it doesn't exist
        tag_mappings_dir = output_dir / 'data' / 'tag_mappings' / 'dataset_specific'
        os.makedirs(tag_mappings_dir, exist_ok=True)
        
        self.logger.info(f"Analyzing markdown files in {input_dir}")
        
        # Get all markdown files
        markdown_files = list(input_dir.glob("**/*.md"))
        self.logger.info(f"Found {len(markdown_files)} markdown files to analyze")
        
        # Analyze each file
        for file_path in markdown_files:
            self.analyze_file(file_path)
        
        self.logger.info(f"Analysis complete. Found {self.total_tags_found} tags across {self.files_with_tags} files")
        
        # Generate reports
        self.generate_reports(output_dir)
        
        # Return analysis results
        return {
            'files_analyzed': self.files_analyzed,
            'files_with_tags': self.files_with_tags,
            'total_tags_found': self.total_tags_found,
            'unique_tags': len(self.tag_data)
        }
    
    def analyze_file(self, file_path: Path) -> Dict[str, int]:
        """
        Analyze a single markdown file for HTML/XML tags.
        
        Args:
            file_path: Path to the markdown file
            
        Returns:
            Dictionary mapping tag names to counts in this file
        """
        tag_counts = defaultdict(int)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.files_analyzed += 1
            
            # Find all opening tags
            for match in self.opening_tag_pattern.finditer(content):
                tag_name = match.group(1)
                tag_counts[tag_name] += 1
            
            # Find all closing tags
            for match in self.closing_tag_pattern.finditer(content):
                tag_name = match.group(1)
                tag_counts[tag_name] += 1
            
            # Find all self-closing tags
            for match in self.self_closing_tag_pattern.finditer(content):
                tag_name = match.group(1)
                tag_counts[tag_name] += 1
            
            # Extract examples of complete tags
            for match in self.complete_tag_pattern.finditer(content):
                tag_name = match.group(1)
                example = match.group(0)
                
                # Truncate very long examples
                if len(example) > 100:
                    example = example[:97] + "..."
                
                # Update tag data
                self.tag_data[tag_name]['count'] += 1
                self.tag_data[tag_name]['files'].add(str(file_path))
                
                # Store up to 3 examples per tag
                if len(self.tag_data[tag_name]['examples']) < 3:
                    self.tag_data[tag_name]['examples'].append(example)
            
            # Update file count if tags were found
            if tag_counts:
                self.files_with_tags += 1
            
            # Update total tag count
            file_tag_count = sum(tag_counts.values())
            self.total_tags_found += file_tag_count
            
            return dict(tag_counts)
            
        except Exception as e:
            self.logger.error(f"Error analyzing file {file_path}: {e}")
            return {}
    
    def generate_reports(self, output_dir: Path) -> None:
        """
        Generate analysis reports.
        
        Args:
            output_dir: Directory to save reports
        """
        # Prepare data for reports
        analysis_data = {
            'metadata': {
                'timestamp': datetime.datetime.now().isoformat(),
                'files_analyzed': self.files_analyzed,
                'files_with_tags': self.files_with_tags,
                'total_tags_found': self.total_tags_found,
                'unique_tags': len(self.tag_data)
            },
            'tags': {}
        }
        
        # Convert sets to lists for JSON serialization
        for tag_name, data in self.tag_data.items():
            analysis_data['tags'][tag_name] = {
                'count': data['count'],
                'files': len(data['files']),
                'examples': data['examples']
            }
        
        # Generate complete analysis results
        analysis_file = output_dir / 'tag_analysis_results.json'
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved complete analysis results to {analysis_file}")
        
        # Generate new tags file (for manual editing)
        new_tags = {
            'metadata': {
                'timestamp': datetime.datetime.now().isoformat(),
                'dataset': output_dir.name,
                'known_tags_used': len(self.known_mappings),
                'new_tags_found': 0
            },
            'new_tags': {}
        }
        
        # Identify new tags
        for tag_name, data in self.tag_data.items():
            if tag_name not in self.known_mappings:
                new_tags['new_tags'][tag_name] = {
                    'count': data['count'],
                    'files': len(data['files']),
                    'examples': data['examples'],
                    'translation': '',  # Empty, to be filled by human
                    'preserve_content': True  # Default suggestion
                }
        
        new_tags['metadata']['new_tags_found'] = len(new_tags['new_tags'])
        
        # Save new tags file
        new_tags_file = output_dir / 'data' / 'tag_mappings' / 'dataset_specific' / f"{output_dir.name}_tags.json"
        with open(new_tags_file, 'w', encoding='utf-8') as f:
            json.dump(new_tags, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved {len(new_tags['new_tags'])} new tags to {new_tags_file}")
        
        # Generate human-readable report
        self._generate_markdown_report(output_dir)
    
    def _generate_markdown_report(self, output_dir: Path) -> None:
        """
        Generate a human-readable markdown report.
        
        Args:
            output_dir: Directory to save the report
        """
        report_file = output_dir / 'tag_analysis_report.md'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            # Report header
            f.write("# Tag Analysis Report\n\n")
            f.write(f"*Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            
            # Overall statistics
            f.write("## Overall Statistics\n\n")
            f.write(f"- **Files Analyzed**: {self.files_analyzed}\n")
            f.write(f"- **Files Containing Tags**: {self.files_with_tags} ({self.files_with_tags/self.files_analyzed*100:.1f}% of total)\n")
            f.write(f"- **Total Tags Found**: {self.total_tags_found}\n")
            f.write(f"- **Unique Tag Types**: {len(self.tag_data)}\n")
            f.write(f"- **Known Tags**: {len(self.known_mappings)}\n")
            f.write(f"- **New Tags**: {len(self.tag_data) - len(self.known_mappings)}\n\n")
            
            # Tag frequency table
            f.write("## Tag Frequency\n\n")
            f.write("| Tag Name | Count | Files | Known | Examples |\n")
            f.write("|----------|-------|-------|-------|----------|\n")
            
            # Sort tags by frequency
            sorted_tags = sorted(self.tag_data.items(), key=lambda x: x[1]['count'], reverse=True)
            
            for tag_name, data in sorted_tags:
                is_known = "Yes" if tag_name in self.known_mappings else "No"
                examples = data['examples'][0] if data['examples'] else ""
                
                # Truncate long examples
                if len(examples) > 40:
                    examples = examples[:37] + "..."
                
                f.write(f"| `{tag_name}` | {data['count']} | {len(data['files'])} | {is_known} | `{examples}` |\n")
            
            # New tags section
            new_tags = [tag for tag in sorted_tags if tag[0] not in self.known_mappings]
            if new_tags:
                f.write("\n## New Tags Requiring Translation\n\n")
                f.write("The following tags were found but don't have translations in the known mappings:\n\n")
                
                for tag_name, data in new_tags:
                    f.write(f"### `{tag_name}`\n\n")
                    f.write(f"- **Occurrences**: {data['count']} (in {len(data['files'])} files)\n")
                    f.write("- **Examples**:\n")
                    
                    for example in data['examples']:
                        f.write(f"  - `{example}`\n")
                    
                    f.write("\n")
                
                f.write("\n## Next Steps\n\n")
                f.write("1. Review the new tags in the generated JSON file\n")
                f.write("2. Add translations for each tag\n")
                f.write("3. Decide which tags should preserve their content\n")
                f.write(f"4. Save your changes to the JSON file at: `{output_dir}/data/tag_mappings/dataset_specific/{output_dir.name}_tags.json`\n")
        
        self.logger.info(f"Generated human-readable report at {report_file}")


def main():
    """Run the tag detector as a standalone script."""
    parser = argparse.ArgumentParser(description='Detect and analyze HTML/XML tags in markdown files')
    parser.add_argument('--input_dir', required=True, help='Directory containing markdown files to analyze')
    parser.add_argument('--output_dir', required=True, help='Directory to save analysis reports')
    parser.add_argument('--known_mappings', help='Path to a JSON file containing known tag mappings')
    parser.add_argument('--log_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                        help='Logging level')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    
    # Run analysis
    detector = TagDetector(known_mappings_file=args.known_mappings, log_level=log_level)
    results = detector.analyze_directory(args.input_dir, args.output_dir)
    
    print(f"\nAnalysis complete!")
    print(f"Files analyzed: {results['files_analyzed']}")
    print(f"Files with tags: {results['files_with_tags']}")
    print(f"Total tags found: {results['total_tags_found']}")
    print(f"Unique tag types: {results['unique_tags']}")
    print(f"\nReports saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
