import logging
from pathlib import Path
import os
import pandas as pd
import random
from typing import Dict, Optional, Union, List, Any
import shutil

from .gloss_extract import GlossExtract
from .gloss_section import GlossSection
from .gloss_section_classifier import GlossSectionClassifier
from .gloss_downloader import GlossDownloader
from .gloss_preprocess import GlossPreprocess

class Corpus:
    """
    A high-level wrapper for the GlossAPI academic document processing pipeline.
    
    This class provides a unified interface to extract PDFs to markdown,
    extract sections, and classify them using machine learning.
    
    Example:
        corpus = Corpus(input_dir="path/to/pdfs", output_dir="path/to/output")
        corpus.extract()  # Extract PDFs to markdown
        corpus.section()  # Extract sections from markdown files
        corpus.annotate()  # Classify sections using ML
    """
    
    def __init__(
        self, 
        input_dir: Union[str, Path], 
        output_dir: Union[str, Path],
        section_classifier_model_path: Optional[Union[str, Path]] = None,
        extraction_model_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
        annotation_mapping: Optional[Dict[str, str]] = None,
        downloader_config: Optional[Dict[str, Any]] = None,
        log_level: int = logging.INFO
    ):
        """
        Initialize the Corpus processor.
        
        Args:
            input_dir: Directory containing input files (PDF or markdown)
            output_dir: Base directory for all outputs
            section_classifier_model_path: Path to the pre-trained section classifier model
            extraction_model_path: Path to the pre-trained kmeans clustering model for extraction
            metadata_path: Path to metadata file with document types (optional)
            annotation_mapping: Dictionary mapping document types to annotation methods (optional)
                               e.g. {'Κεφάλαιο': 'chapter'} means documents with type 'Κεφάλαιο' use chapter annotation
            downloader_config: Configuration parameters for the GlossDownloader (optional)
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
        self.output_dir = Path(output_dir)
        
        # Package directory for default models
        package_dir = Path(__file__).parent
        
        # Handle section classifier model path
        if section_classifier_model_path:
            self.section_classifier_model_path = Path(section_classifier_model_path)
        else:
            # Use default model path in the package
            self.section_classifier_model_path = package_dir / "models" / "section_classifier.joblib"
        
        # Handle extraction model path
        if extraction_model_path:
            self.extraction_model_path = Path(extraction_model_path)
        else:
            # Use default model path in the package
            self.extraction_model_path = package_dir / "models" / "kmeans_weights.joblib"
            
        self.metadata_path = Path(metadata_path) if metadata_path else None
        
        # Store annotation mapping - default is to treat 'Κεφάλαιο' as chapter
        self.annotation_mapping = annotation_mapping or {'Κεφάλαιο': 'chapter'}
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize component classes
        self.extractor = GlossExtract()
        self.sectioner = GlossSection()
        self.classifier = GlossSectionClassifier()
        
        # Create necessary directories
        self.markdown_dir = self.output_dir / "markdown"
        self.sections_dir = self.output_dir / "sections"
        # Define models_dir path but don't create the directory yet - only create it when needed
        self.models_dir = self.output_dir / "models"
        
        os.makedirs(self.markdown_dir, exist_ok=True)
        os.makedirs(self.sections_dir, exist_ok=True)
        
        # Setup output files
        self.sections_parquet = self.sections_dir / "sections_for_annotation.parquet"
        self.classified_parquet = self.output_dir / "classified_sections.parquet"
        self.fully_annotated_parquet = self.output_dir / "fully_annotated_sections.parquet"
        
        # Initialize document type mapping
        self.filename_to_doctype = {}
        
        # Initialize downloader config
        self.downloader_config = downloader_config or {}
        
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load metadata file if provided and extract document type mapping."""
        if self.metadata_path and self.metadata_path.exists():
            try:
                self.logger.info(f"Loading metadata from {self.metadata_path}")
                metadata_df = pd.read_parquet(self.metadata_path)
                
                # Debug information
                self.logger.info(f"Metadata file has {len(metadata_df)} rows and columns: {metadata_df.columns.tolist()}")
                self.logger.info(f"Sample filenames: {metadata_df['filename'].head(3).tolist()}")
                self.logger.info(f"Sample document types: {metadata_df['document_type'].head(3).tolist()}")
                
                # Create a mapping from filename to document_type
                if 'filename' in metadata_df.columns and 'document_type' in metadata_df.columns:
                    self.logger.info("Both 'filename' and 'document_type' columns found in metadata")
                    
                    # Check if filenames have extensions
                    sample_filenames = metadata_df['filename'].head(100).tolist()
                    if any('.' in str(f) for f in sample_filenames):
                        self.logger.warning("Some filenames in metadata contain extensions. This may cause matching issues.")
                        self.logger.warning("Will attempt to match filenames both with and without extensions.")
                        
                        # Create a mapping that works with or without extensions
                        self.filename_to_doctype = {}
                        
                        for idx, row in metadata_df.iterrows():
                            filename = row['filename']
                            doctype = row['document_type']
                            
                            # Add the original filename
                            self.filename_to_doctype[filename] = doctype
                            
                            # Add filename without extension
                            if '.' in filename:
                                base_filename = filename.rsplit('.', 1)[0]
                                self.filename_to_doctype[base_filename] = doctype
                            
                            # Add filename with .md extension
                            if not filename.endswith('.md'):
                                md_filename = f"{filename}.md"
                                self.filename_to_doctype[md_filename] = doctype
                    else:
                        # Simple dictionary mapping without extension handling
                        self.filename_to_doctype = dict(zip(
                            metadata_df['filename'], 
                            metadata_df['document_type']
                        ))
                    
                    self.logger.info(f"Loaded {len(self.filename_to_doctype)} filename-to-doctype mappings")
                else:
                    self.logger.warning("Metadata file does not contain 'filename' or 'document_type' columns")
            except Exception as e:
                self.logger.error(f"Error loading metadata: {e}")
        else:
            if self.metadata_path:
                self.logger.warning(f"Metadata file not found: {self.metadata_path}")
    
    def filter(self, input_dir: Union[str, Path] = None, split_bad: bool = True, model_path: Optional[Union[str, Path]] = None, reports_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Filter markdown files based on quality to separate good from bad quality.
        
        Args:
            input_dir: Directory containing markdown files to filter (defaults to self.markdown_dir)
            split_bad: Whether to separate files into good/bad quality or keep all as good
            model_path: Path to the pre-trained model for clustering (defaults to self.extraction_model_path)
            reports_dir: Directory to save reports (optional)
        """
        # Handle default parameters
        if input_dir is None:
            input_dir = self.markdown_dir
            
        # Skip directory creation if not splitting bad files
        if not split_bad:
            self.logger.info("Skipping quality clustering as split_bad=False")
            # Just use the original markdown directory for good files
            self.good_markdown_dir = input_dir
            self.markdown_dir = input_dir
            self.logger.info(f"Using all files from {input_dir} as good quality")
            return
            
        # Only create directory structure if we're doing actual clustering
        quality_dir = self.output_dir / 'quality_clustering'
        os.makedirs(quality_dir, exist_ok=True)
        
        good_dir = quality_dir / 'good'
        bad_dir = quality_dir / 'bad'
        os.makedirs(good_dir, exist_ok=True)
        os.makedirs(bad_dir, exist_ok=True)
        
        if split_bad:
            # Only create bad dir if we're going to use it
            os.makedirs(bad_dir, exist_ok=True)
            
            # Set extraction model path
            if model_path is None:
                model_path = str(self.extraction_model_path)
            
            # Check if model exists
            if not os.path.exists(model_path):
                self.logger.warning(f"Clustering model not found at {model_path}. Training a new model...")
                # Create models directory only when needed for training a new model
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                # Train model
                self.extractor.training(str(input_dir), model_path=model_path)
                self.logger.info(f"Model trained and saved to {model_path}")
            
            # Set up report path if provided
            report_path = None
            if reports_dir is not None:
                reports_dir = Path(reports_dir)
                os.makedirs(reports_dir, exist_ok=True)
                report_path = reports_dir / "quality_clustering_report.md"
            
            # Run split_bad to separate good and bad files
            self.logger.info("Running clustering to separate good and bad quality documents...")
            self.extractor.split_bad(
                input_folder=str(input_dir),
                output_folder=str(quality_dir),
                model_file=model_path,
                report_path=report_path
            )
            self.logger.info(f"Clustering complete. Files sorted into good/bad folders.")
        else:
            # If split_bad is disabled, just copy all files to the good directory
            self.logger.info("Clustering disabled. Copying all files to 'good' folder...")
            
            # Get all markdown files
            markdown_files = list(Path(input_dir).glob("*.md"))
            
            # Copy all files to good folder
            copied_count = 0
            for source_path in markdown_files:
                filename = source_path.name
                dest_path = good_dir / filename
                try:
                    shutil.copy2(source_path, dest_path)
                    copied_count += 1
                except Exception as e:
                    self.logger.error(f"Error copying {filename}: {e}")
            
            self.logger.info(f"Copied {copied_count} files to good folder")
        
        # Update markdown_dir to use good files for further processing
        self.good_markdown_dir = good_dir
        self.logger.info(f"Files processed. Good files saved to {self.good_markdown_dir}")
        
        # For subsequent operations, use the good files
        self.markdown_dir = self.good_markdown_dir

    def preprocess(self) -> None:
        """
        Preprocess input files to identify problematic files before extraction.
        
        This method analyzes files in the input directory, collects metadata,
        and identifies potentially problematic files based on specific criteria:
        - File extension issues (missing, multiple, or unusual extensions)
        - Suspiciously small file sizes
        - PDF-specific issues (font embedding, subsetting problems)
        
        The problematic files are excluded from further processing.
        """
        self.logger.info(f"Preprocessing files in {self.input_dir}...")
        
        # Create a centralized reports directory
        reports_dir = Path(self.input_dir).parent.parent / "reports"
        preprocess_reports_dir = reports_dir / "preprocessing"
        os.makedirs(preprocess_reports_dir, exist_ok=True)
        
        # Initialize the preprocessor with the preprocessing reports directory
        preprocessor = GlossPreprocess(str(self.input_dir), str(preprocess_reports_dir))
        
        # Analyze files and identify problematic ones
        preprocessor.analyze_files()
        
        # Generate reports
        preprocessor.generate_reports()
        
        # Get clean files (non-problematic)
        clean_files = preprocessor.get_clean_files()
        
        self.logger.info(f"Preprocessing complete. Found {len(clean_files)} clean files out of {len(preprocessor.file_data)} total files.")
        self.logger.info(f"Identified {len(preprocessor.problematic_files)} problematic files that will be excluded from processing.")
        self.logger.info(f"Preprocessing reports saved to {preprocess_reports_dir}")
        
        # Store clean files for later use in the extraction process
        self.clean_files = clean_files

    def extract(
        self, 
        input_format: str = "all", 
        num_threads: int = 4, 
        accel_type: str = "Auto",
        split_bad: bool = True,
        model_path: str = None,
        run_filter: bool = False
    ) -> None:
        """
        Extract input files to markdown format.
        
        Args:
            input_format: Input format ("pdf", "docx", "xml_jats", "html", "pptx", "csv", "md", "all") (default: "all")
                          Note: Old .doc format (pre-2007) is not supported
            num_threads: Number of threads for processing (default: 4)
            accel_type: Acceleration type ("Auto", "CPU", "CUDA", "MPS") (default: "Auto")
            split_bad: Whether to perform clustering to separate good and bad files (default: True)
            model_path: Path to the KMeans clustering model (default: None, will use default path)
            run_filter: Whether to automatically run filtering after extraction (default: False)
        """
        self.logger.info(f"Extracting {input_format} files to markdown...")
        
        # Create extraction reports directory
        reports_dir = Path(self.input_dir).parent.parent / "reports"
        extraction_reports_dir = reports_dir / "extraction"
        os.makedirs(extraction_reports_dir, exist_ok=True)
        
        # Prepare extractor
        self.extractor.enable_accel(threads=num_threads, type=accel_type)
        self.extractor.create_extractor()
        
        # Create output directory
        os.makedirs(self.markdown_dir, exist_ok=True)
        
        # Define supported formats
        supported_formats = ["pdf", "docx", "xml", "html", "pptx", "csv", "md"]
        
        # Look for the downloads directory first
        downloads_dir = self.output_dir / "downloads"
        
        # If downloads directory doesn't exist or is empty, check input directory and move files
        if not downloads_dir.exists() or not any(downloads_dir.iterdir()):
            self.logger.info(f"Downloads directory not found or empty at {downloads_dir}, checking input directory...")
            
            # Create downloads directory if it doesn't exist
            os.makedirs(downloads_dir, exist_ok=True)
            
            # Check input directory for supported files and move them
            input_files_to_move = []
            for ext in supported_formats:
                found_files = list(self.input_dir.glob(f"*.{ext}"))
                if found_files:
                    self.logger.info(f"Found {len(found_files)} .{ext} files in input directory, moving to downloads...")
                    input_files_to_move.extend(found_files)
            
            # Move files to downloads directory
            for file_path in input_files_to_move:
                target_path = downloads_dir / file_path.name
                if not target_path.exists():
                    shutil.copy2(file_path, target_path)
                    self.logger.debug(f"Copied {file_path.name} to downloads directory")
            
            self.logger.info(f"Moved {len(input_files_to_move)} files to downloads directory")
        
        # Get input files from downloads directory
        if input_format.lower() == "all":
            # Include all supported formats
            input_files = []
            for ext in supported_formats:
                found_files = list(downloads_dir.glob(f"*.{ext}"))
                input_files.extend(found_files)
                if found_files:
                    self.logger.info(f"Found {len(found_files)} .{ext} files in downloads directory")
            
            # Log a warning about doc files
            doc_files = list(downloads_dir.glob("*.doc"))
            if doc_files:
                self.logger.warning(f"Found {len(doc_files)} .doc files which are not supported by Docling (pre-2007 Word format)")
        elif "," in input_format.lower():
            # Handle comma-separated format list
            input_files = []
            formats = [fmt.strip().lower() for fmt in input_format.split(",")]
            for ext in formats:
                # Handle special case for XML formats
                if ext == "xml_jats":
                    ext = "xml"  # Use the file extension .xml
                    
                if ext == "doc":
                    self.logger.warning(f"The .doc format (pre-2007 Word) is not supported by Docling. Please convert to .docx first.")
                    continue
                    
                current_files = list(downloads_dir.glob(f"*.{ext}"))
                self.logger.info(f"Found {len(current_files)} files with extension .{ext}")
                input_files.extend(current_files)
        else:
            # Handle special case for XML formats
            if input_format.lower() == "xml":
                ext = "xml"  # Still use the file extension .xml
            else:
                ext = input_format.lower()
                
            if ext == "doc":
                self.logger.error(f"The .doc format (pre-2007 Word) is not supported by Docling. Please convert to .docx first.")
                return
                
            input_files = list(downloads_dir.glob(f"*.{ext}"))
        
        if not input_files:
            self.logger.warning(f"No {input_format} files found in {downloads_dir}")
            return
        
        self.logger.info(f"Found {len(input_files)} files to extract")
        
        # Filter out problematic files if preprocessing was done
        if hasattr(self, 'clean_files') and self.clean_files:
            # Convert paths to strings for comparison
            clean_files_str = set(str(Path(f).resolve()) for f in self.clean_files)
            
            # Filter input_files to only include clean files
            filtered_files = [f for f in input_files if str(f.resolve()) in clean_files_str]
            
            skipped_count = len(input_files) - len(filtered_files)
            if skipped_count > 0:
                self.logger.info(f"Skipping {skipped_count} problematic files identified during preprocessing")
                input_files = filtered_files
        
        # Process all files
        self.logger.info(f"Processing {len(input_files)} files...")
        
        # Extract files to markdown
        os.makedirs(self.markdown_dir, exist_ok=True)
        
        # Use multiple threads for extraction
        self.extractor.extract_path(input_files, self.markdown_dir)
        
        self.logger.info(f"Extraction complete. Markdown files saved to {self.markdown_dir}")
        
        # Run filtering on extracted markdown files if requested
        if run_filter:
            self.filter(input_dir=self.markdown_dir, split_bad=split_bad, model_path=model_path, reports_dir=extraction_reports_dir)
    
    def split_bad(self, model_path: Optional[Union[str, Path]] = None) -> None:
        """
        Analyze markdown files for extraction quality and update the input parquet file.
        This adds an 'extraction' column to the parquet with values 'good' or 'bad'.
        
        Unlike the filter() method, this doesn't create separate folders but updates the 
        parquet file directly for more efficient pipeline processing.
        
        Args:
            model_path: Path to the pre-trained model for clustering (defaults to self.extraction_model_path)
        """
        self.logger.info("Analyzing extraction quality and updating parquet file...")
        
        # Set extraction model path
        if model_path is None:
            model_path = str(self.extraction_model_path)
        
        # Check if model exists
        if not os.path.exists(model_path):
            self.logger.warning(f"Clustering model not found at {model_path}. Training a new model...")
            # Create models directory only when needed for training a new model
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            # Train model
            self.extractor.training(str(self.markdown_dir), model_path=model_path)
            self.logger.info(f"Model trained and saved to {model_path}")
        
        # Run the parquet annotation
        self.logger.info("Annotating input parquet with extraction quality information...")
        success = self.extractor.annotate_parquet_with_extraction_quality(
            markdown_folder=str(self.markdown_dir),
            input_dir=str(self.input_dir),
            model_file=model_path
        )
        
        if success:
            self.logger.info("Parquet file successfully updated with extraction quality information.")
        else:
            self.logger.warning("Failed to update parquet file with extraction quality information.")
    
    def section(self) -> None:
        """
        Extract sections from markdown files and save to Parquet format.
        
        Uses files marked with 'good' extraction quality (if available) or all markdown files.
        """
        self.logger.info("Extracting sections from markdown files...")
        
        # Create output directory
        os.makedirs(self.sections_dir, exist_ok=True)
        
        # Filter markdown files based on extraction quality in parquet files
        # Initialize the good_filenames list that will be used with the sectioner
        good_filenames = []
        
        # Try to find files marked as 'good' in the parquet
        from glossapi.parquet_schema import ParquetSchema
        # Initialize with proper URL column configuration
        parquet_schema = ParquetSchema({
            'url_column': 'preferred_url'  # Use the default URL column
        })
        self.logger.info(f"Using URL column for parquet search: {parquet_schema.url_column}")
        
        # Look for input parquet with extraction column
        input_parquet_path = parquet_schema.find_metadata_parquet(self.input_dir)
        
        # If not in input_dir, check download_results folder
        if input_parquet_path is None:
            download_results_dir = self.input_dir / "download_results"
            if download_results_dir.exists():
                input_parquet_path = parquet_schema.find_metadata_parquet(download_results_dir)
            
        if input_parquet_path is not None:
            try:
                # Load parquet and filter by 'good' extraction
                df = pd.read_parquet(input_parquet_path)
                if 'filename' in df.columns and 'extraction' in df.columns:
                    good_rows = df[df['extraction'] == 'good']
                    if not good_rows.empty:
                        # Get filenames (without extension) of good extractions
                        good_filenames = [
                            os.path.splitext(filename)[0] 
                            for filename in good_rows['filename'].tolist() 
                            if filename
                        ]
                        self.logger.info(f"Found {len(good_filenames)} files marked as 'good' in parquet")
                        
                        # Update the processing_stage in the download results parquet
                        try:
                            # Update processing_stage for all good rows
                            if 'processing_stage' in df.columns:
                                # Only update rows where extraction is 'good'
                                for idx in good_rows.index:
                                    current_stage = df.loc[idx, 'processing_stage']
                                    # Append section to stages if not already there
                                    if current_stage is not None and 'section' not in str(current_stage):
                                        df.loc[idx, 'processing_stage'] = current_stage + ',section'
                            else:
                                # Create processing_stage column if it doesn't exist
                                df['processing_stage'] = None
                                for idx in good_rows.index:
                                    df.loc[idx, 'processing_stage'] = 'download,extract,section'
                            
                            standard_path = Path(os.path.dirname(input_parquet_path)) / "download_results.parquet"
                            df.to_parquet(standard_path, index=False)
                            self.logger.info(f"Updated processing_stage column in {standard_path} for good quality files")
                            
                            # If we renamed the file, log this
                            if standard_path != input_parquet_path:
                                self.logger.info(f"Standardized parquet name from {os.path.basename(input_parquet_path)} to download_results.parquet")
                        except Exception as e:
                            self.logger.warning(f"Error updating processing_stage in download results parquet: {e}")
            except Exception as e:
                self.logger.warning(f"Error reading parquet for extraction quality: {e}")
        
        # Check if we found any good files to process
        self.logger.info(f"Found {len(good_filenames)} good quality files for sectioning")
        if good_filenames:
            self.logger.info(f"Good filenames: {good_filenames}")
            
        if not good_filenames:
            error_msg = "No good quality files found for sectioning. Check extraction quality or run split_bad() first."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Extract sections - pass list of good filenames to the sectioner
        # We will pass the original markdown directory and the list of good filenames 
        # rather than creating a separate directory
        self.sectioner.to_parquet(
            input_dir=str(self.markdown_dir),  # Use the markdown directory directly
            output_dir=str(self.sections_dir),
            filenames_to_process=good_filenames  # Pass the list of good filenames
        )
        
        self.logger.info(f"Finished sectioning {len(good_filenames)} good quality files")
        self.logger.info(f"Section extraction complete. Parquet file saved to {self.sections_parquet}")
    

    def annotate(self, annotation_type: str = "text", fully_annotate: bool = True) -> None:
        """
        Annotate extracted sections with classification information.
        
        Args:
            annotation_type: Type of annotation to use: 'text' or 'chapter'
                           - 'text': Use text-based annotation with section titles (default)
                           - 'chapter': Use chapter-based annotation with chapter numbers
            fully_annotate: Whether to perform full annotation of sections (default: True)
        """
        self.logger.info("Running section classification...")
        
        # Check if input parquet file exists
        if not self.sections_parquet.exists():
            self.logger.error(f"Sections file not found: {self.sections_parquet}. Please run section() first.")
            return
        
        # Check if section classifier model exists
        model_exists = self.section_classifier_model_path.exists()
        if not model_exists:
            self.logger.warning(f"Model file not found at {self.section_classifier_model_path}. To train a new model, run GlossSectionClassifier.train_from_csv()")
        
        # Use section classifier model path
        model_path = str(self.section_classifier_model_path) if model_exists else None
        
        # Classify sections and save output to 'classified_sections.parquet'
        self.classifier.classify_sections(
            input_parquet=str(self.sections_parquet),
            output_parquet=str(self.classified_parquet),
            model_path=model_path,
            n_cpus=4,
            column_name='title'
        )
        
        # Perform full annotation if requested
        if fully_annotate:
            self.logger.info("Performing full annotation...")
            
            # If we're using auto annotation and have document types and annotation mappings available
            if annotation_type == "auto" and self.filename_to_doctype and self.annotation_mapping:
                # Create a mapping from filename to annotation type based on document types
                filename_to_annotation = {}
                for filename, doc_type in self.filename_to_doctype.items():
                    # Look up the annotation method for this document type in our mapping
                    # Default to 'text' if no mapping exists
                    filename_to_annotation[filename] = self.annotation_mapping.get(doc_type, 'text')
                
                self.logger.info(f"Using document-type specific annotation based on metadata")
                
                # Read the classified parquet file
                df = pd.read_parquet(str(self.classified_parquet))
                
                # Group by filename and process each document according to its annotation type
                updated_groups = []
                
                for filename, group in df.groupby('filename'):
                    # Determine annotation type for this file
                    doc_annotation = filename_to_annotation.get(filename, 'text')
                    
                    # Process according to annotation type
                    if doc_annotation == 'chapter':
                        self.logger.debug(f"Processing {filename} as chapter")
                        updated_group = self.classifier.fully_annotate_chapter_group(group)
                    else:
                        self.logger.debug(f"Processing {filename} as text")
                        updated_group = self.classifier.fully_annotate_text_group(group)
                    
                    if updated_group is not None:
                        updated_groups.append(updated_group)
                
                # Concatenate and save results
                if updated_groups:
                    df_updated = pd.concat(updated_groups)
                    df_updated.to_parquet(str(self.fully_annotated_parquet), index=False)
                else:
                    self.logger.warning("No valid document groups to process. Output file not created.")
            else:
                # Use the standard fully_annotate method with the specified annotation type
                self.classifier.fully_annotate(
                    input_parquet=str(self.classified_parquet),
                    output_parquet=str(self.fully_annotated_parquet),
                    document_types=self.filename_to_doctype if self.filename_to_doctype else None,
                    annotation_type=annotation_type
                )
            
            # Use the fully annotated output for adding document types
            self._add_document_types(self.fully_annotated_parquet)
            
            # Update processing_stage in the fully annotated parquet
            try:
                # Read the fully annotated parquet
                df = pd.read_parquet(self.fully_annotated_parquet)
                
                # Add annotate to processing stage
                if 'processing_stage' in df.columns:
                    df['processing_stage'] = df['processing_stage'].apply(lambda x: x + ',annotate' if 'annotate' not in str(x) else x)
                else:
                    df['processing_stage'] = 'section,annotate'
                    
                # Write back
                df.to_parquet(self.fully_annotated_parquet, index=False)
                self.logger.info("Updated processing_stage to include 'annotate' stage")
            except Exception as e:
                self.logger.warning(f"Failed to update processing_stage in fully annotated parquet: {e}")
        else:
            # Add document types to the classified output
            self._add_document_types(self.classified_parquet)
            
            # Update processing_stage in the classified parquet when not doing full annotation
            try:
                # Read the classified parquet
                df = pd.read_parquet(self.classified_parquet)
                
                # Add annotate to processing stage
                if 'processing_stage' in df.columns:
                    df['processing_stage'] = df['processing_stage'].apply(lambda x: x + ',annotate' if 'annotate' not in str(x) else x)
                else:
                    df['processing_stage'] = 'section,annotate'
                    
                # Write back
                df.to_parquet(self.classified_parquet, index=False)
                self.logger.info("Updated processing_stage to include 'annotate' stage")
            except Exception as e:
                self.logger.warning(f"Failed to update processing_stage in classified parquet: {e}")
    
    def _add_document_types(self, parquet_file: Path) -> None:
        """
        Add document_type information to the classified sections.
        
        Args:
            parquet_file: Path to the Parquet file to update
        """
        if not self.filename_to_doctype:
            self.logger.warning("No document type information available. Skipping document type addition.")
            return
        
        if parquet_file.exists():
            try:
                # Read the parquet file
                df = pd.read_parquet(parquet_file)
                
                # Add document_type based on filename
                df['document_type'] = df['filename'].map(self.filename_to_doctype)
                
                # Check for missing document types
                missing_count = df['document_type'].isna().sum()
                if missing_count > 0:
                    self.logger.warning(f"{missing_count} sections ({missing_count/len(df):.2%}) have no document type!")
                    missing_filenames = df[df['document_type'].isna()]['filename'].unique()[:5]
                    self.logger.warning(f"Sample filenames with missing document types: {missing_filenames}")
                    
                    # Check if the issue might be due to .md extension
                    if any('.md' in str(f) for f in self.filename_to_doctype.keys()):
                        self.logger.warning("Possible cause: Metadata filenames contain .md extension but sections filenames don't")
                    elif any('.md' in str(f) for f in df['filename'].unique()[:100]):
                        self.logger.warning("Possible cause: Sections filenames contain .md extension but metadata filenames don't")
                
                # Save the updated file
                df.to_parquet(parquet_file, index=False)
                self.logger.info(f"Added document types to {parquet_file}")
            except Exception as e:
                self.logger.error(f"Error adding document types: {e}")
        else:
            self.logger.warning(f"File not found: {parquet_file}")
    
    def download(
        self,
        input_parquet: Optional[Union[str, Path]] = None,
        url_column: str = 'url',
        **kwargs
    ) -> pd.DataFrame:
        """
        Download files from URLs in a parquet file.
        
        If input_parquet is not specified, it will automatically look for any .parquet file
        in the input_dir and use the first one found.
        
        Args:
            input_parquet: Path to input parquet file with URLs (optional)
                          If not provided, will search input_dir for parquet files
            url_column: Name of column containing URLs (defaults to 'url')
            **kwargs: Additional parameters to override default downloader config
                      Supported parameters include: concurrency, sleep, max_retries, etc.
                      See GlossDownloader documentation for all options
        
        Returns:
            pd.DataFrame: DataFrame with download results
        """
        # If input_parquet not specified, find parquet files in input_dir
        if input_parquet is None:
            parquet_files = list(self.input_dir.glob('*.parquet'))
            if not parquet_files:
                raise ValueError(f"No parquet files found in {self.input_dir}")
            input_parquet = parquet_files[0]
            self.logger.info(f"Using parquet file: {input_parquet}")
        else:
            input_parquet = Path(input_parquet)
        
        self.logger.info(f"Downloading files from URLs in {input_parquet}...")
        
        # Prepare downloader config with defaults and overrides
        config = self.downloader_config.copy()
        config['url_column'] = url_column  # Always set url_column, with default 'url'
        config.update(kwargs)
        
        # Set output directory to output_dir
        config['output_dir'] = str(self.output_dir)
        
        # Initialize and run downloader
        downloader = GlossDownloader(**config)
        df = downloader.download_files(input_parquet=str(input_parquet))
        
        # Save the updated dataframe with download results
        parquet_download_dir = self.output_dir / "download_results"
        os.makedirs(parquet_download_dir, exist_ok=True)
        output_parquet = parquet_download_dir / f"download_results_{input_parquet.name}"
        df.to_parquet(str(output_parquet), index=False)
        
        self.logger.info(f"Download complete. {len(df[df['download_success'] == True])} files downloaded to {self.output_dir / 'downloads'}")
        self.logger.info(f"Download results saved to {output_parquet}")
        
        return df
    
    def process_all(self, input_format: str = "pdf", fully_annotate: bool = True, annotation_type: str = "auto", download_first: bool = False) -> None:
        """
        Run the complete processing pipeline: extract, section, and annotate.
        
        Args:
            input_format: Input format (default: "pdf")
            fully_annotate: Whether to perform full annotation after classification (default: True)
            annotation_type: Annotation method to use (default: "auto")
            download_first: Whether to run the downloader before extraction (default: False)
        """
        if download_first:
            try:
                self.download()
                self.logger.info("Download step completed, proceeding with extraction...")
            except Exception as e:
                self.logger.error(f"Error during download step: {e}")
                self.logger.warning("Continuing with extraction of already downloaded files...")
                
        self.preprocess()
        self.extract(input_format=input_format)
        self.section()
        self.annotate(fully_annotate=fully_annotate, annotation_type=annotation_type)
        
        self.logger.info("Complete processing pipeline finished successfully.")