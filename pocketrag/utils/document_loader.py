"""
PocketRAG - Document Processing Utilities

Supports multiple document formats:
- Plain text files (txt, md, code files)
- PDF documents
- Microsoft Word (.docx)
- CSV files
- JSON/YAML structured data
"""
import logging
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Handles loading and extracting text from various document formats.
    
    Supported formats:
    - Text: .txt, .md, and all code file extensions
    - PDF: .pdf (requires pypdf)
    - Word: .docx (requires python-docx)
    - Structured: .json, .yaml, .yml, .csv
    - Web: .html, .xml
    """
    
    SUPPORTED_EXTENSIONS = {
        # Plain text and code
        '.txt': '_load_text',
        '.md': '_load_text',
        '.py': '_load_text',
        '.js': '_load_text',
        '.ts': '_load_text',
        '.tsx': '_load_text',
        '.jsx': '_load_text',
        '.json': '_load_json',
        '.yaml': '_load_yaml',
        '.yml': '_load_yaml',
        '.csv': '_load_csv',
        '.html': '_load_html',
        '.xml': '_load_xml',
        '.css': '_load_text',
        '.java': '_load_text',
        '.cpp': '_load_text',
        '.c': '_load_text',
        '.h': '_load_text',
        '.hpp': '_load_text',
        '.go': '_load_text',
        '.rs': '_load_text',
        '.rb': '_load_text',
        '.php': '_load_text',
        '.swift': '_load_text',
        '.kt': '_load_text',
        '.scala': '_load_text',
        '.sh': '_load_text',
        '.bash': '_load_text',
        '.sql': '_load_text',
        '.r': '_load_text',
        '.matlab': '_load_text',
        # Documents
        '.pdf': '_load_pdf',
        '.docx': '_load_docx',
    }
    
    def __init__(self):
        self._pdf_reader = None
        self._docx_doc = None
    
    def _get_pdf_reader(self):
        """Lazy load PDF reader to avoid import overhead when not needed."""
        if self._pdf_reader is None:
            try:
                from pypdf import PdfReader
                self._pdf_reader = PdfReader
            except ImportError:
                logger.warning("pypdf not installed. PDF support disabled.")
                return None
        return self._pdf_reader
    
    def _get_docx_doc(self):
        """Lazy load DOCX reader."""
        if self._docx_doc is None:
            try:
                from docx import Document
                self._docx_doc = Document
            except ImportError:
                logger.warning("python-docx not installed. DOCX support disabled.")
                return None
        return self._docx_doc
    
    def load(self, file_path: Path) -> Optional[str]:
        """
        Extract text content from a file.
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            Extracted text content or None if loading failed
        """
        if not file_path.exists():
            logger.debug(f"File not found: {file_path}")
            return None
        
        ext = file_path.suffix.lower()
        
        if ext not in self.SUPPORTED_EXTENSIONS:
            logger.debug(f"Unsupported file type: {ext}")
            return None
        
        method_name = self.SUPPORTED_EXTENSIONS[ext]
        method = getattr(self, method_name)
        
        try:
            content = method(file_path)
            if content and content.strip():
                return content
            logger.debug(f"No content extracted from {file_path.name}")
            return None
        except Exception as e:
            logger.warning(f"Failed to load {file_path.name}: {e}")
            return None
    
    def load_multiple(self, file_paths: List[Path]) -> Dict[str, Optional[str]]:
        """
        Load multiple files at once.
        
        Args:
            file_paths: List of file paths to load
            
        Returns:
            Dictionary mapping file paths to their content
        """
        results = {}
        for path in file_paths:
            results[str(path)] = self.load(path)
        return results
    
    def _load_pdf(self, file_path: Path) -> Optional[str]:
        """Extract text from PDF files."""
        PdfReader = self._get_pdf_reader()
        if PdfReader is None:
            return None
        
        try:
            reader = PdfReader(file_path)
            pages_text = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    pages_text.append(f"[Page {i+1}]\n{text}")
            return "\n\n".join(pages_text) if pages_text else None
        except Exception as e:
            logger.warning(f"PDF extraction failed for {file_path}: {e}")
            return None
    
    def _load_docx(self, file_path: Path) -> Optional[str]:
        """Extract text from DOCX files."""
        Document = self._get_docx_doc()
        if Document is None:
            return None
        
        try:
            doc = Document(file_path)
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            return "\n\n".join(paragraphs) if paragraphs else None
        except Exception as e:
            logger.warning(f"DOCX extraction failed for {file_path}: {e}")
            return None
    
    def _load_text(self, file_path: Path) -> Optional[str]:
        """Load plain text files with encoding fallback."""
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                content = file_path.read_text(encoding=encoding)
                # Basic cleaning: remove null bytes which can crash some processors
                return content.replace('\x00', '')
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.debug(f"Failed to read {file_path} with {encoding}: {e}")
                break
        
        logger.warning(f"Could not decode {file_path} with any supported encoding")
        return None
    
    def _load_json(self, file_path: Path) -> Optional[str]:
        """Load and format JSON files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Pretty print JSON
            return json.dumps(data, indent=2, ensure_ascii=False)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in {file_path}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to load JSON {file_path}: {e}")
            return None
    
    def _load_yaml(self, file_path: Path) -> Optional[str]:
        """Load YAML files."""
        try:
            import yaml
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # Convert back to string representation
            if isinstance(data, dict):
                lines = []
                for key, value in data.items():
                    lines.append(f"{key}: {value}")
                return "\n".join(lines)
            return str(data)
        except ImportError:
            logger.warning("PyYAML not installed. YAML support disabled.")
            return self._load_text(file_path)
        except Exception as e:
            logger.warning(f"Failed to load YAML {file_path}: {e}")
            return self._load_text(file_path)
    
    def _load_csv(self, file_path: Path) -> Optional[str]:
        """Load CSV files as formatted text."""
        try:
            import csv
            with open(file_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.reader(f)
                rows = list(reader)
            
            if not rows:
                return None
            
            # Format as a simple table
            max_cols = max(len(row) for row in rows)
            lines = []
            for row in rows:
                # Pad row to max_cols
                row = row + [''] * (max_cols - len(row))
                lines.append(' | '.join(str(cell).strip() for cell in row))
            
            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"Failed to load CSV {file_path}: {e}")
            return self._load_text(file_path)
    
    def _load_html(self, file_path: Path) -> Optional[str]:
        """Load HTML files and extract text content."""
        try:
            from bs4 import BeautifulSoup
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
            
            # Remove script and style elements
            for element in soup(['script', 'style', 'meta', 'link']):
                element.decompose()
            
            text = soup.get_text(separator='\n', strip=True)
            return text if text else None
        except ImportError:
            logger.debug("BeautifulSoup not installed, loading HTML as plain text")
            return self._load_text(file_path)
        except Exception as e:
            logger.warning(f"Failed to parse HTML {file_path}: {e}")
            return self._load_text(file_path)
    
    def _load_xml(self, file_path: Path) -> Optional[str]:
        """Load XML files and extract text content."""
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Get all text content
            text_parts = []
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    text_parts.append(elem.text.strip())
                if elem.tail and elem.tail.strip():
                    text_parts.append(elem.tail.strip())
            
            return '\n'.join(text_parts) if text_parts else None
        except Exception as e:
            logger.warning(f"Failed to parse XML {file_path}: {e}")
            return self._load_text(file_path)
