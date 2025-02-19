# Excel Processor

A production-grade Python package for processing and analyzing Excel spreadsheets. This tool provides advanced features for spreadsheet compression, formula analysis, and semantic extraction.

## Features

- **Spreadsheet Compression**: Efficiently compress spreadsheets while preserving structure and data integrity
- **Formula Analysis**: Extract and analyze Excel formulas with dependency tracking
- **Semantic Clustering**: Group cells based on semantic similarity using advanced NLP techniques
- **Data Format Aggregation**: Automatically categorize cells by their data formats
- **Structural Anchor Extraction**: Preserve table structure while removing empty rows/columns
- **Inverted Index Translation**: Create efficient lookup structures for cell values

## Installation

```bash
pip install -r requirements.txt
```


## Requirements

- Python 3.8+
- pandas
- numpy
- openpyxl
- sentence-transformers
- matplotlib

## Usage

### Basic Processing

```python
from excel_processor import ExcelProcessor

processor = ExcelProcessor('path/to/your/spreadsheet.xlsx')
processor.process()
```


## API Documentation

### Main Classes

- **ExcelPipeline**: Main processing pipeline
- **SpreadsheetLLM**: Core spreadsheet processing logic
- **ExcelLoader**: Load and parse Excel files
- **SheetCompressor**: Compress multiple sheets
- **FormulaAnalyzer**: Analyze Excel formulas
- **OutputGenerator**: Generate structured output


## Performance

The processor is optimized for large spreadsheets with:
- Memory-efficient processing
- Parallel computation where applicable
- Intelligent caching mechanisms