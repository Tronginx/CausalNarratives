# CausalNarratives
CS546 Team6 Project

## Quick Start

### Batch Extract Narratives
```bash
# Test mode (first 3 analyses)
python batch_process_narratives.py --test

# Process all analyses
python batch_process_narratives.py

# Custom limit
python batch_process_narratives.py --limit 5
```

**Input:** `Data/test_data.txt` (multiple analyses separated by `--------`)  
**Output:** `output/batch_narratives_test.json` or `output/batch_narratives.json`

### Visualize Results
```bash
# Install dependencies first
pip install pyvis matplotlib networkx

# Generate all visualizations
python visualize_narratives.py output/batch_narratives_test.json

# HTML only (interactive)
python visualize_narratives.py output/batch_narratives_test.json --format html

# PNG only (static images)
python visualize_narratives.py output/batch_narratives_test.json --format png

# Single narrative
python visualize_narratives.py output/batch_narratives_test.json --narrative 0
```

**Output:** `visualizations/` folder with HTML, PNG, and text reports

### View Interactive Graphs
```bash
open visualizations/TSM_long_interactive.html
```
