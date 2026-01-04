# Experiment Browser

A Panel-based web interface for browsing and visualizing HDF5/NPZ experiment files.

## Files

| File | Description |
|------|-------------|
| `db.py` | Database schema initialization |
| `scan_file.py` | Scans HDF5/NPZ files to extract dataset metadata |
| `index_experiments.py` | Indexes files from a directory into the database |
| `browser.py` | Panel web application for browsing and plotting |
| `metrics.py` | Derived metrics computation (optional) |

## Setup

### 1. Create/Update the Database

Edit `index_experiments.py` to set your root directory:

```python
ROOT_DIR = "/path/to/your/experiment/files"
```

Then run:

```bash
cd index_experiments
python index_experiments.py
```

This will:
- Scan for `.h5`, `.hdf5`, and `.npz` files
- Extract dataset metadata (keys, shapes, dtypes)
- Store everything in `experiments.db`

### 2. Launch the Browser

**Local machine:**
```bash
panel serve browser.py
```
Open http://localhost:5006/browser in your web browser.

**Remote server (SSH tunnel):**

1. On the remote server, start the browser:
```bash
cd index_experiments
panel serve browser.py --port 5006 --allow-websocket-origin="*"
```

2. On your local machine, create an SSH tunnel:
```bash
ssh -N -L 5006:localhost:5006 user@remote-server
```

3. Open http://localhost:5006/browser in your local browser.

**Alternative: Single command with SSH tunnel**
```bash
ssh -L 5006:localhost:5006 user@remote-server "cd /path/to/index_experiments && panel serve browser.py --port 5006"
```

## Usage

### Browsing Files
- Use **Filters** to narrow down files by type, owner, or date
- Select files in the **Files** table using checkboxes
- View file details and datasets in **Selected Files**

### Plotting
1. Check datasets you want to plot in the Selected Files panel
2. Datasets become available as variables named `{filename}_{dataset_key}`
3. Write matplotlib code in the **Plot Controls** editor
4. Click **Run** to execute

### Example Plot Code

```python
# Single plot
plt.plot(myfile_loss)
plt.title("Training Loss")
plt.show()

# Multiple plots (side by side)
plt.figure()
plt.plot(exp1_accuracy)
plt.title("Experiment 1")
plt.show()

plt.figure()
plt.plot(exp2_accuracy)
plt.title("Experiment 2")
plt.show()
```

### Chunk Range
Use the **Range** slider to select a subset of data along dimension 0. The shape display shows the resulting array dimensions.

## Requirements

- panel
- param
- pandas
- numpy
- h5py
- matplotlib
