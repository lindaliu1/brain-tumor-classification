#  Cleaned Brain Tumor MRI Dataset (NumPy Version)

A **cleaned, verified, and NumPy-ready** version of the original [Figshare Brain Tumor MRI Dataset (Cheng et al., 2017)](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427).  
This release converts all `.mat` files into consistent `.npy` arrays — ensuring **no corruption, no missing keys, and fully aligned masks, labels, and patient IDs.**

---

## 📘 Overview

MRI brain tumor data is widely used for **classification**, **segmentation**, and **deep learning** research.  
However, the original Figshare dataset contained several issues:

- ❌ Corrupted `.mat` archives  
- ❌ Missing segmentation masks or mismatched shapes  
- ❌ Non-standard key structures across tumor types  

This cleaned version fixes all of that.  
All data are now provided as **NumPy arrays**, which can be loaded directly into Python with `numpy.load()` — **no MATLAB required.**

---

## 📂 Dataset Structure

Each MRI slice corresponds to **five synchronized NumPy files**, all sharing the same numeric prefix (e.g., `1000`).

| Type | Folder | Example Filename | Description |
|------|---------|------------------|--------------|
| Image | `images/` | `1000_image.npy` | 512×512 grayscale MRI slice |
| Mask | `masks/` | `1000_mask.npy` | Binary tumor segmentation mask |
| Label | `labels/` | `1000_label.npy` | Tumor class (integer 1–3) |
| Border | `borders/` | `1000_border.npy` | Edge/boundary map for visualization |
| PID | `pids/` | `1000_PID.npy` | Patient/slice identifier array |

All files with the same prefix represent one MRI sample.

---

## Classes and Labels

| Label ID | Tumor Type | Description |
|-----------|-------------|-------------|
| `1` | **Meningioma** | Tumors that arise from meninges (outer brain layer) |
| `2` | **Glioma** | Tumors originating from glial cells (most common) |
| `3` | **Pituitary** | Tumors found in the pituitary gland region |

---

## ⚙️ Quick Usage (Python)

Load any MRI slice and its related data:

```python
import numpy as np

base_id = "1000"

image  = np.load(f"images/{base_id}_image.npy")
mask   = np.load(f"masks/{base_id}_mask.npy")
label  = np.load(f"labels/{base_id}_label.npy")
border = np.load(f"borders/{base_id}_border.npy")
pid    = np.load(f"pids/{base_id}_PID.npy")

print("Image shape:", image.shape)
print("Label:", label)
```


## 📊 Dataset Statistics

| Property | Description |
| :--- | :--- |
| **Format** | NumPy `.npy` arrays |
| **Image size** | 512×512 pixels |
| **Channels** | Grayscale (1 channel) |
| **Classes** | 3 (Meningioma, Glioma, Pituitary) |
| **File integrity** | All verified and loadable |
| **Approx. Samples** | [Fill after verification] |

---

##  Source & Credits

**Original dataset:**

* Cheng, Jun et al. Brain Tumor Dataset. Figshare (2017).
* DOI: 10.6084/m9.figshare.1512427

**Cleaned NumPy version:**

* Converted, verified, and curated by Atharav Sonawane (2025)
* *Repaired corrupted .mat files, unified structure, and exported standardized .npy arrays.*

---
## 🧾 Citation

If you use this dataset, please cite both the original authors and this NumPy version:

```bibtex
@dataset{cheng2017brain,
  author    = {Cheng, Jun and others},
  title     = {Brain Tumor Dataset},
  year      = {2017},
  publisher = {Figshare},
  doi       = {10.6084/m9.figshare.1512427}
}
```

## 💡 **Recommended Use Cases**

* 🧠 Brain tumor classification and segmentation
* 🧩 U-Net / Autoencoder training and benchmarking
* 🔁 Transfer learning on MRI data
* 🧰 Data preprocessing or augmentation research
* 🧪 Educational and research projects in medical imaging