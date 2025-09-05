## 🧬 Motif Prior Setup

BRIDGE supports two modes for incorporating **motif prior information** into the prediction pipeline:

---

### 🛠️ Option 1: Generate Motif Priors (Requires MEME)

To generate motif priors from input sequences, you must make sure:

- The `meme` binary and its dependencies are correctly installed
- Your `PATH` includes the motif generation script:

```bash
export PATH="/YourPath/BRIDGE/utils/motif_prior:$PATH"
```

💡 **Note**: Motif generation can be time-consuming, especially on large datasets. Use this option if you wish to derive custom priors from your own data.

---

### ⚡ Option 2: Use Precomputed Motif Priors (Recommended)

To streamline your workflow, we provide **precomputed motif prior files** covering all 261 RBP datasets across six human cell lines. These files are ready to use for downstream analysis.

📥 **Download Link**:
🔗 [https://doi.org/10.6084/m9.figshare.29819843.v2](https://doi.org/10.6084/m9.figshare.29819843.v2)

After downloading, extract the contents into the following directory:

```
BRIDGE/utils/motif_prior/output/
```

Ensure this folder is properly populated **before** running BRIDGE modules that rely on motif prior integration.
