## Motif Prior Requirements

This module requires the **MEME** to be installed and accessible in your environment.  
Please ensure that the `meme` binary and its dependencies are properly configured in your system's `PATH`.

Since generating motif priors can be time-consuming, we provide **precomputed motif prior files** to facilitate fast and reproducible analysis.  
You can download them from the following repository:

**🔗 [https://doi.org/10.6084/m9.figshare.29819843.v1](https://doi.org/10.6084/m9.figshare.29819843.v1)**

After downloading, extract the contents into the appropriate directory:

```
utils/motif_prior/out/
```

Make sure this directory is correctly populated before executing any downstream analyses that rely on motif prior inputs.
