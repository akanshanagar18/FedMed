# FEDMED OS — BRATS-GLI 2024 OFFICIAL LABEL SCHEMA SPECIFICATION

**Date:** August 15, 2026  
**Auditor:** Senior ML Systems Engineer  
**Dataset:** BraTS-GLI 2024 (Adult Glioma Post-Treatment Training Cohort)  
**Classification:** `CANONICAL_LABEL_SCHEMA_SPEC`  

---

## 1. Official Ground-Truth Voxel Labels

The official corrected v2 BraTS-GLI 2024 training cohort defines five distinct voxel classes:

| Raw Label Value | Label Name | Clinical & Pathological Description | Contrast Characteristic |
|---|---|---|---|
| **0** | **BG** | Background / Normal Brain Tissue | Non-tumor tissue and air |
| **1** | **NETC** | Non-enhancing Tumor Core | Hypointense on T1c, solid tumor |
| **2** | **SNFH** | Surrounding Non-enhancing FLAIR Hyperintensity | Peritumoral edema / infiltration |
| **3** | **ET** | Enhancing Tumor / Enhancing Tissue | Hyperintense on T1c post-gadolinium |
| **4** | **RC** | Resection Cavity / Cystic Component | Fluid-filled cavity / post-op void |

---

## 2. Canonical FedMed 3-Channel Target Formulation

FedMed maintains a standard 3-output channel segmentation contract:
- **Channel 0: Tumor Core (TC)**
- **Channel 1: Whole Tumor (WT)**
- **Channel 2: Enhancing Tumor (ET)**

For BraTS-GLI 2024, the composite binary target regions are constructed as follows:

$$\mathbf{TC} = (\text{label} == 1) \lor (\text{label} == 3)$$
$$\mathbf{WT} = (\text{label} == 1) \lor (\text{label} == 2) \lor (\text{label} == 3)$$
$$\mathbf{ET} = (\text{label} == 3)$$

---

## 3. Handling of Resection Cavity (RC / Label 4)

- **Exclusion from Target Channels:** Under the standard 3-region evaluation protocol, Resection Cavity (label 4) is **not** included in TC or WT.
- **Metadata & Provenance Preservation:** The canonical adapter measures and records RC voxel counts in dataset manifests and evaluation metadata, ensuring zero data loss and complete clinical transparency.
