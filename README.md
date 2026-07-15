# 🧠 FedMed – Cross-Silo Federated Learning Engine

> **Privacy-Preserving Federated Learning for Brain Tumor MRI Segmentation**

FedMed is a Privacy-Preserving Machine Learning (PPML) project that enables multiple hospitals to collaboratively train a deep learning model for brain tumor segmentation without sharing sensitive patient data.

Instead of transferring MRI scans to a central server, each participating hospital trains the model locally on its own infrastructure. Only encrypted model updates are shared with a central aggregation server, ensuring that patient data never leaves the hospital while still allowing all participants to benefit from a globally improved model.

By combining **Federated Learning**, **Homomorphic Encryption**, and **Differential Privacy**, FedMed provides a secure and scalable framework for collaborative healthcare AI while maintaining compliance with regulations such as **HIPAA** and **GDPR**.

---

## 📌 Problem Statement

Developing accurate AI models for medical imaging requires large and diverse datasets collected from multiple healthcare institutions. However, strict privacy regulations and data ownership policies prevent hospitals from sharing sensitive patient information.

As a result, institutions often train models independently on limited datasets, leading to reduced model performance and poor generalization.

FedMed addresses this challenge by enabling hospitals to collaboratively train a shared deep learning model without exchanging raw patient data. Each institution performs local training on its private MRI scans and securely shares only encrypted model updates, preserving patient privacy throughout the training process.

---

## 💡 Proposed Solution

FedMed follows a **Cross-Silo Federated Learning** architecture in which multiple hospitals jointly train a global model while retaining complete control over their local datasets.

The central server initializes a global model and distributes it to each participating hospital. Every hospital trains the model locally using its private MRI dataset before encrypting the updated model parameters. These encrypted updates are securely transmitted to the central server, where they are aggregated without accessing any patient data. The updated global model is then redistributed for the next training round until convergence is achieved.

This decentralized approach enables collaborative model training while ensuring that sensitive medical information remains within each institution's secure environment.

---
## ✨ Key Features

- 🏥 **Cross-Silo Federated Learning** enabling multiple hospitals to collaboratively train a shared model without exchanging raw patient data.
- 🧠 **3D U-Net for Medical Image Segmentation** using PyTorch and MONAI for accurate brain tumor segmentation from MRI scans.
- 🔒 **Homomorphic Encryption** with TenSEAL to encrypt model updates and preserve data privacy during collaborative training.
- 🛡️ **Differential Privacy** to protect model updates against potential privacy attacks.
- 📊 **Interactive Training Dashboard** built with React and Recharts to visualize training progress, loss, and accuracy in real time.

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Programming Language** | Python |
| **Deep Learning** | PyTorch, MONAI |
| **Federated Learning** | Flower |
| **Privacy & Security** | TenSEAL, Differential Privacy |
| **Frontend** | React, Recharts |
