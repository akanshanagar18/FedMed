# Medical Model Module Guide

**Owners:** Mohit & Shreya
**Folder:** `model/`

## Objective
Implement the 3D U-Net architecture, BraTS dataset loaders, and medical evaluation metrics (Dice Score, IoU) using PyTorch and MONAI. 

## Dependencies
- **None.** This module is purely mathematical and domain-specific.

## Integration Points
- **Expected Inputs:** Raw hyperparameters (Learning Rate, Epochs) passed in by the FL Client.
- **Expected Outputs:** Standard PyTorch State Dictionaries (weights) and float metrics.

## Development Checklist
- [ ] Build the BraTS MRI dataloader.
- [ ] Define the 3D U-Net.
- [ ] Implement local train/test functions.
- [ ] Define the Dice/IoU loss calculation.

## Common Mistakes to Avoid
- **Do not** write any network or Flower code. Your training loops must be isolated so they can be tested locally without a network.
- **Do not** encrypt data here. Pass raw float tensors; the FL client will handle encryption routing.
