RGB (current + history)
   ↓
DINOv2 + LoRA
   ↓
Temporal Attention
   ↓
Tri-plane Map
   ↓
Diffusion Head → K coordinate samples
   ↓
EPro-PnP (differentiable)
   ↓
RANSAC selection
   ↓
Pose loss


# ACE-G Temporal Improvement

Minimal research extension with:

- current + history frames
- DINOv2 + LoRA
- temporal attention
- deterministic coordinate head

## Run

```bash
python trainers/train_temporal.py