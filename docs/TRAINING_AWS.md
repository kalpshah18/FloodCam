# Training SegFormer on one AWS GPU instance

Everything here is run by you; nothing in the repo downloads data or weights on its own.
Checkpoints and `results/` are the only things you copy back.

## 0. Before you launch

1. **Budget alert first.** AWS Billing console -> Budgets -> create a monthly cost budget
   (e.g. a small amount you are comfortable losing) with an email alert at 50% and 80%.
2. **Datasets and licences.** Download the public datasets on your laptop or on the instance,
   read each licence at the source, tick the checklist printed by
   `python scripts/prepare_public_data.py`, then set `license_verified: true` in
   `configs/seg_classmap.yaml` per dataset. Do not upload private recordings.
3. **S3 bucket** (private, block-public-access on), e.g. `s3://<your-bucket>/floodcam/`.
4. **IAM role, no keys.** Create an EC2 instance role with a policy limited to that bucket
   (`s3:GetObject`, `s3:PutObject`, `s3:ListBucket` on the bucket/prefix only) and attach it
   as the instance profile. Never `aws configure` with long-lived keys, never paste keys into
   the shell or the repo; the instance picks up temporary credentials from the role.

## 1. Launch

- Instance: **g4dn.xlarge** (1x NVIDIA T4, 16 GB) is enough for SegFormer-B0/B1 at 512 px.
  An **A10G-class** instance (g5.xlarge) is faster if the T4 is too slow.
- AMI: **AWS Deep Learning AMI (PyTorch, Ubuntu)** - CUDA and PyTorch preinstalled.
- Storage: >= 150 GB gp3 if you use full ADE20K + COCO-Stuff; less for a subset.
- Security group: SSH from your IP only. Attach the IAM instance profile from step 0.

## 2. Set up

```bash
ssh -i <key.pem> ubuntu@<instance-dns>
source activate pytorch     # or the AMI's PyTorch env; check the login banner
git clone <your-repo-url> floodcam && cd floodcam
pip install -e ".[ml]"      # base deps + torch/transformers/timm/accelerate
python -m pytest tests -q -m "not torch" && python scripts/train_seg.py --smoke --overwrite
```

The smoke run (tiny random model, 2 steps, no downloads) must pass before you spend GPU hours.
If the AMI's torch differs from the pin in `pyproject.toml`, that is fine; it is recorded in
`results/<run>/meta.json`.

## 3. Data

```bash
aws s3 sync s3://<your-bucket>/floodcam/public data/public     # or download from the source
python scripts/prepare_public_data.py                           # prints the layout you need
python scripts/prepare_public_data.py --verify                  # checks files, class names, hashes
git add data/splits/public_seg_splits.json data/public/DATASETS.md   # IDs + hashes only
```

`--verify` fails loudly if a mapped class is not in the dataset's own label file; fix
`configs/seg_classmap.yaml` rather than the data. FloodNet stays off unless you set
`enabled: true` (aerial imagery; see `docs/DECISIONS.md`).

## 4. Train (exact commands)

Stage 1, pretrain on public data (SegFormer-B0, ImageNet encoder, downloads `nvidia/mit-b0`
from the Hugging Face hub on first use):

```bash
python scripts/train_seg.py --config configs/seg_pretrain.yaml \
    --run-id seg-pre-001 --output-dir checkpoints
```

B1 instead: set `model: segformer_b1` in the config (or copy it to a new YAML). To start from
an ADE-pretrained SegFormer instead of the ImageNet encoder, set `init_weights:` to that
checkpoint's hub id or a local path; whichever you use is written to `meta.json`.

Stage 2, fine-tune on your self-labeled ground-level frames (only after milestone 10 produced
`data/self_labeled/manifest.jsonl` and `data/splits/real_seg_finetune_splits.json`; the
locked final-test scene must never appear in either):

```bash
python scripts/train_seg.py --config configs/seg_finetune.yaml \
    --run-id seg-ft-001 --output-dir checkpoints
```

Stage 2 refuses synthetic rows and the locked scene by design. Evaluate (validation scenes
only; the locked scene goes through `scripts/final_test.py`, later):

```bash
python scripts/eval_seg.py --backend segformer --ckpt checkpoints/seg-pre-001/best \
    --split val --run-id evalseg-pre-001
```

Outputs: `results/<run-id>/{metrics.json,curves.csv,meta.json}` and
`checkpoints/<run-id>/{best,last}/` (gitignored).

## 5. Copy results out and stop

```bash
aws s3 sync checkpoints s3://<your-bucket>/floodcam/checkpoints
aws s3 sync results     s3://<your-bucket>/floodcam/results
sudo shutdown -h now    # then STOP/TERMINATE the instance in the console
```

Verify in the EC2 console that the instance state is *stopped* (a stopped instance still
bills for its EBS volume; terminate and delete the volume once the S3 copy is confirmed).
Commit `results/<run-id>/{metrics.json,curves.csv,meta.json}` (small); never the checkpoints.

## Cost

`cost = hourly_price x hours` (plus EBS storage and S3). Look up the **current** on-demand
(or spot) price for your region on the AWS pricing page; do not trust a number written in a
document. Example: if the instance costs P USD/hour and you train 6 hours, cost = 6P.
Add a margin for setup and data transfer.

## Free alternatives

Google Colab (free T4, session limits) and Kaggle notebooks (free GPU quota) can run
`pip install -e ".[ml]"` plus the same commands; keep checkpoints in the notebook's storage.
