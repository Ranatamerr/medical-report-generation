# Reinforced Cross-modal Alignment for Radiology Report Generation

This is the implementation of Reinforced Cross-modal Alignment for Radiology Report Generation.

## Requirements
Our code works with the following environment.
- `torch==1.5.1`
- `torchvision==0.6.1`
- `opencv-python==4.4.0.42`

Clone the evaluation tools from the [website](https://github.com/salaniz/pycocoevalcap).

## Running
For `IU X-Ray`,
* `bash scripts/iu_xray/run.sh` to train the `Base+cmn` model on `IU X-Ray`.
* `bash scripts/iu_xray/run_rl.sh` to train the `Base+cmn+rl` model on `IU X-Ray`.