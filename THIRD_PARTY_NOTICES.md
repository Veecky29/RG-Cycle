# Third-party notices

Parts of this repository are adapted from **CycleNet** (https://github.com/ACAT-SCUT/CycleNet, commit d807e51),
Copyright the CycleNet authors, licensed under the Apache License, Version 2.0
(http://www.apache.org/licenses/LICENSE-2.0):

- `rgcycle/cycle.py`: `RecurrentCycle` (from `models/CycleNet.py` / `models/CycleiTransformer.py`); cycle-index convention
  from `data_provider/data_loader.py`.
- `rgcycle/backbones.py`: `CycleNetNorm`, `FullAttention`, `AttentionLayer`, `CNEncoderLayer`, `CycleNetITransformer`
  (from `models/CycleiTransformer.py`, `layers/Embed.py`, `layers/Transformer_EncDec.py`, `layers/SelfAttention_Family.py`).
  Changes: unused branches (masking, timestamp covariates, attention output) removed; the cycle is applied by a wrapper.
- `rgcycle/trainer.py`: the `cyclenet` recipe re-implements the schedule, early stopping and validation behaviour of
  `exp/exp_main.py`, `utils/tools.py` and `data_provider/data_factory.py`.

The iTransformer architecture is by Liu et al. (https://github.com/thuml/iTransformer, MIT License).
