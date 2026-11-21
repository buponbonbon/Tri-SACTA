# QUARANTINE NOTICE: SYNTHETIC FIXTURES (DO NOT USE AS REAL DATA)

**Date of Quarantine:** 2026-11-21  
**Audit Finding:** V4 Critical Scientific Integrity Audit  
**Status:** BANNED FROM REAL SCIENTIFIC BENCHMARKS — PRESERVED FOR HISTORICAL AUDIT ONLY

## Reason for Quarantine
The artifacts in this directory were generated during V3 testing using `np.random` feature vectors, random labels, and artificial video/speaker IDs (`vid_trn_*`, `spk_*`), but were mistakenly placed under directories named `cmu_mosei` and `meld` and flagged as `real`.

Independent inspection confirmed:
1. Both datasets had identical class distributions and random labels.
2. Identical video IDs across datasets.
3. Feature checksums were unpopulated.
4. Chance-level accuracy (~12%) indicating random noise.

## Strict Policy
- These files MUST NEVER be loaded by any `real` mode script.
- Any attempt by dataset loaders or runners to treat these files as authentic empirical evidence must fail closed with `DataIntegrityError`.
