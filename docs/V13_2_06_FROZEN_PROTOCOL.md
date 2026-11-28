# Tri-SACTA V13.2: Frozen Executable Confirmatory Protocol

* **Manifest Path:** [freeze_manifest.json](../results/v13_2/protocol/freeze_manifest.json)
* **Detached Digest Path:** [freeze_manifest.sha256](../results/v13_2/protocol/freeze_manifest.sha256)
* **Manifest SHA-256:** `fb23d9e16f8557e6aaf807d893fd5d8bf756d96e792e9c835934046e960cbdb3`
* **Scientific Source Commit SHA:** `d59c6f5ba6834494078d9cece2241e7838723df6`
* **Eligible Holdout Manifest SHA-256:** `146157febd1f2c775e4195f694293d6e8701fcf4e8537884e4d15ad55251502e` (`4131` utterances / `617` video clusters)
* **Raw Asset (`mosei.pkl`) SHA-256:** `90701ad5bfa8767ca253af729c1d346e9a8d259ed24f16de113c845ffb60d90e`
* **Locked Checkpoints:** `21` subset MLPs ([checkpoint_hashes.json](../results/v13_2/protocol/checkpoint_hashes.json))
* **Primary Resource Endpoint:** Measured application-protocol bytes per submitted request across 7 workloads $\times$ 3 seeds, jointly with Service-Quality Macro-F1, vs `AlwaysOn_Direct_Serial`, `AlwaysOn_Direct_Parallel`, and `Static_TA_Direct`.
* **Phase II Execution Gate:** `python scripts/v13_2_run_phase2_holdout.py --execute-holdout --approval <AUTHOR_APPROVAL_JSON>` (currently `PENDING_AUTHOR_APPROVAL`).
