# V14-05: Development-Only Identifiability & Oracle Headroom (`val`, `N=250`)

Executed via `scripts/v14_research/run_v14_diagnostics.py` (29 video clusters, frozen `seed=42`, zero Phase II holdout access):

## 1. Non-Deployable Oracle Upper Bounds (Service-Quality Macro-F1)
| Workload | `Static_T` | `Static_TAV` | `Oracle_Reachable` | `Oracle_Joint_SeqPar_Feasible` | Reachable Headroom over `Static_TAV` |
|---|---|---|---|---|---|
| `CLEAN_MULTIMODAL` | `0.5636` | `0.5686` | `0.8949` | `0.8949` | `+0.3263` |
| `TEXT_EMBEDDING_NOISE` | `0.4672` | `0.4864` | `0.8727` | `0.8727` | `+0.3863` |
| `AUDIO_FEATURE_MASKING` | `0.5636` | `0.5380` | `0.7993` | `0.7993` | `+0.2613` |
| `VISUAL_ZERO_MASK` | `0.5636` | `0.5106` | `0.7769` | `0.7769` | `+0.2663` |
| `COMPOUND_FAULT` | `0.4468` | `0.4579` | `0.6916` | `0.6916` | `+0.2337` |
| `INTERMITTENT_MISSINGNESS` | `0.5636` | `0.5686` | `0.7946` | `0.7946` | `+0.2260` |
| `HETEROGENEOUS_DEADLINE_STRESS` | `0.5636` | `0.5686` | `0.8949` | `0.7572` | `+0.1886` |

## 2. Group-Cross-Fitted Learnability (5-Fold `GroupKFold` by `video_id`, `N=750`)
| Transition Target | Positive Rate (`Count`) | Brier (Constant) | Brier (State) | AUROC (Conf-Only) | AUROC (Observable State) |
|---|---|---|---|---|---|
| `rescue_T_to_TA` | `0.0747` (`56`) | `0.0694` | `0.0653` | `0.7278` | **`0.7659`** |
| `harm_T_to_TA` | `0.0560` (`42`) | `0.0533` | `0.0507` | `0.7179` | **`0.7671`** |
| `rescue_T_to_TAV` | `0.1013` (`76`) | `0.0913` | `0.0867` | **`0.7324`** | `0.7198` |
| `harm_T_to_TAV` | `0.0987` (`74`) | `0.0899` | `0.0854` | `0.6561` | **`0.7115`** |

**Diagnostic Conclusion:** Meaningful reachable headroom (`+0.19` to `+0.39` Macro-F1) and out-of-group signal beyond `max(p_T)` exist (`+0.038` to `+0.055` AUROC on `harm` and `T->TA rescue`), **but** positive transition counts (`42–76` on `val`) mean training data size (`train` split `N~16k` vs `val` `N=250`) and calibration stability must be verified in V15.
