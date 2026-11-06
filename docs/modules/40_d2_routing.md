# Module 40: D2, abstention routing with a disease branch

**Compute:** CPU. Runs locally in a few seconds.

**Question.** When a leaf is outside the new crop's healthy envelope, does D2 send it to the right route (proposal section 6)? Routes simulated: **answer** (inside an envelope), **disease route** (outside every envelope but matching a stored disease signature), and **wrong-crop route** (outside every envelope and matching no signature). Guided recapture and referral need repeated captures and are not simulated.

**Setup.**
- 16-dimensional projection (module 08), PlantVillage only. 8 crops are enrolled and 3 are held out as unenrolled (apple, raspberry, blueberry, chosen with a fixed seed).
- **Envelope:** Mahalanobis distance to the crop's healthy Gaussian at most the 95th percentile of that crop's own training distances.
- **Signature:** log-likelihood under the disease Gaussian at least the 5th percentile of that disease's own training log-likelihoods. 19 disease signatures.
- Thresholds use the **training split only**. Tests use the **validation split**.
- **Baseline:** a single abstention route: any leaf outside every envelope abstains.

## Results (`results/40_d2_routing/routing_rates.csv`)

| Test set | Leaves | Answer | Disease route | Wrong-crop route |
|---|---|---|---|---|
| Diseased leaves, enrolled crops | 2,858 | 0.02 | **0.93** | 0.04 |
| Healthy leaves, enrolled crops | 1,119 | **0.99** | 0.00 | 0.01 |
| Leaves of unenrolled crops | 444 | **0.76** | 0.02 | **0.22** |

- Of the diseased leaves that reach the disease route, **82% get the correct disease class**.
- Baseline: the single abstention route sends 98% of diseased leaves to abstention (no answer), where D2 returns a disease class for 93%.

## Reading

- **The disease branch works.** In the baseline, a diseased leaf of an enrolled crop is always treated as outside the healthy envelope and refused. With the disease branch, 93% are routed to a disease answer, and 82% of those are right. This is the improvement the branch was added for.
- **Wrong-crop detection is weak.** Only 22% of leaves from unenrolled crops go to the wrong-crop route. **76% are answered** as healthy, because they fall inside the envelope of another enrolled crop. This matches the D1 finding (module 39): in this projection, healthy leaves of different crops are not well separated by an envelope. The proposal's wrong-crop route cannot be relied on for detection with these tests.
- A healthy leaf is rarely misrouted: 99% are answered.
- 2% of diseased leaves are answered as healthy (inside an envelope), which is a miss.

## Limits

- 3 unenrolled crops and a single random draw of which crops are enrolled. A different draw would change the wrong-crop rate.
- The envelope threshold (95th percentile) and signature threshold (5th percentile) are set once, not tuned. A tighter envelope would catch more wrong-crop leaves and refuse more healthy ones.
- Only PlantVillage (laboratory images) is used. Field images may behave differently.
- Recapture and referral are not simulated. No confidence intervals.

**How to re-run:**
```bash
python features/40_d2_routing.py --features results/05_features_dinov2/remote/features_dinov2 --labelled results/01_crop_disease_table/labelled_images.csv --projection results/08_projection/projection_k16.npz --output results/40_d2_routing
```
