# Conditional Quantum GAN vs Conditional Classical GAN

Clean-room implementation of a **conditional** patch-method quantum GAN
trained on the UCI handwritten-digits dataset, alongside two classical
conditional baselines for comparison. Both notebooks share a single
quantitative-evaluation suite so the models are scored on identical
metrics.

The model learns to generate 8×8 grayscale digits **on demand**: given a
class label `c ∈ {0, …, 9}` and a random latent vector `z`, the
generator produces an image of digit `c`.

## Project layout

```
.
├── data/
│   └── optdigits.tra              # UCI handwritten-digits dataset (training split)
├── dataset.py                     # DigitsDataset (multi-class) + load_full_digits()
├── discriminator.py               # ConditionalDiscriminator + ConditionalDiscriminatorAC (Pass-3)
├── qcircuit.py                    # Patch-method quantum circuit (PennyLane QNode)
├── qgenerator.py                  # ConditionalPatchQuantumGenerator
├── classical_generator.py         # ConditionalClassicalGenerator + ConditionalTinyGenerator
├── metrics.py                     # Quantitative evaluation suite (judge + FID + IS + speed + ...)
├── quantum_gan.ipynb              # Train + evaluate the conditional quantum GAN
├── classical_gan.ipynb            # Train + evaluate the classical baselines, cross-compare
├── requirements.txt
└── README.md
```

## Conditioning approach

* **Quantum cGAN.** The generator keeps the original four sub-generators
  (5 qubits, depth 6, 1 ancilla each) and adds a single shared learnable
  per-class rotation offset `θ_class ∈ ℝ^{10 × 5}` (a small
  `nn.Embedding`). The latent vector fed to every sub-generator's
  circuit is `z + θ_class[c]`. Because consecutive `RY` rotations on
  the same wire compose as `RY(a) RY(b) = RY(a+b)`, this is
  mathematically identical to inserting per-class `RY(θ_class[c, i])`
  rotations right after the state-embedding stage of the circuit -- the
  circuit topology in `qcircuit.py` is therefore unchanged.

* **Classical cGAN.** The label is fed through `nn.Embedding(10, e)`
  and the embedding is concatenated with `z` before the MLP. Same idea
  for the discriminator: the label embedding is concatenated with the
  flattened image.

## Parameter budgets

| Model | Generator params | Discriminator params |
| --- | ---: | ---: |
| **Conditional Quantum GAN — Pass 2** | **170** (120 sub-gen + 50 class emb) | 6 401 (label-conditional) |
| **Conditional Quantum GAN — Pass 3** | **250** (200 sub-gen + 50 class emb) | 5 387 (AC-GAN, no label-as-input) |
| Standard classical cGAN | ~9 888 | 6 401 |
| Parameter-matched classical cGAN | ~189 | 6 401 |

The parameter-matched classical generator is the like-for-like
comparison: both it and the Pass-2 quantum cGAN have ~170–190
trainable parameters. The Pass-3 quantum cGAN raises the generator
to ~250 params and switches the discriminator to the AC-GAN
formulation — see the "Tuning" section for the motivation.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The dataset (`data/optdigits.tra`) is included in the repository. To
re-download:

```bash
curl -fsSL \
  https://raw.githubusercontent.com/PennyLaneAI/qml/master/_static/demonstration_assets/quantum_gans/optdigits.tra \
  -o data/optdigits.tra
```

## Running

Open the notebooks in order:

1. **`quantum_gan.ipynb`** — trains and evaluates the conditional patch
   quantum GAN. Takes roughly 12–25 minutes per 300 iterations on a
   modern CPU because the parameter-shift gradient runs two extra
   circuit evaluations per parameter and the conditional case uses
   `batch_size=8` to ensure each batch sees multiple classes.
2. **`classical_gan.ipynb`** — trains and evaluates two classical
   baselines (≈ tens of seconds each) and, if `quantum_gan.pt` exists,
   automatically loads the quantum metrics for a side-by-side
   comparison.

Each notebook saves a checkpoint (`quantum_gan.pt` / `classical_gan.pt`)
containing trained weights, training history, **and** the full
metrics dict, so the scoreboard can be rebuilt without retraining.

> The first run of either notebook will download the InceptionV3
> weights (~100 MB) used by the standard Inception Score. Set
> `SKIP_STANDARD_IS = True` in the relevant cell to bypass; the
> domain-specific FID and judge metrics do not require the download.

## Tuning

The model architectures are deliberately fixed (so the parameter budget
stays at ~170 for the quantum cGAN and ~190 for the parameter-matched
classical cGAN). Quality was instead improved with optimizer- and
loss-side tricks, applied in two passes.

### Pass 1 — controlled hyper-parameter sweep

The first pass was a five-seed sweep that held seeds, the dataloader
and the model initialisation fixed across configurations.

**Classical cGAN sweep** (5 seeds × 2000 iters, judge accuracy ↑):

| Config (additive) | mean acc | min acc | max acc |
| --- | ---: | ---: | ---: |
| Baseline (BCE 1.0/0.0, random fake labels) | 0.929 | 0.853 | 1.000 |
| + label smoothing 0.9 | 0.899 | 0.587 | 0.999 |
| + label smoothing 0.9 + matched fake labels | 0.936 | 0.888 | 1.000 |
| 5000 iters, smoothing + match | 0.887 | 0.800 | 0.995 |
| 5000 iters, smoothing + match, fresh fakes for G | 0.914 | 0.820 | 0.990 |

**Quantum cGAN sweep** (single seed × 100 iters, FID ↓):

| Config | FID overall | per-class FID (mean) |
| --- | ---: | ---: |
| Baseline: SGD (lrG=0.3, lrD=0.01) | 153.85 | 223.05 |
| SGD + label smoothing 0.9 | 156.79 | 224.72 |
| SGD + smoothing + matched labels | 156.52 | 226.40 |
| Adam (lr 5e-3 both, β₁ 0.5) + smoothing | 140.33 | 215.02 |
| **Adam TTUR (lrG 1e-2, lrD 1e-3) + smoothing** | **87.31** | **167.49** |

For the classical case label smoothing alone has a long worst-case
tail (min 0.587) and only stabilises when combined with matched fake
labels; doubling iterations to 5000 actively hurts because D
over-trains. For the quantum case the original tutorial's high-LR SGD
makes virtually no progress at `batch_size = 8` (D and G stay on the
BCE plateau), and replacing it with Adam plus a 10× higher G learning
rate (TTUR-style) cuts domain FID from 154 to 87 in the same
wall-clock budget.

### Pass 2 — diagnosing mode collapse

Pass 1 reduced quantum FID dramatically but the actual quantum run
showed a textbook **mode-collapse failure on the class axis**: the
confusion matrix had every requested class decoded as a single digit
(~8) and per-class FID ranged from 75 to 200. The G loss climbed
monotonically while D saturated below 1.0.

The diagnosis is that **the discriminator was minimising its loss
without using the class label**. Concatenating a 16-dim label embedding
to a 64-dim image gives D an easy way out: just classify by pixels and
ignore the label. Without label-driven gradients, G has no reason to
differentiate classes either.

The fix is the **matching-aware discriminator loss** from
[Reed et al. 2016, "Generative Adversarial Text to Image Synthesis"](https://arxiv.org/abs/1605.05396) §3.3.
On top of the usual real and fake pairs, D is also trained to call
the pair (real_image, *wrong*_label) "fake":

```
D(real,    correct_label) -> 1   (smoothed to 0.9)
D(fake,    fake_label)    -> 0
D(real,    *wrong*_label) -> 0   <-- new term
```

This forces the gradient that flows back into G to actually depend on
the class label. The wrong label is generated as
`y' = (y + Uniform[1, N_CLASSES)) mod N`, which guarantees `y' ≠ y`
without rejection sampling.

**Classical cGAN, before vs. after match-aware D** (5 seeds × 2000 iters):

| Config | mean acc | min acc | max acc |
| --- | ---: | ---: | ---: |
| Pass-1 winner (smoothing + matched fakes) | 0.905 | 0.812 | 0.988 |
| **+ matching-aware D loss** | **1.000** | **1.000** | **1.000** |

The match-aware term reproducibly drives the classical conditional
generator to **100 % judge accuracy on every seed**. The same loss is
applied in the quantum notebook for the same reason — at the
quantum's 500-iter budget the loss curves show D held near 1.7 (vs.
~0.5 without match-aware) and the class embedding norm grows to
roughly 2.0 instead of staying near its initialisation. The remaining
quantum-side limit at 500 iters is the slow rate of parameter-shift
gradients on 170 parameters, not the conditioning signal.

Other knobs were ablated and discarded: fresh latents for the G step
(no gain), equal Adam LRs for G and D (D collapsed), 5000 classical
iterations (D over-trains), aggressive label smoothing (0.7) and
class-embedding learning-rate up-weighting (no consistent gain on
quantum).

### Pass 3 — architectural changes for the quantum cGAN

Pass 2 closed the gap on the classical cGAN (100% judge accuracy
across all five seeds) but the quantum cGAN's confusion matrix still
collapsed most requested classes onto digit `8`. After exhausting
the non-architectural lever space — strengthening label smoothing,
lowering D's learning rate, R1 gradient penalty, fresh fakes for G —
none materially shifted the accuracy structure: each lever just
swapped which classes were modelled. The remaining wall is the
generator's 170-parameter capacity for ten distinct digit shapes,
38× smaller than the discriminator.

`quantum_gan.ipynb` therefore keeps the Pass-2 model and scoreboard
intact for comparison, then introduces a **second, architecturally
upgraded model** in §8 with two changes:

1. **AC-GAN style discriminator**
   ([Odena, Olah & Shlens 2017](https://arxiv.org/abs/1610.09585)).
   D no longer receives the class label as input; instead a second
   linear head predicts the class as supervision (cross-entropy on
   both real and fake samples). The matching-aware mismatch loss is
   removed — the auxiliary class head replaces it as the conditional
   pressure on D, and a class-supervised gradient now reaches G on
   every step. The new module
   `discriminator.ConditionalDiscriminatorAC` lives next to the
   original `ConditionalDiscriminator`, which is unchanged.
2. **Larger patch quantum generator** — `q_depth` raised from 6 to
   10, so each sub-generator carries 50 weights instead of 30. Total
   trainable parameters rise from 170 to 250. The qubit topology,
   ancilla count and patch layout are unchanged, so this is the
   cheapest single capacity knob; per-iteration wall-clock cost
   scales roughly linearly with the parameter count under the
   parameter-shift gradient.
3. **Discriminator class-head pre-training (bootstrap).** Before
   adversarial training begins, D's class head is trained for 200
   mini-batches of plain supervised cross-entropy on real labelled
   digits at `lr = 5e-3`. This converges the head from random
   `ln 10 ≈ 2.30` CE down to ~0.25 (≈ 90 % top-1 accuracy on real
   digits) in under a second of wall-clock and breaks the AC-GAN
   deadlock that otherwise leaves both `D cls` and `G cls` near
   their random baselines for the first few hundred adversarial
   iters. A separate Adam optimiser is used for the pretrain step so
   the adversarial optimiser starts with fresh momentum on the
   warm-started parameters.

| Model | Generator params | Discriminator params |
| --- | ---: | ---: |
| Pass-2 quantum cGAN | 170 (120 sub-gen + 50 class emb) | 6 401 |
| **Pass-3 quantum cGAN** | **250** (200 sub-gen + 50 class emb) | **5 387** |
| Standard classical cGAN | ~9 888 | 6 401 |
| Parameter-matched classical cGAN | ~189 | 6 401 |

The Pass-3 D loss has two terms (BCE + λ·CE with `λ = 5.0`,
heavier than the textbook `1.0` because D's small trunk needs the
class signal to dominate) and the G loss similarly. All other
Pass-2 settings carry over unchanged (Adam TTUR, label smoothing
0.9, matched fake labels, `cond_init_scale = 0.5`, 500 iters,
batch size 8). The notebook saves a separate `quantum_gan_ac.pt`
checkpoint and renders a side-by-side scoreboard plus per-class
accuracy / FID bar charts so the architectural delta is visible.

### Final recipes

**Classical cGAN** (`classical_gan.ipynb`):
- Adam, `lr = 2e-4`, `β₁ = 0.5`
- 2000 iterations
- One-sided label smoothing (real targets = 0.9, G aims at 1.0)
- Matched fake labels
- Matching-aware D loss

**Quantum cGAN — Pass 2** (`quantum_gan.ipynb` §1–§7, baseline architecture):
- Adam TTUR, `lrG = 1e-2`, `lrD = 1e-3`, `β₁ = 0.5`
- 500 iterations, batch size 8
- `cond_init_scale = 0.5` for the class embedding (default 0.1 is dwarfed by `z ~ U[0, π/2]`)
- One-sided label smoothing (real targets = 0.9, G aims at 1.0)
- Matched fake labels
- Matching-aware D loss

**Quantum cGAN — Pass 3** (`quantum_gan.ipynb` §8, architectural upgrade):
- Same optimiser / smoothing / matched-labels recipe as Pass 2
- `q_depth = 10` instead of 6 (sub-generator weights 30 → 50)
- `ConditionalDiscriminatorAC` (no label as input, 10-way class head)
- Pass-2's matching-aware mismatch loss is replaced by AC-GAN's
  cross-entropy class loss (`λ = 5.0`)
- D class-head bootstrap: 200 supervised-CE iters at `lr = 5e-3`
  on real labelled digits before adversarial training (separate
  optimiser; G untouched)

## Quantitative evaluation

`metrics.py` implements the full suite. Every metric works for any
generator that exposes `forward(z, y) -> (B, 64)`, plumbed through a
generic `sample_inputs(n) -> (z, y)` callable.

| Metric | What it measures | Notes |
| --- | --- | --- |
| **Mode-collapse grid (10×10)** | One row per requested class, ten samples per row | Visual sanity check |
| **Latent-space interpolation** | Smoothness of the latent manifold *within a class* | Class held constant, `z` interpolated |
| **Pixel-level diversity** | Mean pairwise L2 + per-pixel std | Coarse intra-batch diversity |
| **Judge confusion matrix** | Random-Forest 10-class judge predictions vs. the requested class | Diagonal = per-class accuracy |
| **Per-class accuracy** | Diagonal of the confusion matrix | Higher = better class fidelity |
| **Judge inception score** | `exp(E[KL(p(y\|x) \|\| p(y))])` from the judge | Now meaningful (multiple classes) |
| **Domain-specific FID — overall** | Fréchet distance in `DigitFeatureExtractor` features, real-all vs fake-all | Lower = better |
| **Domain-specific FID — per class** | Same metric, real-class-c vs fake-class-c, for each `c` | Per-class quality breakdown |
| **Standard Inception Score** | ImageNet-Inception based; ~1.0 on 8×8 digits | Reported for completeness |
| **Generation speed** | Samples per second | Includes label-embedding lookup |
| **Quality vs. parameters / training time / speed** | Cross-model scatter and bar charts | Plotted in §9.7 of the classical notebook |

## References

1. Goodfellow et al., *Generative Adversarial Networks*,
   [arXiv:1406.2661](https://arxiv.org/abs/1406.2661) (2014).
2. Mirza & Osindero, *Conditional Generative Adversarial Nets*,
   [arXiv:1411.1784](https://arxiv.org/abs/1411.1784) (2014).
3. Huang et al., *Experimental Quantum Generative Adversarial Networks
   for Image Generation*,
   [arXiv:2010.06201](https://arxiv.org/abs/2010.06201) (2020).
