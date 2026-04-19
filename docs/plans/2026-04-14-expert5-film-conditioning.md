# Expert 5 — FiLM Domain Conditioning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add FiLM domain conditioning to the Res-U-Net Autoencoder (Expert 5, slot index=5) so the decoder knows which of the 5 medical domains it is reconstructing, eliminating domain-confusion false positives in OOD detection.

**Architecture:** New classes `FiLMGenerator`, `FiLMLayer`, and `ConditionedBottleneck` are added to `expert6_resunet.py`. The main model is renamed `ConditionedResUNetAE` (with `ResUNetAutoencoder` kept as an alias). `forward` and `reconstruction_error` gain an optional `domain_ids` parameter (None → unknown domain id=5). All callers updated.

**Tech Stack:** PyTorch 2.x, FP32 strict, no AMP. Python `/home/mrsasayo_mesa/venv_global/bin/python`.

---

## Critical design decisions (read before touching code)

1. **`reconstruction_error` returns ONLY the error tensor `[B]`**, not a tuple. The old ResUNetAutoencoder returned `(error, z)` — a tuple. But `ood_detector.py` and `inference_engine.py` call it as if it returns a plain tensor. The new `ConditionedResUNetAE.reconstruction_error` must return `torch.Tensor[B]` only. This fixes a latent bug.

2. **`domain_ids` is optional everywhere.** Default `None` → the method creates `torch.full([B], 5, dtype=torch.long, device=x.device)` internally (domain id 5 = "Unknown"). This keeps all existing callers backward compatible.

3. **Skip_ch in decoder is already correct** in the existing code. Do NOT change the decoder block instantiation. The spec's new pseudocode has the wrong skip_ch values again (same bug as before); ignore those and keep the current values:
   - `dec4: DecoderBlock(ch*8, ch*8, ch*4)` ← in=512, skip=512, out=256 ✓
   - `dec3: DecoderBlock(ch*4, ch*8, ch*2)` ← in=256, skip=512, out=128 ✓
   - `dec2: DecoderBlock(ch*2, ch*4, ch)` ← in=128, skip=256, out=64 ✓
   - `dec1: DecoderBlock(ch, ch*2, ch//2)` ← in=64, skip=128, out=32 ✓

4. **FiLM initialization is critical.** The last Linear layer of `FiLMGenerator.mlp` must be zero-initialized (weights AND bias = 0). Combined with the residual formulation `gamma = 1.0 + raw_gamma`, this ensures at the start of training FiLM acts as identity (no modulation). The model learns deviations from identity.

5. **`ConditionedBottleneck` keeps dropout** (parameter `dropout: float = 0.15`) for consistency with the existing `Bottleneck`.

6. **`ResUNetAutoencoder` stays as a module-level alias** pointing to `ConditionedResUNetAE`. This avoids breaking any code that already imports `ResUNetAutoencoder`. Example at bottom of file: `ResUNetAutoencoder = ConditionedResUNetAE`.

7. **Parameter count**: FiLM adds ~140K params on top of the current ~50.61M → total ~50.75M. The test range `45_000_000 <= n <= 60_000_000` does NOT need to change.

---

## Task 1: Add FiLMGenerator, FiLMLayer, ConditionedBottleneck to `expert6_resunet.py`

**Files:**
- Modify: `src/pipeline/fase3/models/expert6_resunet.py`

**What to add** (insert between the existing `Bottleneck` class and `ResUNetAutoencoder` class):

```python
class FiLMGenerator(nn.Module):
    """Genera parámetros gamma y beta para FiLM conditioning.

    Mapea un domain ID (int) a vectores de escala (gamma) y sesgo (beta)
    para modular los feature maps del bottleneck.

    n_domains=6: 0=CXR, 1=ISIC, 2=Panorama, 3=LUNA16, 4=OA, 5=Unknown.

    Inicialización: el último Linear se inicializa a cero → gamma=1, beta=0
    al inicio del entrenamiento (identidad). El modelo aprende desviaciones
    desde la identidad, lo que garantiza estabilidad en las primeras épocas.

    Args:
        n_domains: número total de dominios (incluyendo "Unknown").
        embed_dim: dimensión del embedding de dominio.
        feature_channels: número de canales del feature map a modular.
    """

    def __init__(
        self,
        n_domains: int = 6,
        embed_dim: int = 64,
        feature_channels: int = 512,
    ) -> None:
        super().__init__()
        self.feature_channels = feature_channels
        self.domain_embed = nn.Embedding(n_domains, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_channels * 2),  # gamma y beta concatenados
        )
        # Inicialización crítica: gamma_raw=0, beta=0 → FiLM actúa como
        # identidad al inicio. El modelo aprende desviaciones desde 1 y 0.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self, domain_ids: torch.LongTensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Genera gamma y beta para los domain_ids dados.

        Args:
            domain_ids: [B] tensor int64 con IDs de dominio (0–5).

        Returns:
            Tupla (gamma [B, C, 1, 1], beta [B, C, 1, 1]) para broadcasting
            sobre feature maps [B, C, H, W].
        """
        e = self.domain_embed(domain_ids)   # [B, embed_dim]
        params = self.mlp(e)                # [B, 2*C]
        gamma_raw, beta = params.chunk(2, dim=-1)  # cada uno [B, C]
        # Formulación residual: base gamma=1 (identidad), aprende delta
        gamma = (1.0 + gamma_raw).view(-1, self.feature_channels, 1, 1)
        beta = beta.view(-1, self.feature_channels, 1, 1)
        return gamma, beta


class FiLMLayer(nn.Module):
    """Aplica FiLM modulation: x_cond = gamma * x + beta.

    Operación sin parámetros entrenables propios — los parámetros
    de modulación vienen de FiLMGenerator.
    """

    def forward(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: feature map [B, C, H, W].
            gamma: escala [B, C, 1, 1].
            beta: sesgo [B, C, 1, 1].

        Returns:
            Feature map modulado [B, C, H, W].
        """
        return gamma * x + beta


class ConditionedBottleneck(nn.Module):
    """Bottleneck con FiLM domain conditioning.

    Flujo:
      x [B,512,7,7] → ResBlocks×4 → FiLM(domain_ids) → feat_cond [B,512,7,7]
                                                        → GAP → z [B,512]

    El FiLM se aplica DESPUÉS de los bloques residuales, modulando el
    espacio latente espacial antes de producir el vector z y antes de
    pasarlo al decoder. Esto permite al decoder "saber" qué dominio
    debe reconstruir.

    Args:
        channels: canales del feature map (= dimensión del vector latente).
        n_res_blocks: cantidad de ResBlockPreAct en el bottleneck.
        dropout: dropout para cada ResBlockPreAct.
        n_domains: número de dominios para el FiLMGenerator.
        embed_dim: dimensión del embedding de dominio en FiLMGenerator.
    """

    def __init__(
        self,
        channels: int = 512,
        n_res_blocks: int = 4,
        dropout: float = 0.15,
        n_domains: int = 6,
        embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[ResBlockPreAct(channels, dropout) for _ in range(n_res_blocks)]
        )
        self.film_gen = FiLMGenerator(
            n_domains=n_domains,
            embed_dim=embed_dim,
            feature_channels=channels,
        )
        self.film = FiLMLayer()
        self.to_latent = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(
        self,
        x: torch.Tensor,
        domain_ids: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: feature map de entrada [B, channels, H, W].
            domain_ids: [B] tensor int64 con IDs de dominio.

        Returns:
            Tupla (feat_cond [B, channels, H, W], z [B, channels]).
        """
        feat = self.blocks(x)
        gamma, beta = self.film_gen(domain_ids)
        feat = self.film(feat, gamma, beta)
        z = self.to_latent(feat)
        return feat, z
```

**Then rename `ResUNetAutoencoder` → `ConditionedResUNetAE`** with these changes:
- Add `n_domains: int = 6` and `embed_dim: int = 64` parameters to `__init__`
- Replace `self.bottleneck = Bottleneck(...)` with `self.bottleneck = ConditionedBottleneck(channels=base_ch * 8, n_res_blocks=4, dropout=0.15, n_domains=n_domains, embed_dim=embed_dim)`
- Update `forward(self, x, domain_ids=None)`:
  - If `domain_ids is None`: create `domain_ids = torch.full((x.shape[0],), 5, dtype=torch.long, device=x.device)`
  - Call `b, z = self.bottleneck(b, domain_ids)` instead of `b, z = self.bottleneck(b)`
- Update `reconstruction_error(self, x, domain_ids=None)`:
  - Add `domain_ids=None` parameter
  - Call `self.forward(x, domain_ids)` instead of `self.forward(x)`
  - **Return only the error tensor `[B]`** (not a tuple). Remove `, z` from return.
- Keep `count_parameters()` unchanged

**After the class definition, add the alias:**
```python
# Backward-compat alias — code that imports ResUNetAutoencoder still works.
ResUNetAutoencoder = ConditionedResUNetAE
```

**Update `__main__` block** to test `ConditionedResUNetAE` with and without `domain_ids`, and verify `reconstruction_error` returns a tensor (not a tuple).

**Verification after this task:**
```bash
/home/mrsasayo_mesa/venv_global/bin/python src/pipeline/fase3/models/expert6_resunet.py
```
Expected output: shapes verified, param count ~50.75M, no errors.

---

## Task 2: Update `expert6_resunet_config.py`

**Files:**
- Modify: `src/pipeline/fase3/expert6_resunet_config.py`

Add these two constants in the `# ── Arquitectura` section, after `EXPERT6_EXPERT_ID`:

```python
EXPERT6_N_DOMAINS: int = 6
"""Número total de dominios para el FiLMGenerator.
Dominios 0-4: CXR14, ISIC, Panorama, LUNA16, OA Knee.
Dominio 5: 'Unknown' — usado en inferencia cuando el dominio no se conoce."""

EXPERT6_EMBED_DIM: int = 64
"""Dimensión del embedding de dominio en FiLMGenerator.
Valor pequeño (64) suficiente para distinguir 6 dominios sin sobreajustar."""
```

Also update the `EXPERT6_CONFIG_SUMMARY` string to mention FiLM conditioning.

**No verification needed** (just constants, no executable path).

---

## Task 3: Update `moe_model.py`

**Files:**
- Modify: `src/pipeline/fase5/moe_model.py`

**Changes:**

1. Update the import line:
   ```python
   # Before:
   from fase3.models.expert6_resunet import ResUNetAutoencoder
   # After:
   from fase3.models.expert6_resunet import ConditionedResUNetAE
   ```

2. Update instantiation in `build_moe_system_dry_run`:
   ```python
   # Before:
   ResUNetAutoencoder(in_ch=3, base_ch=64),  # Expert 5 → Res-U-Net v6 (OOD)
   # After:
   ConditionedResUNetAE(in_ch=3, base_ch=64, n_domains=6),  # Expert 5 → Res-U-Net v6 condicionado (OOD)
   ```

3. Update `MoESystem.forward` signature to accept optional `domain_ids`:
   ```python
   # Before:
   def forward(self, x: torch.Tensor, expert_id: int) -> dict:
   # After:
   def forward(self, x: torch.Tensor, expert_id: int,
               domain_ids: Optional[torch.LongTensor] = None) -> dict:
   ```

4. Update the `expert_id == 5` branch:
   ```python
   # Before:
   recon, z = expert(x)
   # After:
   recon, z = expert(x, domain_ids)
   # (domain_ids=None is fine — ConditionedResUNetAE defaults to unknown domain id=5)
   ```

5. Update the docstring to mention `domain_ids`.

**No standalone verification** — verified through tests in Task 4.

---

## Task 4: Update `tests/test_integration_experts.py` and run full suite

**Files:**
- Modify: `tests/test_integration_experts.py`

**Changes:**

1. Update import to add `ConditionedResUNetAE`:
   ```python
   # Before:
   from pipeline.fase3.models.expert6_resunet import ResUNetAutoencoder  # noqa: E402
   # After:
   from pipeline.fase3.models.expert6_resunet import ConditionedResUNetAE  # noqa: E402
   ```

2. In `TestExpert6ResUNetAutoencoder.test_forward_shape_and_values`:
   - Replace `ResUNetAutoencoder(in_ch=3, base_ch=64)` with `ConditionedResUNetAE(in_ch=3, base_ch=64, n_domains=6)`
   - Add `domain_ids = torch.zeros(BATCH_SIZE, dtype=torch.long, device=DEVICE)` before the forward call
   - Change `result = model(x)` to `result = model(x, domain_ids)`

3. In `TestCountParameters.test_expert5_count_parameters`:
   - Replace `ResUNetAutoencoder(in_ch=3, base_ch=64)` with `ConditionedResUNetAE(in_ch=3, base_ch=64)`
   - Range stays `45_000_000 <= n <= 60_000_000` (FiLM adds ~140K, stays in range)

4. In `TestMoESystemDryRun.test_manual_moe_assembly`:
   - Replace `ResUNetAutoencoder(in_ch=3, base_ch=64)` with `ConditionedResUNetAE(in_ch=3, base_ch=64, n_domains=6)`

**Run suite and confirm 14 passed:**
```bash
/home/mrsasayo_mesa/venv_global/bin/python -m pytest tests/test_integration_experts.py -v
```
Expected: **14 passed, 0 failed**.

---

## Final state after all tasks

| File | Change |
|---|---|
| `src/pipeline/fase3/models/expert6_resunet.py` | FiLMGenerator + FiLMLayer + ConditionedBottleneck added; ResUNetAutoencoder renamed to ConditionedResUNetAE; forward/reconstruction_error accept domain_ids; alias preserved |
| `src/pipeline/fase3/expert6_resunet_config.py` | EXPERT6_N_DOMAINS=6 and EXPERT6_EMBED_DIM=64 added |
| `src/pipeline/fase5/moe_model.py` | Import + instantiation updated; MoESystem.forward accepts domain_ids |
| `tests/test_integration_experts.py` | All references to ResUNetAutoencoder → ConditionedResUNetAE; domain_ids passed in tests |

**Not changed in this plan** (stale references, separate task):
- `src/pipeline/fase6/ood_detector.py` — calls `reconstruction_error(x)` without domain_ids; works because domain_ids=None defaults to id=5
- `src/pipeline/fase6/inference_engine.py` — same, backward compatible
