import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List

basic_dims = 8
num_modals = 4

#------------------------------
# Wavelet-like helpers 
#------------------------------

def gap_3d(x: torch.Tensor) -> torch.Tensor:
    """ x: (B, C, H, W, D) --> (B, C) """
    return x.mean(dim=(2,3,4))

def wavelet_like_split_x5(x5: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x5: (B, C, h, w, d)
    LL: avgpool3d(x5, 2) --> (B, C, h/2, w/2, d/2)
    HF: x5 - upsample(LL) --> (B, C, h, w, d)
    """
    ll = F.avg_pool3d(x5, kernel_size=2, stride=2)
    ll_up = F.interpolate(ll, scale_factor=2, mode="trilinear", align_corners=True)
    hf = x5 - ll_up
    return ll, hf

#------------------------------
# Base UNet blocks
#------------------------------

class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch, norm="in"):
        super().__init__()
        Norm = nn.InstanceNorm3d if norm == "in" else nn.BatchNorm3d
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            Norm(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            Norm(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class UNetEncoder3D(nn.Module):
    """
    Returns x1...x5
    x1: (B, base, H, W, D)
    x2: (B, base*2, H/2, W/2, D/2)
    x3: (B, base*4, H/4, W/4, D/4)
    x4: (B, base*8, H/8, W/8, D/8)
    x5: (B, base*16, H/16, W/16, D/16)
    """
    def __init__(self, in_ch=1, base=8, norm="in"):
        super().__init__()
        self.enc1=DoubleConv3D(in_ch, base, norm=norm)
        self.down1 = nn.MaxPool3d(2)

        self.enc2=DoubleConv3D(base, base*2, norm=norm)
        self.down2 = nn.MaxPool3d(2)

        self.enc3=DoubleConv3D(base*2, base*4, norm=norm)
        self.down3 = nn.MaxPool3d(2)

        self.enc4=DoubleConv3D(base*4, base*8, norm=norm)
        self.down4 = nn.MaxPool3d(2)

        self.enc5=DoubleConv3D(base*8, base*16, norm=norm)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.down1(x1))
        x3 = self.enc3(self.down2(x2))
        x4 = self.enc4(self.down3(x3))
        x5 = self.enc5(self.down4(x4))
        return x1, x2, x3, x4, x5

class SharedDecoder(nn.Module):
    """
    UNet-like decoder that expects fused skips at each level (x1...x5)
    """
    def __init__(self, num_cls=4):
        super().__init__()
        self.up4 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.dec4 = nn.Sequential(
            nn.Conv3d(128 + 64, 64, 3, padding=1, bias=False),
            nn.InstanceNorm3d(64, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(64, 64, 3, padding=1, bias=False),
            nn.InstanceNorm3d(64, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec3 = nn.Sequential(
            nn.Conv3d(64 + 32, 32, 3, padding=1, bias=False),
            nn.InstanceNorm3d(32, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(32, 32, 3, padding=1, bias=False),
            nn.InstanceNorm3d(32, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec2 = nn.Sequential(
            nn.Conv3d(32 + 16, 16, 3, padding=1, bias=False),
            nn.InstanceNorm3d(16, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(16, 16, 3, padding=1, bias=False),
            nn.InstanceNorm3d(16, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec1 = nn.Sequential(
            nn.Conv3d(16 + 8, 8, 3, padding=1, bias=False),
            nn.InstanceNorm3d(8, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(8, 8, 3, padding=1, bias=False),
            nn.InstanceNorm3d(8, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.seg = nn.Conv3d(8, num_cls, kernel_size=1)

    def forward(self, x1, x2, x3, x4, x5):
        d4 = self.up4(x5)
        d4 = self.dec4(torch.cat([d4, x4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, x3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, x2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, x1], dim=1))

        return self.seg(d1)

#------------------------------
# Memory Bank (per modality queue) storing keys + 2*val
#------------------------------

class ModalityMemoryBank(nn.Module):
    """
    Per-modality FIFO memory storing:
    keys: (K, N, key_dim)
    vals: (K, N, val_dim) where val_dim = 2*c5 (LL vector + HF vector)
    """
    def __init__(self, num_modals: int, mem_size: int, key_dim: int, val_dim: int):
        super().__init__()
        self.num_modals = num_modals
        self.mem_size = mem_size
        self.key_dim = key_dim
        self.val_dim = val_dim

        self.register_buffer("keys", torch.zeros(num_modals, mem_size, key_dim))
        self.register_buffer("vals", torch.zeros(num_modals, mem_size, val_dim))
        self.register_buffer("ptr", torch.zeros(num_modals, dtype=torch.long))
        self.register_buffer("filled", torch.zeros(num_modals, dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, modal_idx: int, k: torch.Tensor, v: torch.Tensor):
        """
        modal_idx: int
        k: (B, key_dim)
        v: (B, val_dim)
        """
        assert k.dim() ==2 and v.dim() == 2
        B = k.size(0)
        assert k.size(1) == self.key_dim and v.size(1) == self.val_dim

        k = k.detach()
        v = v.detach()

        ptr = int(self.ptr[modal_idx].item())
        for b in range(B):
            self.keys[modal_idx, ptr] = k[b]
            self.vals[modal_idx, ptr] = v[b]
            ptr = (ptr + 1) % self.mem_size

        self.ptr[modal_idx] = ptr
        self.filled[modal_idx] = torch.clamp(self.filled[modal_idx] + B, max=self.mem_size)

    def get_bank(self, modal_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            keys_m: (N_valid, key_dim)
            vals_m: (N_valid, val_dim)
        """
        n = int(self.filled[modal_idx].item())
        if n <= 0:
            return self.keys.new_zeros((0, self.key_dim)), self.vals.new_zeros((0, self.val_dim))
        return self.keys[modal_idx, :n], self.vals[modal_idx, :n]

#---------------------------------
# Retriever
#---------------------------------

class MemoryRetriever(nn.Module):
    """
    Dot-product retrieval with softmax attention.
    """
    def __init__(self, temperature: float=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, q: torch.Tensor, keys_m: torch.Tensor, vals_m: torch.Tensor):
        """
        q: (B, key_dim)
        keys_m: (N, key_dim)
        vals_m: (N, val_dim)
        Returns:
            v_hat: (B, val_dim)
            attn: (B, N)
        """
        B, key_dim = q.shape
        N = keys_m.shape[0]
        if N == 0:
            return q.new_zeros((B, vals_m.shape[1] if vals_m.numel() > 0 else 0)), None

        keys_m = F.normalize(keys_m, dim=-1)
        logits = (q @ keys_m.t()) / max(self.temperature, 1e-6)
        attn = F.softmax(logits, dim=-1)
        v_hat = attn @ vals_m
        return v_hat, attn

#--------------------------------
# Wavelet-aware Query Builder (LL + HF context)
#---------------------------------

class ContextQueryBuilderWavelet(nn.Module):
    """
    Build q from AVAILABLE modalities using both:
    ctx_LL = avg over modalities of GAP(LL)
    ctx_HF = avg over modalities of GAP(|HF|)
    """
    def __init__(self, c5: int, key_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2 * c5, 2 * c5, bias= False),
            nn.ReLU(inplace=True),
            nn.Linear(2 * c5, key_dim, bias= False),
        )

    def forward(self, x5_stack: torch.Tensor, mask: torch.Tensor):
        """
        x5_stack: (B, K, C, h, w, d)
        mask: (B, K) bool (true=present)
        """
        if mask.dtype != torch.bool:
            mask = mask.bool()
        B, K, C, h, w, d = x5_stack.shape

        ll_list, hf_list = [], []
        for m in range(K):
            ll, hf = wavelet_like_split_x5(x5_stack[:, m])
            ll_list.append(gap_3d(ll))          # (B, C)
            hf_list.append(gap_3d(hf.abs()))    # (B, C)

        ll = torch.stack(ll_list, dim=1) # (B, K, C)
        hf = torch.stack(hf_list, dim=1) # (B, K, C)

        ll = ll * mask[:, :, None].to(ll.dtype)
        hf = hf * mask[:, :, None].to(hf.dtype)

        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(ll.dtype)
        ctx_ll = ll.sum(dim=1) / denom
        ctx_hf = hf.sum(dim=1) / denom

        ctx = torch.cat([ctx_ll, ctx_hf], dim=1) #(B, 2C)
        q = self.fc(ctx)
        q = F.normalize(q, dim=-1)
        return q

#---------------------------------
# Vector <--> Map decoders for wavelet components
#---------------------------------

class ValToMap(nn.Module):
    """
    Convert Vector (B,C) --> map (B, C, h, w, d) using MLP+reshape.
    Good because x5 is tiny (e.g., 8^3).
    """
    def __init__(self, C: int, out_shape: Tuple[int, int, int], hidden_multi: int=2):
        super().__init__()
        h, w, d = out_shape
        self.C = C
        self.hwd = (h, w, d)
        self.mlp = nn.Sequential(
            nn.Linear(C, C * hidden_multi),
            nn.ReLU(inplace=True),
            nn.Linear(C * hidden_multi, C * h * w * d),
        )
        self.post = nn.Sequential(
            nn.Conv3d(C, C, kernel_size=1, bias=False),
            nn.InstanceNorm3d(C, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, v: torch.Tensor):
        B, C = v.shape
        assert C == self.C
        h, w, d = self.hwd
        y = self.mlp(v).view(B, C, h, w, d)
        y = self.post(y)
        return y

#---------------------------------
# Spatially-adapted fusion (MoE gating) for x5/x4
#---------------------------------

class SpatialGatedModalFuse3D(nn.Module):
    """
    Spatial fusion gates at low_res:
    s: (B, K, C, H, W, D)
    node_type: (B, K) 1=real, 0=retrieved
    conf: (B, K) confidence in [0,1] (real=1, retrieved=from memory attn)

    Output:
        fused: (B, C, H, W, D)
        gates: (B, K, H, W, D) (for debugging/visualization)
    """

    def __init__(self, channels: int, hidden: int=32, dropout: float=0.0):
        super().__init__()
        #inputs per modality: C + 2 meta channels (type, conf)
        in_ch = channels + 2
        self.gate_net = nn.Sequential(
            nn.Conv3d(in_ch, hidden, kernel_size=1, bias=False),
            nn.InstanceNorm3d(hidden, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(hidden, 1, kernel_size=1, bias=True) #gate logit per modality
        )

    @staticmethod
    def masked_softmax_logits(logits: torch.Tensor, mask: torch.Tensor, dim: int):
        """
        logits: (..., K, ...) but softmax over K dimension.
        mask: broadcastable bool mask with K at same dim position.
        """
        if mask.dtype != torch.bool:
            mask = mask.bool()
        neg = torch.finfo(logits.dtype).min     # if a modality is "not allowed", it sets its logits to -inf, so softmax gives it -0 weight
        logits = logits.masked_fill(~mask, neg)
        probs = F.softmax(logits, dim=dim)      # softmax gives weight that sum to 1 across modalities
        probs = probs * mask.to(probs.dtype)    # multiple by mask zeros out forbidden modalities
        denom = probs.sum(dim=dim, keepdim=True).clamp_min(1e-12)   # re-normalize so the remaining sum is still 1
        return probs / denom

    def forward(self, s: torch.Tensor, mask: torch.Tensor, node_type: torch.Tensor, conf: torch.Tensor):
        """
        s: (B, K, C, H, W, D)
        mask: (B, K) bool --> which nodes exists (after filling you can set all True)
        node_type: (B, K) long {0,1} --> tells the fusion module which modality features are real(1) vs reconstructed(0)
        conf: (B, K) float in [0,1] --> tells fusion how confident the reconstruction is (real=1, recon=retrieval max-attn)
        """

        B, K, C, H, W, D = s.shape
        if mask.dtype != torch.bool:
            mask = mask.bool()
        if node_type.dtype != torch.long:
            node_type = node_type.long()
        if conf.dtype != torch.float32 and conf.dtype != torch.float16 and conf.dtype != torch.bfloat16:
            conf = conf.float()

        #Build per-modality gate logits
        gate_logits = []
        for m in range(K):
            xm = s[:, m] # (B, C, H, W, D)

            #meta channels as maps
            tm = node_type[:, m].float().view(B, 1,1,1,1).expand(B,1,H,W,D)
            cm = conf[:, m].float().view(B, 1,1,1,1).expand(B,1,H,W,D)

            inp = torch.cat([xm, tm, cm], dim=1) # (B, C+2, H, W, D)
            lm = self.gate_net(inp)
            gate_logits.append(lm)

        gate_logits = torch.cat(gate_logits, dim=1) #(B, K, H, W, D)

        #Masked softmax over K
        mask_map = mask[:, :, None, None, None].expand(B, K, H, W, D)
        gates = self.masked_softmax_logits(gate_logits, mask_map, dim=1) # (B, K, H, W, D)

        fused = (s * gates[:, :, None, :, :, :]).sum(dim=1) #(B, C, H, W, D)
        # fused = (s * gates[:, :, None, :, :, :]).sum(dim=1)

        return fused, gates

#---------------------------------
#Full Model: Wavelet Memory + Spatial Fusion at x5/x4
#---------------------------------

class MultiModalUNet_WaveletMem_SpatialFuse(nn.Module):
    """
    key_design:
    - modality-specific encoders (4)
    - wavelet-like x5 split --> memory stores [v_ll, v_HF]
    - query uses LL+HF context from available modalities
    - retrieve missing --> decoder LL-map + HF-map --> reconstruct x5
    - generate x4..x1 from x5 for missing samples (same hallucination path)
    - fusion:
        x5, x4 use spatial gating (MoE)
        x3, x2, x1 use simple mean (cheap & stable)
    """

    def __init__(self,
                 num_cls: int=4,
                 base: int= basic_dims,
                 key_dim: int=128,
                 mem_size: int=1024,
                 temperature: float=0.07,
                 dropout: float=0.0,
                 lambda_hf_energy: float=0.1,   #HF-energy consistency loss when missin
                 lambda_use: float=0.0,         #optional: encourage some usage of retrieved (see below)
                 tau_use: float=0.10,           #threshold for usage regularizer (if lambda_use>0)
                 update_memory: bool=True,
                 ):
        super().__init__()
        self.update_memory = update_memory
        self.lambda_hf_energy = lambda_hf_energy
        self.lambda_use = lambda_use
        self.tau_use = tau_use

        #encoders
        self.flair_encoder = UNetEncoder3D(in_ch=1, base=base)
        self.t1ce_encoder = UNetEncoder3D(in_ch=1, base=base)
        self.t1_encoder = UNetEncoder3D(in_ch=1, base=base)
        self.t2_encoder = UNetEncoder3D(in_ch=1, base=base)

        #scale channels
        self.c1, self.c2, self.c3, self.c4, self.c5 = base, base*2, base*4, base*8, base*16

        #memory + retrieval
        self.query_builder = ContextQueryBuilderWavelet(c5=self.c5, key_dim=key_dim)
        self.retriever = MemoryRetriever(temperature=temperature)
        self.memory = ModalityMemoryBank(num_modals=num_modals, mem_size=mem_size, key_dim=key_dim, val_dim=2*self.c5)

        #lazy init for LL/HF decoders
        self._val_to_ll5 = None #(B, c5, h/2, w/2, d/2)
        self._val_to_hf5 = None #(B, c5, h, w, d)

        #hallucination path from x5 --> x4...x1
        self.gen4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(self.c5, self.c4, 3, padding=1, bias=False),
            nn.InstanceNorm3d(self.c4, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.gen3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(self.c4, self.c3, 3, padding=1, bias=False),
            nn.InstanceNorm3d(self.c3, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.gen2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(self.c3, self.c2, 3, padding=1, bias=False),
            nn.InstanceNorm3d(self.c2, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.gen1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(self.c2, self.c1, 3, padding=1, bias=False),
            nn.InstanceNorm3d(self.c1, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

        #spatial fusion at x5 and x4
        self.fuse5 = SpatialGatedModalFuse3D(channels=self.c5, hidden=32, dropout=dropout)
        self.fuse4 = SpatialGatedModalFuse3D(channels=self.c4, hidden=32, dropout=dropout)

        self.decoder = SharedDecoder(num_cls=num_cls)
        self.softmax = nn.Softmax(dim=1)

    def _init_val_to_maps_if_needed(self, x5_map: torch.Tensor):
        if (self._val_to_ll5 is None) or (self._val_to_hf5 is None):
            _, _, h, w, d = x5_map.shape
            assert (h % 2 == 0) and (w % 2 ==0) and (d % 2 == 0), "x5 dims must be divisible by 2 for LL split."

            self._val_to_ll5 = ValToMap(C=self.c5, out_shape=(h//2, w//2, d//2)).to(x5_map.device)
            self._val_to_hf5 = ValToMap(C=self.c5, out_shape=(h,w,d)).to(x5_map.device)

    @staticmethod
    def _x5_to_memvals(x5_map: torch.Tensor) -> torch.Tensor:
        """
        Returns val vector (B, 2c5) where:
            v_ll = GAP(LL)
            v_hf = GAP(|HF|)
        """
        ll, hf = wavelet_like_split_x5(x5_map)
        v_ll = gap_3d(ll)
        v_hf = gap_3d(hf.abs())
        return torch.cat([v_ll, v_hf], dim=1)

    @torch.no_grad()
    def _memory_update_from_present(self, q: torch.Tensor, s5_real: torch.Tensor, mask: torch.Tensor):
        """
        q: (B, key_dim)
        s5_real: (B, K, c5, h, w, d)
        mask: (B,K) bool
        """
        B, K, C, h, w, d = s5_real.shape
        for m in range(K):
            present = mask[:, m]
            if present.any():
                vals = self._x5_to_memvals(s5_real[present,m]) # (Bp, 2c5)
                self.memory.enqueue(m, q[present], vals)

    @staticmethod
    def _attn_confidence(attn: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        attn: (B, N)
        Return confidence in [0,1]. Cheap choice: max probability.
        """
        if attn is None:
            return None
        return attn.max(dim=1).values.clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, return_aux: bool=True):
        """
        x: (B, 4, H, W, D)
        mask: (B, 4) bool (True=present)
        """
        if mask.dtype != torch.bool:
            mask = mask.bool()

        #----- Encode -----
        flair_x1, flair_x2, flair_x3, flair_x4, flair_x5 = self.flair_encoder(x[:, 0:1])
        t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5 = self.t1ce_encoder(x[:, 1:2])
        t1_x1, t1_x2, t1_x3, t1_x4, t1_x5 = self.t1_encoder(x[:, 2:3])
        t2_x1, t2_x2, t2_x3, t2_x4, t2_x5 = self.t2_encoder(x[:, 3:4])

        # Stack real features
        s1_real = torch.stack([flair_x1, t1ce_x1, t1_x1, t2_x1], dim=1)
        s2_real = torch.stack([flair_x2, t1ce_x2, t1_x2, t2_x2], dim=1)
        s3_real = torch.stack([flair_x3, t1ce_x3, t1_x3, t2_x3], dim=1)
        s4_real = torch.stack([flair_x4, t1ce_x4, t1_x4, t2_x4], dim=1)
        s5_real = torch.stack([flair_x5, t1ce_x5, t1_x5, t2_x5], dim=1)  # (B,K,c5,h,w,d)

        # ----- Build wavelet-aware query from AVAILABLE MODALITIES -----
        q = self.query_builder(s5_real, mask) # (B, key_dim)

        #----- Update memory using present modalities -----
        if self.training and self.update_memory:
            self._memory_update_from_present(q, s5_real, mask)

        # ----- Init LL/ HF decoders once we know x5 size -----
        self._init_val_to_maps_if_needed(flair_x5)

        #----- Reconstruct x5 for missing modalities; track confidence -----
        real_x5_maps = [flair_x5, t1ce_x5, t1_x5, t2_x5]
        x5_hat_list: List[torch.Tensor] = []
        conf = torch.ones_like(mask, dtype=torch.float32, device=mask.device) # (B, K), real=1 by default

        hf_energy_loss = x.new_zeros(()) #scalar
        hf_loss_count = 0
        attn_debug: Dict[str, torch.Tensor] = {}

        for m in range(num_modals):
            present_m = mask[:, m]
            x5_m = real_x5_maps[m].clone()

            missing_idx = (~present_m).nonzero(as_tuple=True)[0]
            if missing_idx.numel() > 0:
                keys_m, vals_m = self.memory.get_bank(m) # vals: (N, 2c5)
                v_hat, attn = self.retriever(q[missing_idx], keys_m, vals_m)

                if v_hat.numel() == 0:
                    v_hat = x.new_zeros((missing_idx.numel(), 2*self.c5))
                    #confidence stays low-ish
                    conf[missing_idx, m] = 0.0
                else:
                    c = self._attn_confidence(attn)
                    if c is not None:
                        conf[missing_idx, m] = c

                v_ll_hat = v_hat[:, :self.c5]
                v_hf_hat = v_hat[:, self.c5:]

                ll_hat_map = self._val_to_ll5(v_ll_hat) # (Bm, c5, h/2, w/2, d/2)
                ll_hat_up = F.interpolate(ll_hat_map, scale_factor=2, mode='trilinear', align_corners=True)
                hf_hat_map = self._val_to_hf5(v_hf_hat) #(Bm, c5, h, w, d)
                x5_recon = ll_hat_up + hf_hat_map
                x5_recon = x5_recon.to(dtype=x5_m.dtype)  # <<< add this
                x5_m[missing_idx] = x5_recon

                #HF-energy loss (only when missing, training)
                if self.training and self.lambda_hf_energy > 0:
                    _, hf_true = wavelet_like_split_x5(real_x5_maps[m][missing_idx])
                    _, hf_pred = wavelet_like_split_x5(x5_recon)
                    e_true = gap_3d(hf_true.abs())
                    e_pred = gap_3d(hf_pred.abs())
                    hf_energy_loss = hf_energy_loss + F.l1_loss(e_pred, e_true, reduction="mean")
                    hf_loss_count += 1

                if attn is not None:
                    attn_debug[f"attn_m{m}"] = attn.detach()

            x5_hat_list.append(x5_m)

        if hf_loss_count > 0:
            hf_energy_loss = hf_energy_loss / float(hf_loss_count)

        #------ Generate x4...x1 for mising samples from reconstructed x5 -----
        real_x4_maps = [flair_x4, t1ce_x4, t1_x4, t2_x4]
        real_x3_maps = [flair_x3, t1ce_x3, t1_x3, t2_x3]
        real_x2_maps = [flair_x2, t1ce_x2, t1_x2, t2_x2]
        real_x1_maps = [flair_x1, t1ce_x1, t1_x1, t2_x1]

        x4_hat_list, x3_hat_list, x2_hat_list, x1_hat_list = [], [], [], []

        for m in range(num_modals):
            present_m = mask[:, m]
            x5_m = x5_hat_list[m]

            x4_m = real_x4_maps[m].clone()
            x3_m = real_x3_maps[m].clone()
            x2_m = real_x2_maps[m].clone()
            x1_m = real_x1_maps[m].clone()

            missing_idx = (~present_m).nonzero(as_tuple=True)[0]
            if missing_idx.numel() > 0:
                g4 = self.gen4(x5_m[missing_idx])
                g3 = self.gen3(g4)
                g2 = self.gen2(g3)
                g1 = self.gen1(g2)

                x4_m[missing_idx] = g4
                x3_m[missing_idx] = g3
                x2_m[missing_idx] = g2
                x1_m[missing_idx] = g1

            x4_hat_list.append(x4_m)
            x3_hat_list.append(x3_m)
            x2_hat_list.append(x2_m)
            x1_hat_list.append(x1_m)

        # ---- Stack for fusion ----
        s5 = torch.stack(x5_hat_list, dim=1)  # (B,K,c5,h,w,d)
        s4 = torch.stack(x4_hat_list, dim=1)  # (B,K,c4,16,16,16) typically
        s3 = torch.stack(x3_hat_list, dim=1)
        s2 = torch.stack(x2_hat_list, dim=1)
        s1 = torch.stack(x1_hat_list, dim=1)

        # Node type: 1=real, 0=retrieved
        node_type = mask.long()
        # After filling, we consider all nodes "available" for fusion
        mask_all = torch.ones_like(mask, dtype=torch.bool)

        # ---- Fusion strategy ----
        # x5 and x4: spatial gating (MoE)
        fx5, gates5 = self.fuse5(s5, mask_all, node_type=node_type, conf=conf)
        fx4, gates4 = self.fuse4(s4, mask_all, node_type=node_type, conf=conf)

        # x3,x2,x1: simple mean (stable & cheap) over modalities
        # (you can swap these for gating too, but compute grows)
        fx3 = s3.mean(dim=1)
        fx2 = s2.mean(dim=1)
        fx1 = s1.mean(dim=1)

        # ---- Decode ----
        logits = self.decoder(fx1, fx2, fx3, fx4, fx5)
        probs = self.softmax(logits)

        # Optional: utilization regularizer to prevent ignoring retrieved modalities entirely
        # This is mild and can be turned off (lambda_use=0).
        loss_use = x.new_zeros(())
        if self.training and self.lambda_use > 0:
            # average gate mass assigned to missing modalities at x5
            missing = (~mask).float()  # (B,K) 1 if missing
            # gate mass per modality: mean over spatial dims
            g5_mass = gates5.mean(dim=(2, 3, 4))  # (B,K)
            # only consider missing entries
            missing_vals = g5_mass[missing.bool()]
            if missing_vals.numel() > 0:
                loss_use = F.relu(self.tau_use - missing_vals).mean()

        if not return_aux:
            return probs

        aux = {
            "q": q.detach(),
            "conf": conf.detach(),  # (B,K)
            "gates4": gates4.detach(),  # (B,K,H,W,D) at x4
            "gates5": gates5.detach(),  # (B,K,H,W,D) at x5
            "hf_energy_loss": hf_energy_loss,  # scalar
            "use_loss": loss_use,  # scalar (maybe 0)
        }
        aux.update(attn_debug)
        return probs, aux







