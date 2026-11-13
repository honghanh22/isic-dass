"""DASS-RC — chọn ảnh sinh theo margin mục tiêu (TMU) và độ phủ ảnh thật (RDC). numpy thuần.

Quy trình cho MỖI lớp c (thuyết minh DASS-RC, 13/11/2026):

1. NC (kiểm tra gần trùng, encoder độc lập Inception-v3):
       d_i   = min_{j≠i} ‖e(x_i) − e(x_j)‖           (ảnh thật train, bỏ chính nó)
       τ_NC  = Q_0.01({d_i})
   Ảnh sinh trùng pixel -> loại trực tiếp. Ảnh sinh có khoảng cách tới train < τ -> NGHI VẤN, chỉ loại sau khi
   xác minh (danh sách xác minh truyền vào từ ngoài). Trùng trong pool: giữ ID nhỏ nhất.
2. Gate:  m_c(q) ≥ δ_c (= 0); sigmoid một logit z: m_1 = z, m_0 = −z.
3. TMU:   m̄ = (m − a_c) / max(b_c, ε_m),  T = exp(−(m̄ − μ_c)² / 2σ_c²),
          a_c, b_c = trung bình / độ lệch chuẩn margin ảnh thật train lớp c; μ_c = phân vị của m̄ trên ảnh thật có m > 0.
4. RDC:   κ(x, q) = exp(−‖e(x) − e(q)‖² / 2h²),  h = η · max(ε_h, median khoảng cách cặp ảnh thật cùng lớp),
          C(U) = Σ_x w(x) · log(1 + Σ_{q∈U} κ(x, q)),  w = 1/N_c   (độ phủ LOG, hàm lõm của tổng kernel).
          Đơn điệu, submodular, lợi ích giảm dần nhưng không bão hoà. Không dùng (người dùng chọn, 13/11/2026):
          facility location Σ_x w·max_q κ — với K = N_c bão hoà sau ~60 % ngân sách (ISIC), tập chọn không phụ thuộc
          h; độ phủ xác suất Σ_x w·[1 − Π(1 − κ)] — lợi ích giảm theo cấp số nhân, ≈ 0 sau 13–29 ảnh.
5. Greedy: F(U) = (α/K)·Σ_{q∈U} u(q) + (1−α)·C(U); lazy greedy CHÍNH XÁC (F submodular, đơn điệu), phá hoà bằng
          chỉ số nhỏ hơn; ghi riêng phần tăng utility và coverage ở mỗi lượt.

Biến thể (cùng tập ứng viên qua NC + gate, cùng ngân sách K):
    RC_RANDOM        ngẫu nhiên
    RC_TMU           top-K theo T                              (α = 1)
    RC_RDC           greedy chỉ RDC                            (α = 0)
    RC_TMU_POOLCOV   TMU + coverage với mốc = chính tập ứng viên (cùng h)
    RC_MAXMARGIN_RDC u = rank thực nghiệm của margin trong [0, 1] + RDC
    RC_DASS          TMU + RDC (DASS-RC)
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

RC_RANDOM = "A0_random"
RC_TMU = "A1_tmu"
RC_RDC = "A2_rdc"
RC_TMU_POOLCOV = "A3_tmu_poolcov"
RC_MAXMARGIN_RDC = "A4_maxmargin_rdc"
RC_DASS = "A6_dass_rc"
RC_SELECTION_METHODS = (RC_RANDOM, RC_TMU, RC_RDC, RC_TMU_POOLCOV, RC_MAXMARGIN_RDC, RC_DASS)


# ----------------------------------------------------------------------------------------------- tiện ích khoảng cách
def l2n(x: np.ndarray) -> np.ndarray:
    """Chuẩn hoá L2 theo hàng; hàng có norm 0 hoặc không hữu hạn -> NaN (để bị gắn cờ, không âm thầm giữ)."""
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    bad = ~np.isfinite(n) | (n == 0)
    out = x / np.where(bad, 1.0, n)
    out[bad[:, 0]] = np.nan
    return out


def valid_rows(x: np.ndarray) -> np.ndarray:
    return np.isfinite(np.asarray(x)).all(axis=1)


def _sqdist_block(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T
    return np.maximum(d, 0.0)


def nearest(query: np.ndarray, ref: np.ndarray, exclude_self: bool = False,
            block: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Khoảng cách Euclid và chỉ số tới phần tử gần nhất trong `ref` (tính theo block để tiết kiệm RAM).

    exclude_self=True: `query` là `ref` (cùng thứ tự), bỏ cặp (i, i).
    """
    query = np.asarray(query, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    dist = np.empty(len(query))
    idx = np.empty(len(query), dtype=np.int64)
    for s in range(0, len(query), block):
        d = _sqdist_block(query[s:s + block], ref)
        if exclude_self:
            r = np.arange(s, min(s + block, len(query)))
            d[r - s, r] = np.inf
        j = np.argmin(d, axis=1)
        idx[s:s + block] = j
        dist[s:s + block] = np.sqrt(d[np.arange(len(j)), j])
    return dist, idx


# --------------------------------------------------------------------------------------------------------------- NC
def nc_threshold(real_feats: np.ndarray, quantile: float = 0.01) -> tuple[float, np.ndarray]:
    """τ_NC = Q_q của khoảng cách láng giềng gần nhất giữa các ảnh thật train (bỏ chính nó)."""
    d, _ = nearest(real_feats, real_feats, exclude_self=True)
    return float(np.quantile(d, quantile)), d


@dataclass
class NCResult:
    tau: float
    nn_dist: np.ndarray                 # khoảng cách mỗi ảnh sinh tới ảnh train gần nhất
    nn_index: np.ndarray                # chỉ số ảnh train gần nhất
    pixel_duplicate: np.ndarray         # trùng pixel với train -> loại trực tiếp
    suspect: np.ndarray                 # nn_dist < τ (chưa trùng pixel) -> cần xác minh
    pool_duplicate_of: np.ndarray       # −1 hoặc chỉ số ảnh sinh (ID nhỏ hơn) mà nó trùng trong pool
    reasons: list[str] = field(default_factory=list)

    def keep(self, confirmed_copies: set[int] | None = None) -> np.ndarray:
        """Mặt nạ a(q): loại trùng pixel, trùng trong pool và ảnh nghi vấn ĐÃ xác minh là bản sao."""
        confirmed = np.zeros(len(self.nn_dist), dtype=bool)
        if confirmed_copies:
            confirmed[sorted(confirmed_copies)] = True
        if (confirmed & ~(self.suspect | self.pixel_duplicate)).any():
            raise ValueError("Có ảnh được xác minh là bản sao nhưng không nằm trong danh sách nghi vấn.")
        return ~(self.pixel_duplicate | confirmed | (self.pool_duplicate_of >= 0))


def nc_check(pool_feats: np.ndarray, train_feats: np.ndarray, tau: float, pool_hashes: list[str] | None = None,
             train_hashes: list[str] | None = None) -> NCResult:
    """Kiểm tra gần trùng: so với TOÀN BỘ train của GAN (cả hai lớp) và trong nội bộ pool.

    Trùng trong pool = hash pixel bằng nhau, hoặc khoảng cách tới một ảnh sinh ID nhỏ hơn < τ.
    """
    n = len(pool_feats)
    dist, idx = nearest(pool_feats, train_feats)
    pix = np.zeros(n, dtype=bool)
    dup_of = np.full(n, -1, dtype=np.int64)
    if pool_hashes is not None:
        th = set(train_hashes or [])
        pix = np.array([h in th for h in pool_hashes], dtype=bool)
        first: dict[str, int] = {}
        for i, h in enumerate(pool_hashes):
            if h in first:
                dup_of[i] = first[h]
            else:
                first[h] = i
    # gần trùng trong pool: chỉ xét ảnh có ID nhỏ hơn -> ảnh đầu tiên của một cụm được giữ
    pf = np.asarray(pool_feats, dtype=np.float64)
    for s in range(0, n, 512):
        d = _sqdist_block(pf[s:s + 512], pf)
        for r in range(d.shape[0]):
            i = s + r
            if dup_of[i] >= 0 or i == 0:
                continue
            row = d[r, :i]
            j = int(np.argmin(row))
            if np.sqrt(row[j]) < tau:
                dup_of[i] = j
    suspect = (dist < tau) & ~pix
    return NCResult(tau=tau, nn_dist=dist, nn_index=idx, pixel_duplicate=pix, suspect=suspect,
                    pool_duplicate_of=dup_of)


# ---------------------------------------------------------------------------------------------------- margin + gate
def class_margin(logit: np.ndarray, cls: int) -> np.ndarray:
    """Sigmoid một logit z: m_1 = z, m_0 = −z."""
    z = np.asarray(logit, dtype=np.float64).reshape(-1)
    return z if cls == 1 else -z


def gate(margin: np.ndarray, delta: float = 0.0) -> np.ndarray:
    m = np.asarray(margin, dtype=np.float64)
    return np.isfinite(m) & (m >= delta)


# -------------------------------------------------------------------------------------------------------------- TMU
@dataclass(frozen=True)
class MarginStats:
    a: float
    b: float
    eps: float = 1e-6

    def normalize(self, m: np.ndarray) -> np.ndarray:
        return (np.asarray(m, dtype=np.float64) - self.a) / max(self.b, self.eps)


def margin_stats(real_margin: np.ndarray, eps: float = 1e-6) -> MarginStats:
    m = np.asarray(real_margin, dtype=np.float64)
    m = m[np.isfinite(m)]
    return MarginStats(float(m.mean()), float(m.std()), eps)


def tmu_center(real_margin: np.ndarray, stats: MarginStats, percentile: float) -> float:
    """μ_c = phân vị của margin chuẩn hoá trên ảnh thật train có m > 0."""
    m = np.asarray(real_margin, dtype=np.float64)
    pos = m[np.isfinite(m) & (m > 0)]
    if len(pos) == 0:
        raise ValueError("Không có ảnh thật nào đúng phía biên (m > 0): xem lại encoder trước khi áp TMU.")
    return float(np.percentile(stats.normalize(pos), percentile))


def tmu(margin: np.ndarray, stats: MarginStats, mu: float, sigma: float) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma phải > 0")
    mb = stats.normalize(margin)
    return np.exp(-((mb - mu) ** 2) / (2.0 * sigma ** 2))


def margin_rank(margin: np.ndarray) -> np.ndarray:
    """Rank thực nghiệm của margin trong tập ứng viên, về [0, 1] (margin lớn -> 1); hoà -> rank trung bình."""
    m = np.asarray(margin, dtype=np.float64)
    n = len(m)
    if n <= 1:
        return np.ones(n)
    order = np.argsort(m, kind="stable")
    ranks = np.empty(n)
    ranks[order] = np.arange(n, dtype=np.float64)
    for v in np.unique(m):                         # hoà -> cùng rank trung bình
        sel = m == v
        if sel.sum() > 1:
            ranks[sel] = ranks[sel].mean()
    return ranks / (n - 1)


# ---------------------------------------------------------------------------------------------------- kernel + RDC
def median_bandwidth(real_emb: np.ndarray, max_pairs: int = 10_000, seed: int = 0, eps: float = 1e-8) -> float:
    """h⁽⁰⁾ = max(ε_h, median ‖e_i − e_j‖) trên tối đa `max_pairs` cặp khác chỉ số (seed cố định)."""
    e = np.asarray(real_emb, dtype=np.float64)
    n = len(e)
    if n < 2:
        raise ValueError("Cần ít nhất 2 ảnh thật để tính bandwidth.")
    total = n * (n - 1) // 2
    if total <= max_pairs:
        i, j = np.triu_indices(n, k=1)
    else:
        rng = np.random.default_rng(seed)
        pairs: set[tuple[int, int]] = set()
        while len(pairs) < max_pairs:
            a = rng.integers(0, n, size=2 * max_pairs)
            b = rng.integers(0, n, size=2 * max_pairs)
            for x, y in zip(a, b):
                if x != y:
                    pairs.add((min(x, y), max(x, y)))
                    if len(pairs) == max_pairs:
                        break
        i, j = np.array(sorted(pairs)).T
    med = float(np.median(np.linalg.norm(e[i] - e[j], axis=1)))
    if med <= eps:
        log.warning("Median khoảng cách ≈ 0: đặc trưng suy biến, cần kiểm tra dữ liệu / encoder.")
    return max(eps, med)


def kernel_column(anchors: np.ndarray, q: np.ndarray, h: float) -> np.ndarray:
    d2 = np.maximum(((anchors - q[None, :]) ** 2).sum(1), 0.0)
    return np.exp(-d2 / (2.0 * h * h))


def coverage(anchors: np.ndarray, selected: np.ndarray, h: float, weights: np.ndarray | None = None) -> float:
    """C(U) = Σ_x w(x) · log(1 + Σ_{q∈U} κ(x, q)); C(∅) = 0. Đơn điệu, submodular, không bão hoà."""
    if len(selected) == 0:
        return 0.0
    w = np.full(len(anchors), 1.0 / len(anchors)) if weights is None else np.asarray(weights, dtype=np.float64)
    total = np.zeros(len(anchors))
    for q in np.asarray(selected, dtype=np.float64):
        total += kernel_column(anchors, q, h)
    return float(w @ np.log1p(total))


# ----------------------------------------------------------------------------------------------------------- greedy
@dataclass
class GreedyLog:
    order: list[int]
    gain_utility: list[float]
    gain_coverage: list[float]


def greedy_select(cand_emb: np.ndarray, utility: np.ndarray, anchors: np.ndarray, n_select: int, alpha: float,
                  h: float, weights: np.ndarray | None = None, block: int = 1024,
                  max_kernel_bytes: float = 16e9) -> GreedyLog:
    """Lazy greedy CHÍNH XÁC cho F(U) = (α/K)·Σ u(q) + (1−α)·C(U).

    F đơn điệu, submodular -> lợi ích biên của mỗi ứng viên chỉ giảm khi U lớn dần, nên chặn trên cũ trong heap
    là hợp lệ; ứng viên ở đỉnh heap được tính lại, nếu vẫn ≥ đỉnh tiếp theo (so tuple (−lợi ích, chỉ số)) thì chọn.
    Kết quả trùng greedy đầy đủ, phá hoà bằng chỉ số nhỏ hơn.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha phải trong [0, 1]")
    E = np.asarray(cand_emb, dtype=np.float64)
    A = np.asarray(anchors, dtype=np.float64)
    u = np.asarray(utility, dtype=np.float64)
    n = len(E)
    k = int(min(n_select, n))
    if k <= 0:
        return GreedyLog([], [], [])
    beta = 1.0 - alpha
    w = np.full(len(A), 1.0 / len(A)) if weights is None else np.asarray(weights, dtype=np.float64)
    if beta > 0 and (w < 0).any():
        raise ValueError("Trọng số coverage phải không âm.")
    ucoef = alpha / k
    # Ma trận kernel |A| × |E| tính trước MỘT lần (PCam âm: 18k × 54k float32 ≈ 3,9 GB); nhỏ thì float64 cho chính xác.
    Kmat = None
    if beta > 0:
        dtype = np.float64 if len(A) * n <= 50_000_000 else np.float32
        if len(A) * n * np.dtype(dtype).itemsize > max_kernel_bytes:
            raise MemoryError(f"Ma trận kernel {len(A)}×{n} vượt {max_kernel_bytes / 1e9:.1f} GB.")
        Kmat = np.empty((len(A), n), dtype=dtype)
        for s in range(0, n, block):
            Kmat[:, s:s + block] = np.exp(-_sqdist_block(A, E[s:s + block]) / (2.0 * h * h))
    cum = np.zeros(len(A), dtype=np.float64)        # Σ_{u∈U} κ(x, u) của mỗi ảnh thật

    def gains(cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gu = ucoef * u[cols]
        if Kmat is None:
            return gu, np.zeros(len(cols))
        base = np.log1p(cum)
        gc = beta * (w @ (np.log1p(cum[:, None] + Kmat[:, cols]) - base[:, None]))
        return gu, gc

    heap: list[tuple[float, int]] = []
    for s in range(0, n, block):
        cols = np.arange(s, min(s + block, n))
        gu, gc = gains(cols)
        heap.extend((-(a + c), int(i)) for a, c, i in zip(gu, gc, cols))
    heapq.heapify(heap)
    out = GreedyLog([], [], [])
    fresh_round = np.full(n, -1)
    while len(out.order) < k and heap:
        neg, i = heapq.heappop(heap)
        t = len(out.order)
        if fresh_round[i] == t:                                     # đã tính lại ở lượt này -> chính xác
            gu, gc = gains(np.array([i]))
            out.order.append(i)
            out.gain_utility.append(float(gu[0]))
            out.gain_coverage.append(float(gc[0]))
            if Kmat is not None:
                cum += Kmat[:, i]
            continue
        gu, gc = gains(np.array([i]))
        fresh_round[i] = t
        heapq.heappush(heap, (-(float(gu[0]) + float(gc[0])), i))
    return out


def topk_ids(score: np.ndarray, k: int) -> list[int]:
    """Top-k theo điểm giảm dần, hoà -> chỉ số nhỏ hơn."""
    s = np.asarray(score, dtype=np.float64)
    order = np.lexsort((np.arange(len(s)), -s))
    return order[:k].tolist()


# ------------------------------------------------------------------------------------------------ chọn cho một lớp
@dataclass(frozen=True)
class RCParams:
    eta: float = 1.0                 # h = η · h⁽⁰⁾
    mu_percentile: float = 50.0      # μ_c = phân vị (25 / 50 / 75)
    sigma: float = 1.0
    alpha: float = 0.5               # trọng số utility; β = 1 − α
    delta: float = 0.0               # ngưỡng gate


@dataclass
class ClassSelection:
    eligible: np.ndarray                         # chỉ số pool qua NC + gate (ℰ_c)
    budget: int                                  # K khả thi = min(K_c, |ℰ_c|)
    shortfall: int                               # K_c − K khả thi
    selections: dict[str, list[int]]             # chỉ số TRONG POOL
    logs: dict[str, GreedyLog]
    h: float
    mu: float
    tmu_values: np.ndarray                       # T trên ℰ_c


def select_class(*, pool_emb: np.ndarray, pool_logit: np.ndarray, real_emb: np.ndarray, real_logit: np.ndarray,
                 cls: int, nc_keep: np.ndarray, k_target: int, params: RCParams, seed: int,
                 methods: tuple[str, ...] = RC_SELECTION_METHODS, bandwidth_seed: int = 0) -> ClassSelection:
    """Chạy toàn bộ bước 2–5 cho lớp `cls`; mọi biến thể dùng chung ℰ_c và ngân sách K."""
    pool_e = l2n(pool_emb)
    real_e = l2n(real_emb)
    m_pool = class_margin(pool_logit, cls)
    m_real = class_margin(real_logit, cls)
    ok = valid_rows(pool_e) & np.isfinite(m_pool) & np.asarray(nc_keep, dtype=bool) & gate(m_pool, params.delta)
    eligible = np.flatnonzero(ok)
    real_ok = valid_rows(real_e) & np.isfinite(m_real)
    anchors = real_e[real_ok]
    budget = int(min(k_target, len(eligible)))
    if budget < k_target:
        log.warning("Lớp %d: chỉ %d ứng viên đạt điều kiện < K_c = %d.", cls, len(eligible), k_target)

    stats = margin_stats(m_real[real_ok])
    mu = tmu_center(m_real[real_ok], stats, params.mu_percentile)
    T = tmu(m_pool[eligible], stats, mu, params.sigma)
    h = params.eta * median_bandwidth(anchors, seed=bandwidth_seed)
    E = pool_e[eligible]

    sel: dict[str, list[int]] = {}
    logs: dict[str, GreedyLog] = {}
    for m in methods:
        if m == RC_RANDOM:
            loc = np.sort(np.random.default_rng(seed).choice(len(eligible), size=budget, replace=False)).tolist()
        elif m == RC_TMU:
            loc = topk_ids(T, budget)
        elif m == RC_RDC:
            logs[m] = greedy_select(E, np.zeros(len(E)), anchors, budget, 0.0, h)
            loc = logs[m].order
        elif m == RC_TMU_POOLCOV:
            logs[m] = greedy_select(E, T, E, budget, params.alpha, h)
            loc = logs[m].order
        elif m == RC_MAXMARGIN_RDC:
            logs[m] = greedy_select(E, margin_rank(m_pool[eligible]), anchors, budget, params.alpha, h)
            loc = logs[m].order
        elif m == RC_DASS:
            logs[m] = greedy_select(E, T, anchors, budget, params.alpha, h)
            loc = logs[m].order
        else:
            raise ValueError(f"Phương pháp chọn không hợp lệ: {m}")
        sel[m] = [int(eligible[j]) for j in loc]
    return ClassSelection(eligible=eligible, budget=budget, shortfall=int(k_target - budget), selections=sel,
                          logs=logs, h=float(h), mu=float(mu), tmu_values=T)
