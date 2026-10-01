"""Clone StyleGAN2-ADA chính thức và vá cho tương thích Python 3.12 / PyTorch 2.x.

Mã gốc viết cho PyTorch 1.7–1.9; Colab không còn wheel PyTorch 1.x nên vá mã nguồn thay vì hạ phiên bản.
Mọi bản vá là idempotent: vá lại lần hai không đổi gì; không tìm thấy đoạn cần vá -> báo lỗi.
"""

from __future__ import annotations

import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from ..utils import add_to_sys_path, run_command

log = logging.getLogger(__name__)

SG2_URL = "https://github.com/NVlabs/stylegan2-ada-pytorch.git"


@dataclass(frozen=True)
class Patch:
    rel_path: str
    old: str
    new: str
    reason: str


PATCHES: tuple[Patch, ...] = (
    Patch("dnnlib/util.py",
          "from distutils.util import strtobool",
          "def strtobool(val):  # [PATCH] replaces distutils.util.strtobool (removed in Python 3.12)\n"
          "    val = str(val).lower()\n"
          "    if val in ('y', 'yes', 't', 'true', 'on', '1'):\n"
          "        return 1\n"
          "    if val in ('n', 'no', 'f', 'false', 'off', '0'):\n"
          "        return 0\n"
          "    raise ValueError(f'invalid truth value {val!r}')",
          "Python 3.12 đã xoá distutils"),
    # grid_sample: KHÔNG tắt custom op. R1 cần đạo hàm bậc 2 qua grid_sample của ADA, PyTorch chưa hỗ trợ.
    Patch("torch_utils/ops/grid_sample_gradfix.py",
          "    if any(torch.__version__.startswith(x) for x in ['1.7.', '1.8.', '1.9']):\n        return True\n",
          "    return True  # [PATCH] custom op made compatible with PyTorch 2.x (needed for R1 through ADA)\n",
          "giữ custom op grid_sample trên PyTorch 2.x"),
    Patch("torch_utils/ops/grid_sample_gradfix.py",
          "        op = torch._C._jit_get_operation('aten::grid_sampler_2d_backward')\n"
          "        grad_input, grad_grid = op(grad_output, input, grid, 0, 0, False)\n",
          "        # [PATCH] PyTorch>=1.11: grid_sampler_2d_backward requires output_mask\n"
          "        grad_input, grad_grid = torch.ops.aten.grid_sampler_2d_backward("
          "grad_output, input, grid, 0, 0, False, [True, True])\n",
          "chữ ký mới của grid_sampler_2d_backward (output_mask)"),
    Patch("torch_utils/ops/conv2d_gradfix.py",
          "def _should_use_custom_op(input):\n",
          "def _should_use_custom_op(input):\n    return False  # [PATCH] PyTorch 2.x: use torch.nn.functional.conv2d\n",
          "conv2d dùng op chuẩn của PyTorch"),
    Patch("training/training_loop.py",
          "        augment_pipe.p.copy_(torch.as_tensor(augment_p))\n",
          "        augment_pipe.p.copy_(torch.as_tensor(augment_p))\n"
          "        if ('resume_data' in locals()) and (resume_data.get('augment_pipe', None) is not None):  # [PATCH]\n"
          "            augment_pipe.p.copy_(resume_data['augment_pipe'].p.to(device))\n"
          "            print(f'[PATCH] Restored ADA p = {float(augment_pipe.p):.3f} from resume pickle')\n",
          "khôi phục xác suất ADA p khi resume (bản gốc reset p = 0)"),
    Patch("torch_utils/custom_ops.py",
          "            torch.utils.cpp_extension.load(name=module_name, build_directory=build_dir,",
          "            module = torch.utils.cpp_extension.load(name=module_name, build_directory=build_dir,",
          "PyTorch>=1.13: dùng module do load() trả về (1/3)"),
    Patch("torch_utils/custom_ops.py",
          "            torch.utils.cpp_extension.load(name=module_name, verbose=verbose_build, sources=sources, **build_kwargs)",
          "            module = torch.utils.cpp_extension.load(name=module_name, verbose=verbose_build, sources=sources, **build_kwargs)",
          "PyTorch>=1.13: dùng module do load() trả về (2/3)"),
    Patch("torch_utils/custom_ops.py",
          "        module = importlib.import_module(module_name)\n",
          "        # [PATCH] PyTorch>=1.13: use the module returned by load() (no longer registered in sys.modules)\n",
          "PyTorch>=1.13: dùng module do load() trả về (3/3)"),
    Patch("torch_utils/misc.py",
          "        super().__init__(dataset)\n",
          "        super().__init__()  # [PATCH] PyTorch>=2.2: Sampler.__init__ no longer takes data_source\n",
          "PyTorch>=2.2: Sampler.__init__ không nhận data_source"),
)


def apply_patch(repo_dir: str | Path, patch: Patch) -> str:
    """Trả về 'patched' | 'already'. Báo lỗi nếu không tìm thấy đoạn cần vá."""
    path = Path(repo_dir) / patch.rel_path
    text = path.read_text()
    if patch.new in text:
        return "already"
    if patch.old not in text:
        raise RuntimeError(f"[patch] Không tìm thấy đoạn cần vá trong {patch.rel_path} ({patch.reason})")
    path.write_text(text.replace(patch.old, patch.new, 1))
    return "patched"


def ensure_stylegan_repo(repo_dir: str | Path, reset: bool = False) -> Path:
    """Clone (nếu chưa có), tuỳ chọn đưa về nguyên bản, áp mọi bản vá và thêm repo vào sys.path."""
    repo_dir = Path(repo_dir)
    if not repo_dir.is_dir():
        run_command(["git", "clone", "-q", SG2_URL, str(repo_dir)])
    if reset:
        run_command(["git", "checkout", "-q", "--", "."], cwd=repo_dir)
    for patch in PATCHES:
        status = apply_patch(repo_dir, patch)
        log.info("[patch] %-40s %-8s %s", patch.rel_path, status, patch.reason)
    add_to_sys_path(repo_dir)
    return repo_dir


def verify_cuda_plugins(repo_dir: str | Path) -> None:
    """Biên dịch và kiểm tra plugin CUDA (True = plugin CUDA; False = bản tham chiếu, đúng nhưng chậm)."""
    code = (
        "import torch\n"
        "from torch_utils.ops import bias_act, upfirdn2d\n"
        "print('torch', torch.__version__, '| CUDA', torch.version.cuda, '| GPU', torch.cuda.get_device_name(0))\n"
        "print('bias_act CUDA plugin:', bias_act._init())\n"
        "print('upfirdn2d CUDA plugin:', upfirdn2d._init())\n"
    )
    run_command([sys.executable, "-c", code], cwd=repo_dir)


def clear_torch_extension_cache() -> None:
    cache = Path.home() / ".cache" / "torch_extensions"
    shutil.rmtree(cache, ignore_errors=True)
    log.info("Đã xoá cache plugin: %s", cache)
