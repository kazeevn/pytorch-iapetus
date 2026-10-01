try:
    import torch
    if hasattr(torch, "_C") and hasattr(torch._C, "_distributed_c10d"):
        c10d = torch._C._distributed_c10d
        if hasattr(c10d, "TCPStore"):
            _orig_init = c10d.TCPStore.__init__
            def _safe_init(self, *args, **kwargs):
                kwargs["use_libuv"] = False
                _orig_init(self, *args, **kwargs)
            c10d.TCPStore.__init__ = _safe_init
except Exception:
    pass
