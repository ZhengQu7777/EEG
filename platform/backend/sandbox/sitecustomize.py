"""脑电数据安全平台沙箱：记录联网、启动子进程和写文件事件。

只记录不阻断，隔离由 bwrap 负责（断网、只读文件系统、只有工作目录和 /tmp 可写）。
通过 PYTHONPATH 注入，沙箱内的每个 Python 进程启动时都会加载。
"""
import json
import os
import sys
import time

_LOG = os.environ.get("EEGSEC_EVENTS")
_WRITABLE = [p for p in os.environ.get("EEGSEC_WRITABLE", "").split(os.pathsep) if p]
_busy = False


def _emit(kind, detail):
    global _busy
    if _busy or not _LOG:
        return
    _busy = True
    try:
        with open(_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": round(time.time(), 3), "pid": os.getpid(), "kind": kind, "detail": str(detail)[:300]}, ensure_ascii=False) + "\n")
    except Exception:
        pass
    finally:
        _busy = False


_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "", None}


def _external(address):
    # Unix 套接字（多进程通信）和本机回环地址不算联网
    if isinstance(address, (str, bytes)):
        return False
    host = address[0] if isinstance(address, tuple) and address else None
    return host not in _LOCAL_HOSTS and not str(host).startswith("127.")


def _hook(event, args):
    if _busy:
        return
    try:
        if event == "socket.connect" and _external(args[1]):
            _emit("network", f"connect {args[1]}")
        elif event == "socket.getaddrinfo" and args[0] not in _LOCAL_HOSTS and not str(args[0]).startswith("127."):
            _emit("network", f"resolve {args[0]}:{args[1]}")
        elif event == "socket.sendto" and _external(args[1]):
            _emit("network", f"sendto {args[1]}")
        elif event == "subprocess.Popen":
            _emit("process", args[1] if len(args) > 1 else args[0])
        elif event in ("os.system", "os.exec", "os.posix_spawn", "os.spawn"):
            _emit("process", args[0] if args else event)
        elif event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            mode = args[1] if len(args) > 1 else None
            flags = args[2] if len(args) > 2 else 0
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or \
                      (mode is None and isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
            if writing:
                p = os.path.realpath(os.fsdecode(args[0]))
                if not any(p == w or p.startswith(w + os.sep) for w in _WRITABLE):
                    _emit("write-outside", p)
    except Exception:
        pass


if _LOG:
    sys.addaudithook(_hook)
