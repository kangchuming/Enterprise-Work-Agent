from pathlib import Path
import re

DENY_PATTERNS = [
    r"\brm\s+-[rf]{1,2}\b",          # rm -r, rm -rf, rm -fr
    r"\bdel\s+/[fq]\b",              # del /f, del /q
    r"\brmdir\s+/s\b",               # rmdir /s
    r"(?:^|[;&|]\s*)format\b",       # format (as standalone command only)
    r"\b(mkfs|diskpart)\b",          # disk operations
    r"\bdd\s+if=",                   # dd
    r">\s*/dev/sd",                  # write to disk
    r"\b(shutdown|reboot|poweroff)\b",  # system power
    r":\(\)\s*\{.*\};\s*:",          # fork bomb
    # Block writes to nanobot internal state files (#2989).
    # history.jsonl / .dream_cursor are managed by append_history();
    # direct writes corrupt the cursor format and crash /dream.
    r">>?\s*\S*(?:history\.jsonl|\.dream_cursor)",            # > / >> redirect
    r"\btee\b[^|;&<>]*(?:history\.jsonl|\.dream_cursor)",     # tee / tee -a
    r"\b(?:cp|mv)\b(?:\s+[^\s|;&<>]+)+\s+\S*(?:history\.jsonl|\.dream_cursor)",  # cp/mv target
    r"\bdd\b[^|;&<>]*\bof=\S*(?:history\.jsonl|\.dream_cursor)",  # dd of=
    r"\bsed\s+-i[^|;&<>]*(?:history\.jsonl|\.dream_cursor)", 
]

class Guard:
    """封住配置Guard防护栏"""

    def _resolve_path(self, path: str, allowed_dir: Path) -> Path:
        p = Path(path).expanduser().resolve()

        # 允许目录不存在时也创建
        allowed_dir.mkdir(parents=True, exist_ok=True)
        allowed_dir = allowed_dir.resolve()
        
        
        try:
            p.relative_to(allowed_dir.resolve())
            return p
        except ValueError:
            raise PermissionError(f"路径 {path} 超出允许范围 {allowed_dir}")
    

    def guard_command(self, cmd: str) -> str | None:
        for pattern in DENY_PATTERNS:
            if re.search(pattern, cmd, re.IGNORECASE):
                return f"⚠️ 危险命令被拦截: {pattern}"
        return None
    
