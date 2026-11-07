import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def defined(module: Path) -> set[str]:
    out = set()
    for n in ast.parse(module.read_text()).body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.Assign):
            out |= {t.id for t in n.targets if isinstance(t, ast.Name)}
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            out.add(n.target.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            out |= {a.asname or a.name.split(".")[0] for a in n.names}
    return out


def resolves(ref: str) -> bool:
    parts = ref.split(".")
    for k in range(len(parts), 0, -1):
        base = ROOT.joinpath("src", *parts[:k])
        module = base.with_suffix(".py") if base.with_suffix(".py").exists() else base / "__init__.py"
        if module.exists():
            rest = parts[k:]
            return not rest or (len(rest) <= 2 and rest[0] in defined(module))
    return False


def test_every_module_named_in_the_architecture_exists():
    text = (ROOT / "docs" / "architecture.md").read_text()
    refs = sorted({r.rstrip(".") for r in re.findall(r"\bp5(?:\.[A-Za-z_]\w*)+\.?", text)})
    assert len(refs) > 20
    assert [r for r in refs if not resolves(r)] == []
