"""Prepare or run the public v1.0.0 banking implementation with BM25."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shlex
import shutil
import subprocess
import tempfile
from score import summarize

HERE = Path(__file__).resolve().parent
PIN = "17e07b1da2bbc0cadfddeea36412686e0604127b"
RELEASE = HERE / "tau2-v1.0.0"


def run(*args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def prepare():
    if not RELEASE.exists():
        run("git", "clone", "--quiet", "--no-checkout", str(HERE.parent / "tau2-bench"), str(RELEASE))
        run("git", "-C", str(RELEASE), "checkout", "--quiet", "--detach", PIN)
    actual = subprocess.check_output(["git", "-C", str(RELEASE), "rev-parse", "HEAD"], text=True).strip()
    if actual != PIN or subprocess.call(["git", "-C", str(RELEASE), "diff", "--quiet", "HEAD"]):
        raise SystemExit("Unexpected release revision or tracked changes in " + str(RELEASE))


def setup(env):
    uv = shutil.which("uv") or str(HERE / ".tools/uv")
    if not Path(uv).is_file():
        raise SystemExit("Install uv or place the uv executable in final/baseline/.tools/.")
    python = RELEASE / ".venv/bin/python"
    run(uv, "sync", "--frozen", "--inexact", "--extra", "knowledge", "--python",
        str(python) if python.exists() else "3.12", cwd=RELEASE, env=env)
    # The v1.0.0 text CLI imports audio playback and synthesis during startup.
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        portaudio = HERE / ".tools/portaudio"
        if not (portaudio / "lib/libportaudio.dylib").exists():
            url = "https://conda.anaconda.org/conda-forge/osx-arm64/portaudio-19.6.0-hb7217d7_7.tar.bz2"
            with tempfile.TemporaryDirectory() as directory:
                archive = Path(directory) / "portaudio.tar.bz2"
                run("curl", "-fsSL", "--retry", "2", "-o", str(archive), url)
                if hashlib.sha256(archive.read_bytes()).hexdigest() != "391742855da750f7c70d5e5d97dd66c4ec1fe87972e3464a41fb3b3be5f72dde":
                    raise SystemExit("PortAudio archive checksum mismatch")
                portaudio.mkdir(parents=True, exist_ok=True)
                run("tar", "-xjf", str(archive), "-C", str(portaudio), "include", "lib", "info/licenses")
        env["CFLAGS"] = f"-I{portaudio}/include " + env.get("CFLAGS", "")
        env["LDFLAGS"] = f"-L{portaudio}/lib -Wl,-rpath,{portaudio}/lib " + env.get("LDFLAGS", "")
    run(uv, "pip", "install", "--python", str(python), "scipy==1.16.3", "jiwer==4.0.0",
        "pydub==0.25.1", "pyaudio==0.2.14", "websockets==15.0.1", "elevenlabs==2.23.0",
        "socksio==1.0.0", cwd=RELEASE, env=env)
    run(str(RELEASE / ".venv/bin/tau2"), "check-data", cwd=RELEASE, env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["setup", "smoke", "dev", "full"])
    parser.add_argument("--profile", choices=["flash", "paper", "openrouter"], default="flash")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    prepare()
    env = {**os.environ, "TAU2_DATA_DIR": str(RELEASE / "data")}
    env.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    if args.stage == "setup":
        setup(env)
        return
    catalog = json.loads((RELEASE / "data/tau2/domains/banking_knowledge/tasks.json").read_text())
    ids = sorted(t["id"] for t in catalog)
    if len(set(ids)) != 97:
        raise SystemExit("Unexpected task catalog")
    known = ["task_" + n for n in ("027", "043", "062", "071", "077", "098")]
    dev = known + random.Random(300).sample(sorted(set(ids) - set(known)), 4)
    selected = {"smoke": ["task_098"], "dev": dev, "full": ids}[args.stage]
    trials = 4 if args.stage == "full" else 1
    user_model = "gpt-5.2"
    if args.profile == "openrouter":
        model = user_model = "openrouter/openai/gpt-5.2"
        llm_args = {"reasoning_effort": "high"}
    elif args.profile == "flash":
        model = os.getenv("TAU_FLASH_MODEL")
        if not model and not args.dry_run:
            raise SystemExit("Set TAU_FLASH_MODEL to the provider's exact LiteLLM model identifier.")
        model = model or "PROVIDER/FLASH_MODEL"
        llm_args = {}
        if os.getenv("TAU_FLASH_API_BASE"):
            llm_args["api_base"] = os.environ["TAU_FLASH_API_BASE"]
    else:
        model, llm_args = "gpt-5.2", {"reasoning_effort": "high"}
    safe_model = "".join(c if c.isalnum() else "_" for c in model)
    output = HERE / "runs" / f"{args.profile}-{safe_model}-{args.stage}"
    cmd = [str(RELEASE / ".venv/bin/tau2"), "run", "--domain", "banking_knowledge",
           "--agent", "llm_agent", "--user", "user_simulator", "--retrieval-config", "bm25",
           "--agent-llm", model, "--agent-llm-args", json.dumps(llm_args),
           "--user-llm", user_model, "--user-llm-args", '{"reasoning_effort":"low"}',
           "--seed", "123", "--max-steps", "200", "--max-errors", "10",
           "--num-trials", str(trials), "--max-concurrency", "1", "--save-to", str(output),
           "--task-ids", *selected]
    if args.dry_run:
        print(shlex.join(cmd))
        return
    if not Path(cmd[0]).is_file():
        raise SystemExit("Install the runner with run.py setup first.")
    if args.profile == "openrouter":
        if not env.get("OPENROUTER_API_KEY"):
            key_file = HERE.parent / "key.txt"
            if not key_file.is_file():
                raise SystemExit("Set OPENROUTER_API_KEY or provide final/key.txt.")
            env["OPENROUTER_API_KEY"] = key_file.read_text().strip()
        if not env["OPENROUTER_API_KEY"].startswith("sk-or-") or any(
            c.isspace() for c in env["OPENROUTER_API_KEY"]
        ):
            raise SystemExit("OpenRouter credential must be a single key.")
    elif not env.get("OPENAI_API_KEY") and not (RELEASE / ".env").is_file():
        raise SystemExit("Configure credentials for the GPT-5.2 user simulator (OPENAI_API_KEY).")
    run(*cmd, cwd=RELEASE, env=env)
    data = json.loads((output / "results.json").read_bytes())
    print(json.dumps(summarize(data, selected, trials), indent=2))


if __name__ == "__main__":
    main()
