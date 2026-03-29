import platform, os
from pathlib import Path

_machine = platform.machine().lower()
_system  = platform.system().lower()

IS_MAC_M1  = (_system == 'darwin' and _machine == 'arm64')
IS_PI5     = (_system == 'linux'  and _machine in ('aarch64', 'arm64'))
IS_X86     = _machine in ('x86_64', 'amd64')
MODEL_DIR = Path(os.environ.get('ANTIMONY_MODELS',
                 '/opt/NewAntimony/models' if IS_PI5 else
                 str(Path.home() / 'NewAntimony' / 'models')))
 
LIGHT_MODEL   = MODEL_DIR / 'qwen2.5-1.5b-instruct-q4_k_m.gguf'
MEDIUM_MODEL  = MODEL_DIR / 'phi-3-mini-4k-instruct-q4_k_m.gguf'
EMBED_MODEL   = 'sentence-transformers/all-MiniLM-L6-v2'
LIGHT_CTX   = 2048
MEDIUM_CTX  = 4096
N_GPU_LAYERS = -1 if (IS_MAC_M1) else 0
CHROMA_PATH = Path(os.environ.get('ANTIMONY_DATA',
                   '/opt/NewAntimony/data' if IS_PI5 else
                   str(Path.home() / 'NewAntimony' / 'data'))) / 'chroma'
EDUCATION_DIR = Path(__file__).parent / 'education'
ALLOWED_EDU_EXTENSIONS = {'.pdf', '.rtf'}
CONFIDENCE_THRESHOLD    = 0.55
INJECTION_PATTERNS_FILE = Path(__file__).parent / 'core' / 'injections.txt'
DDG_MAX_RESULTS   = 8
DDG_TIMEOUT_SEC   = 10
RESEARCH_MAX_ITER = 3
