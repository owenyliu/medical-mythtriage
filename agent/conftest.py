# Makes `import mythtriage` work when pytest runs from this folder,
# without installing anything.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
