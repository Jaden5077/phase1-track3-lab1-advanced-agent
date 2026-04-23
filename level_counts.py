import json
from collections import Counter
from pathlib import Path

path = Path("data/hotpot_100_sample.json")
data = json.loads(path.read_text(encoding="utf-8"))

# Nếu root là list các object
by_level = Counter(item["level"] for item in data)
print(dict(by_level))  # {'easy': ..., 'medium': ..., 'hard': ...}
print(sum(by_level.values()))  # tổng số mục
