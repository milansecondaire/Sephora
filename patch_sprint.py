import yaml
import re

with open('_bmad-output/implementation-artifacts/sprint-status.yaml', 'r') as f:
    content = f.read()

content = re.sub(r'4-5-algorithm-comparison-final-selection:\s*\w+', '4-5-algorithm-comparison-final-selection: done', content)

with open('_bmad-output/implementation-artifacts/sprint-status.yaml', 'w') as f:
    f.write(content)
