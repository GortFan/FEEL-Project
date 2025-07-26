import sys
import os

# Get the parent directory (D:/TMUFEEL)
project_root = os.path.dirname(os.path.dirname(__file__))

# Add build/Release to sys.path
sys.path.append(os.path.join(project_root, 'build', 'Release'))

import mybindings

print(mybindings.add(3, 4))
print(mybindings.subtract(3, 4))

print(mybindings.index(1,2,3, 5, 5))