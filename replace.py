import os
import re

SEARCH = re.compile(r'''= np.random.uniform\(
\s*self.obj_and_goal_space.low,
\s*self.obj_and_goal_space.high,
\s*size=\(self.obj_and_goal_space.low.size\),
\s*\)''')
REPLACE = r'''= self._get_state_rand_vec()'''

for root, _, files in os.walk('metaworld'):
    for name in files:
        if not name.endswith('.py'):
            continue
        filename = os.path.join(root, name)
        with open(filename) as f:
            contents = f.read()
        with open(filename, 'w') as f:
            f.write(SEARCH.sub(REPLACE, contents))
