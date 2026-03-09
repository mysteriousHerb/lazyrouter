import re

with open("tests/test_router_caching.py", "r") as f:
    content = f.read()

content = re.sub(
    r"<<<<<<< HEAD\n.*?\n=======\n(.*?)\n>>>>>>> d0995af \(temp\)\n",
    r"\1\n",
    content,
    flags=re.DOTALL,
)

with open("tests/test_router_caching.py", "w") as f:
    f.write(content)
