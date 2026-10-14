import sys, os
if len(sys.argv) < 3:
    print('Usage: write_file.py <target_path> <hex_content>')
    sys.exit(1)
path = sys.argv[1]
hex_str = sys.argv[2]
os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
with open(path, 'wb') as f:
    f.write(bytes.fromhex(hex_str))
print(f'Wrote {path} ({os.path.getsize(path)} bytes)')
