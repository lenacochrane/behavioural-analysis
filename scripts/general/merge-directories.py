import os
import shutil

def safe_merge_dirs(dir_a, dir_b, output_dir):
    for source_dir in [dir_a, dir_b]:
        for root, _, files in os.walk(source_dir):
            rel_path = os.path.relpath(root, source_dir)
            dst_root = os.path.join(output_dir, rel_path)

            os.makedirs(dst_root, exist_ok=True)

            for file in files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dst_root, file)

                # If file already exists in destination
                if os.path.exists(dst_file):
                    # If contents are identical, skip
                    with open(src_file, 'rb') as f1, open(dst_file, 'rb') as f2:
                        if f1.read() == f2.read():
                            continue

                    # Otherwise rename to avoid conflict
                    base, ext = os.path.splitext(file)
                    counter = 1
                    while True:
                        new_name = f"{base}_conflict{counter}{ext}"
                        new_dst_file = os.path.join(dst_root, new_name)
                        if not os.path.exists(new_dst_file):
                            dst_file = new_dst_file
                            break
                        counter += 1

                shutil.copy2(src_file, dst_file)

# Example usage:
dir_a = "/Users/cochral/The Francis Crick Dropbox/Lena Cochrane/cochral/old-mac/plots"
dir_b = "/Users/cochral/repos/behavioural-analysis/plots"
output = "/Users/cochral/The Francis Crick Dropbox/Lena Cochrane/cochral/old-mac/merged-plots"

safe_merge_dirs(dir_a, dir_b, output)

