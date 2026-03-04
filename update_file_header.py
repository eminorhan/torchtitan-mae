import os

# --- Configuration ---
# The folder containing your code ('.' means current directory)
DIRECTORY = '.' 
# The file extension you are targeting (e.g., '.py', '.cpp', '.js')
FILE_EXTENSION = '.py' 
# Your copyright line
YOUR_COPYRIGHT = '# Copyright (c) Emin Orhan.\n'
# The target line to search for
# Copyright (c) Emin Orhan.
META_COPYRIGHT = '# Copyright (c) Meta Platforms, Inc. and affiliates.'

for root, _, files in os.walk(DIRECTORY):
    for file in files:
        if file.endswith(FILE_EXTENSION):
            filepath = os.path.join(root, file)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.readlines()
            
            content_str = "".join(content)
            
            # Check if Meta's line is there, but yours isn't yet
            if META_COPYRIGHT in content_str and YOUR_COPYRIGHT not in content_str:
                # Find Meta's line and insert yours right above it
                for i, line in enumerate(content):
                    if META_COPYRIGHT in line:
                        content.insert(i, YOUR_COPYRIGHT)
                        break
                
                # Write the changes back to the file
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.writelines(content)
                print(f"Updated: {filepath}")

print("Done!")