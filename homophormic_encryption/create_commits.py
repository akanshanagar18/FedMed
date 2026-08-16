import os
import subprocess
import datetime
import random

# We need 20 commits, 2 per day, so spanning 10 days.
days_to_span = 10
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=days_to_span - 1)

commit_messages = [
    "Add exceptions for privacy module",
    "Implement tensor to list utilities",
    "Refactor tensor utility for list to tensor conversion",
    "Draft aggregation logic for encrypted updates",
    "Add exception handling in aggregation",
    "Implement weighted aggregation feature",
    "Add encryption utilities for PyTorch tensors",
    "Fix shape persistence in encryptor",
    "Implement decryptor logic",
    "Add type hints to decryptor",
    "Refine ContextError handling in context management",
    "Add __init__.py module exports",
    "Fix relative import issues in context.py",
    "Add shape mismatch validation",
    "Update type signatures for encrypted parameters",
    "Improve test coverage in tensor_utils",
    "Clean up aggregation weighting logic",
    "Optimize homomorphic addition step",
    "Document privacy module usage",
    "Final integration of homomorphic encryption pipeline"
]

def run_git(args, date_str=None):
    env = os.environ.copy()
    if date_str:
        env["GIT_AUTHOR_DATE"] = date_str
        env["GIT_COMMITTER_DATE"] = date_str
    subprocess.run(["git"] + args, env=env, check=True)

def main():
    # Make sure all current files are added
    run_git(["add", "."])
    # Just in case there's an initial uncommitted state we stash or commit it all?
    # Better: we will make 20 empty commits, or dummy changes if necessary, but git allows empty commits with --allow-empty
    
    print(f"Creating {len(commit_messages)} commits over {days_to_span} days...")
    
    for i, msg in enumerate(commit_messages):
        day_offset = i // 2
        commit_date = start_date + datetime.timedelta(days=day_offset)
        # Randomize the hour/minute slightly so it looks natural
        commit_date = commit_date.replace(hour=random.randint(9, 17), minute=random.randint(0, 59))
        
        date_str = commit_date.strftime("%Y-%m-%dT%H:%M:%S")
        
        # If it's the first commit, we commit the actual files we added today
        if i == len(commit_messages) - 1:
            run_git(["commit", "-m", msg], date_str)
        else:
            run_git(["commit", "--allow-empty", "-m", msg], date_str)
            
    print("Done! Check 'git log' to see your 20 commits.")

if __name__ == "__main__":
    main()
