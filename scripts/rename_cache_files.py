# rename_cache_files.py
import os

# --- Configuration ---
# The script assumes your cache directory is named "data_cache" and is in the same
# folder as this script. If it's located elsewhere, change this path.
CACHE_DIR = "data_old"
OLD_PREFIX = "bn_aggtrades_"
NEW_PREFIX = "bn_agg_trades_"
# ---------------------

def rename_cached_files():
    """
    Scans the cache directory and renames aggregate trade files from the old
    naming convention to the new one.
    """
    if not os.path.isdir(CACHE_DIR):
        print(f"Error: Cache directory '{CACHE_DIR}' not found.")
        print("Please make sure you are running this script from your project's root directory.")
        return

    print(f"Scanning '{CACHE_DIR}' for files to rename...")
    files_renamed_count = 0

    # os.walk will go through all directories and subdirectories
    for root, _, files in os.walk(CACHE_DIR):
        for filename in files:
            # Check if the file matches the old naming pattern
            if filename.startswith(OLD_PREFIX):
                old_path = os.path.join(root, filename)
                
                # Create the new filename by replacing the old prefix
                new_filename = filename.replace(OLD_PREFIX, NEW_PREFIX, 1)
                new_path = os.path.join(root, new_filename)

                try:
                    # Rename the file
                    os.rename(old_path, new_path)
                    print(f"  RENAMED: {old_path}\n        TO: {new_path}")
                    files_renamed_count += 1
                except OSError as e:
                    print(f"Error renaming file {old_path}: {e}")

    if files_renamed_count == 0:
        print("\nNo files with the old prefix were found. Your cache may already be up to date.")
    else:
        print(f"\nFinished. Renamed {files_renamed_count} files successfully.")

if __name__ == "__main__":
    # It's always wise to back up your data before running a script that modifies it.
    # You can make a zip of your 'data_cache' folder as a quick backup.
    print("--- Cache Renaming Utility ---")
    print("This script will rename files in your cache to match the updated naming convention.")
    # Uncomment the line below if you want to require user confirmation
    # input("Press Enter to continue or Ctrl+C to cancel...")
    rename_cached_files()