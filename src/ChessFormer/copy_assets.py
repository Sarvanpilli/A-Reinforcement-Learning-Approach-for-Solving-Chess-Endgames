import shutil
import os

def copy_assets():
    # Source: src/GUI
    # Dest: src/ChessFormer
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(base_dir, '../GUI')
    dst_dir = os.path.join(base_dir, '.')
    
    files = ['chessboard.gif', 'wking.gif', 'wrook.gif', 'bking.gif']
    
    print(f"Copying from {src_dir} to {dst_dir}")
    
    for f in files:
        src = os.path.join(src_dir, f)
        dst = os.path.join(dst_dir, f)
        
        if os.path.exists(src):
            try:
                shutil.copy(src, dst)
                print(f"SUCCESS: Copied {f}")
            except Exception as e:
                print(f"ERROR: Failed to copy {f}: {e}")
        else:
            print(f"ERROR: Source file not found: {src}")

if __name__ == "__main__":
    copy_assets()
