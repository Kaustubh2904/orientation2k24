# build_game.py

import os
import sys
import shutil
import PyInstaller.__main__

def clean_directories():
    """Clean build and dist directories if they exist"""
    directories = ['build', 'dist']
    for directory in directories:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"Cleaned {directory} directory")

def build_executable():
    """Build the executable using PyInstaller"""
    print("Starting build process...")

    # Ensure the script is in the same directory as the game
    if not os.path.exists('main.py'):
        print("Error: main.py not found in current directory!")
        return False

    try:
        PyInstaller.__main__.run([
            'main.py',  # your main script
            '--onefile',  # create a single executable
            '--windowed',  # prevent console window from appearing
            '--name=AirHockey',  # name of your executable
            '--add-data={}:mediapipe'.format(
                os.path.join(sys.prefix, 'Lib/site-packages/mediapipe')),
            '--hidden-import=mediapipe',
            '--hidden-import=mediapipe.python',
            '--hidden-import=mediapipe.python.solutions',
            '--hidden-import=mediapipe.python.solutions.hands',
            '--hidden-import=mediapipe.python.solutions.drawing_utils',
            '--hidden-import=mediapipe.python.solutions.drawing_styles',
            '--hidden-import=cv2',
            '--hidden-import=numpy',
            '--collect-data=mediapipe',
            '--collect-submodules=mediapipe',
            '--noconfirm',  # replace existing build/dist directories
            '--clean',  # clean PyInstaller cache
        ])
        return True
    except Exception as e:
        print(f"Error during build: {str(e)}")
        return False

def verify_build():
    """Verify that the executable was created successfully"""
    exe_path = os.path.join('dist', 'AirHockey.exe')
    if os.path.exists(exe_path):
        print(f"\nBuild successful! Executable created at: {exe_path}")
        print("File size:", round(os.path.getsize(exe_path) / (1024 * 1024), 2), "MB")
        return True
    else:
        print("\nBuild failed: Executable not found!")
        return False

def main():
    """Main build process"""
    print("=== Air Hockey Game Builder ===")

    # Check if required files exist
    if not os.path.exists('main.py'):
        print("Error: main.py not found!")
        print("Please make sure your game file is named 'main.py' and is in the same directory.")
        return

    # Clean previous builds
    clean_directories()

    # Build the executable
    if build_executable() and verify_build():
        print("\nBuild completed successfully!")
        print("\nInstructions:")
        print("1. Go to the 'dist' folder")
        print("2. Find 'AirHockey.exe'")
        print("3. Double-click to run the game")
        print("\nNote: Make sure you have a webcam connected and enabled!")
    else:
        print("\nBuild process failed!")
        print("Please check the error messages above.")

if __name__ == "__main__":
    main()