import os
import sys
import shutil
import PyInstaller.__main__

def clean_directories():
    """Clean build and dist directories."""
    for dir_name in ['build', 'dist']:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"Cleaned {dir_name} directory")

def create_spec_file():
    """Create a spec file for the Air Canvas Calculator."""
    # Get system prefix for mediapipe path, using forward slashes for compatibility
    mediapipe_path = os.path.join(sys.prefix, 'Lib', 'site-packages', 'mediapipe').replace('\\', '/')

    spec_content = f'''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['air_canvas.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('digit_recognition_model.h5', '.'),
        ('{mediapipe_path}', 'mediapipe')
    ],
    hiddenimports=[
        'tensorflow',
        'tensorflow.lite.python.lite',
        'keras',
        'mediapipe',
        'mediapipe.python',
        'mediapipe.python.solutions',
        'mediapipe.python.solutions.hands',
        'mediapipe.python.solutions.drawing_utils',
        'cv2',
        'numpy'
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='AirCanvasCalculator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)'''

    with open('air_canvas.spec', 'w') as f:
        f.write(spec_content)
    print("Created spec file")

def build_executable():
    """Build the executable using PyInstaller."""
    try:
        PyInstaller.__main__.run([
            'air_canvas.spec',
            '--noconfirm',
            '--clean'
        ])
        return True
    except Exception as e:
        print(f"Build error: {str(e)}")
        return False

def verify_build():
    """Verify the build was successful."""
    exe_path = os.path.join('dist', 'AirCanvasCalculator.exe')
    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print(f"\nBuild successful! Executable size: {size_mb:.2f} MB")
        print(f"Location: {os.path.abspath(exe_path)}")
        return True
    return False

def check_dependencies():
    """Check if all required packages are installed."""
    required_packages = [
        'tensorflow',
        'opencv-python',
        'mediapipe',
        'numpy',
        'keras'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
                print(f"OpenCV version: {cv2.__version__}")
            else:
                module = __import__(package)
                print(f"Found {package} version: {module.__version__}")
        except ImportError as e:
            missing_packages.append(package)
            print(f"Error importing {package}: {str(e)}")
        except Exception as e:
            print(f"Unexpected error with {package}: {str(e)}")

    if missing_packages:
        print("\nMissing required packages:")
        for package in missing_packages:
            print(f"- {package}")
        print("\nPlease install missing packages using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    return True

def main():
    print("=== Air Canvas Calculator Builder ===")

    # Check dependencies first
    if not check_dependencies():
        return

    # Check for required files
    if not os.path.exists('air_canvas.py'):
        print("Error: air_canvas.py not found!")
        return

    if not os.path.exists('digit_recognition_model.h5'):
        print("Warning: digit_recognition_model.h5 not found!")
        input("Press Enter to continue without the model, or Ctrl+C to abort...")

    # Clean previous builds
    clean_directories()

    # Create spec file
    create_spec_file()

    # Build executable
    print("\nBuilding executable (this may take several minutes)...")
    if build_executable() and verify_build():
        print("\nBuild completed successfully!")
        print("\nInstructions:")
        print("1. Go to the 'dist' folder")
        print("2. Run 'AirCanvasCalculator.exe'")
        print("3. Make sure you have a webcam connected")
        print("\nNote: The first launch may take a few seconds.")
    else:
        print("\nBuild failed! Check the error messages above.")

if __name__ == "__main__":
    main()
