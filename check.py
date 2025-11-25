"""
Script untuk mengecek apakah semua library sudah terinstall
"""

import sys

def check_module(module_name, import_name=None):
    """Check if module is installed"""
    if import_name is None:
        import_name = module_name
    
    try:
        mod = __import__(import_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✅ {module_name:20s} → {version}")
        return True
    except ImportError:
        print(f"❌ {module_name:20s} → NOT INSTALLED")
        return False

print("="*60)
print("🔍 CHECKING INSTALLED MODULES")
print("="*60)
print()

modules = [
    ('opencv-python', 'cv2'),
    ('tensorflow', 'tensorflow'),
    ('keras', 'keras'),
    ('numpy', 'numpy'),
    ('matplotlib', 'matplotlib'),
    ('pillow', 'PIL'),
    ('pandas', 'pandas'),
    ('scikit-learn', 'sklearn'),
]

results = []
for display_name, import_name in modules:
    results.append(check_module(display_name, import_name))

print()
print("="*60)
installed = sum(results)
total = len(results)

if installed == total:
    print(f"🎉 SUKSES! Semua {total} modul terinstall dengan baik!")
else:
    print(f"⚠️  {installed}/{total} modul terinstall")
    print(f"   {total - installed} modul perlu diinstall")
    print()
    print("Jalankan: pip install -r requirements.txt")

print("="*60)

# Check Python version
print(f"\n🐍 Python Version: {sys.version}")
print()

# Additional info
try:
    import tensorflow as tf
    print(f"📊 TensorFlow Details:")
    print(f"   - Version: {tf.__version__}")
    print(f"   - GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
except:
    pass