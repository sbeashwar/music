"""
Quick Start Script for Carnatic ML Projects

This script helps you:
1. Test the Raga Generation system (works now!)
2. Set up the Raga Detection system (needs data)
3. Run the data importer for enhanced metadata

Run with: python quick_start.py
"""

import os
import sys
import subprocess
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.absolute()
RAGA_GENERATION = PROJECT_ROOT / "raga_generation"
RAGA_DETECTION = PROJECT_ROOT / "raga_detection"
SHARED = PROJECT_ROOT / "shared"


def check_dependencies():
    """Check if required packages are installed."""
    print("\n🔍 Checking dependencies...")
    
    required = ['librosa', 'tensorflow', 'pretty_midi', 'sklearn', 'numpy']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg.replace('-', '_'))
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ❌ {pkg}")
            missing.append(pkg)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("  All dependencies OK!")
    return True


def test_generation():
    """Test the raga generation system."""
    print("\n" + "=" * 60)
    print("🎵 TESTING RAGA GENERATION")
    print("=" * 60)
    
    os.chdir(RAGA_GENERATION)
    
    # Check for existing model
    model_path = RAGA_GENERATION / "models" / "small_raga_gen.h5"
    tokenizer_path = RAGA_GENERATION / "models" / "tokenizer.json"
    
    if model_path.exists() and tokenizer_path.exists():
        print(f"✅ Found trained model: {model_path}")
        print(f"✅ Found tokenizer: {tokenizer_path}")
    else:
        print("📝 Model not found. Will train from scratch...")
    
    # Run demo
    print("\n▶️  Running raga generation demo...")
    print("-" * 40)
    
    try:
        # Import and run demo
        sys.path.insert(0, str(RAGA_GENERATION))
        from demo_run import main as demo_main
        demo_main()
        print("-" * 40)
        print("✅ Generation test complete! Check for generated MIDI files.")
        
        # List generated MIDI files
        midi_files = list(RAGA_GENERATION.glob("*.mid"))
        if midi_files:
            print("\n📁 Generated MIDI files:")
            for f in midi_files[-5:]:  # Show last 5
                print(f"   {f.name}")
                
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        import traceback
        traceback.print_exc()


def setup_detection():
    """Set up the raga detection system."""
    print("\n" + "=" * 60)
    print("🎧 SETTING UP RAGA DETECTION")
    print("=" * 60)
    
    os.chdir(RAGA_DETECTION)
    
    # Check for model
    model_path = RAGA_DETECTION / "models" / "raga_classifier.pkl"
    
    if model_path.exists():
        print(f"✅ Found trained model: {model_path}")
    else:
        print("⚠️  No trained model found.")
        print("\n   To train a model, you need audio data organized as:")
        print("   raga_detection/data/")
        print("   ├── mohanam/")
        print("   │   ├── clip1.wav")
        print("   │   ├── clip2.wav")
        print("   ├── kalyani/")
        print("   │   ├── clip1.wav")
        print("   │   └── ...")
    
    # Check for training data
    data_dir = RAGA_DETECTION / "data"
    if data_dir.exists():
        ragas = [d for d in data_dir.iterdir() if d.is_dir()]
        if ragas:
            print(f"\n📁 Found {len(ragas)} raga folders with training data")
            for raga in ragas[:10]:
                audio_files = list(raga.glob("*.wav")) + list(raga.glob("*.mp3"))
                print(f"   {raga.name}: {len(audio_files)} files")
        else:
            print(f"\n📁 Data directory exists but is empty: {data_dir}")
    else:
        print(f"\n📁 Creating data directory: {data_dir}")
        data_dir.mkdir(exist_ok=True)
    
    print("\n" + "-" * 40)
    print("Options to get training data:")
    print("1. Download Carnatic audio datasets (Saraga, Comp Music)")
    print("2. Generate synthetic data using raga_generation (MIDI→audio)")
    print("3. Record your own samples (30 sec clips per raga)")
    
    return model_path.exists()


def import_external_data():
    """Run the raga-of-the-week importer."""
    print("\n" + "=" * 60)
    print("📥 IMPORTING EXTERNAL RAGA DATA")
    print("=" * 60)
    
    importer_path = SHARED / "import_raga_of_the_week.py"
    
    if importer_path.exists():
        print(f"Found importer: {importer_path}")
        
        response = input("\nImport data from joereynolds/raga-of-the-week? [y/N]: ")
        if response.lower() == 'y':
            os.chdir(SHARED)
            subprocess.run([sys.executable, str(importer_path)])
    else:
        print("⚠️  Importer not found")


def generate_training_data():
    """Generate synthetic audio from MIDI for detection training."""
    print("\n" + "=" * 60)
    print("🎹 GENERATING SYNTHETIC TRAINING DATA")
    print("=" * 60)
    
    print("""
This would convert MIDI files to audio for detection training.
    
Steps:
1. Generate MIDI for multiple ragas using raga_generation
2. Convert MIDI to audio using FluidSynth or similar
3. Add variations (tempo, pitch, timing)
4. Organize into training folders

Would you like me to create a script for this? [y/N]: """)
    
    response = input().strip().lower()
    if response == 'y':
        print("\n📝 Creating synthetic data generator script...")
        # TODO: Create the script
        print("   (Script would be created here)")


def show_status():
    """Show current status of both projects."""
    print("\n" + "=" * 60)
    print("📊 PROJECT STATUS")
    print("=" * 60)
    
    # Raga Generation Status
    print("\n🎵 RAGA GENERATION:")
    gen_model = RAGA_GENERATION / "models" / "small_raga_gen.h5"
    gen_tokenizer = RAGA_GENERATION / "models" / "tokenizer.json"
    gen_data = list((RAGA_GENERATION / "data").glob("*_sequences.json"))
    
    print(f"   Model:     {'✅' if gen_model.exists() else '❌'} {gen_model.name if gen_model.exists() else 'Not found'}")
    print(f"   Tokenizer: {'✅' if gen_tokenizer.exists() else '❌'} {gen_tokenizer.name if gen_tokenizer.exists() else 'Not found'}")
    print(f"   Data:      {'✅' if gen_data else '❌'} {len(gen_data)} sequence files")
    
    # Raga Detection Status
    print("\n🎧 RAGA DETECTION:")
    det_model = RAGA_DETECTION / "models" / "raga_classifier.pkl"
    det_data = RAGA_DETECTION / "data"
    
    print(f"   Model:     {'✅' if det_model.exists() else '❌'} {det_model.name if det_model.exists() else 'Not found'}")
    
    if det_data.exists():
        raga_folders = [d for d in det_data.iterdir() if d.is_dir()]
        print(f"   Data:      {'✅' if raga_folders else '⚠️'} {len(raga_folders)} raga folders")
    else:
        print(f"   Data:      ❌ No data directory")
    
    # Shared Metadata Status
    print("\n📚 SHARED METADATA:")
    metadata_dir = SHARED / "ragas_metadata"
    if metadata_dir.exists():
        raga_files = list(metadata_dir.glob("*.json"))
        print(f"   Ragas:     ✅ {len(raga_files)} raga definitions")
    else:
        print(f"   Ragas:     ❌ No metadata directory")


def main():
    print("=" * 60)
    print("  CARNATIC ML - QUICK START")
    print("  Raga Detection & Generation")
    print("=" * 60)
    
    while True:
        print("\n📋 MENU:")
        print("  1. Show project status")
        print("  2. Test raga generation (works now!)")
        print("  3. Setup raga detection")
        print("  4. Import external raga data")
        print("  5. Check dependencies")
        print("  6. Exit")
        
        choice = input("\nSelect option [1-6]: ").strip()
        
        if choice == '1':
            show_status()
        elif choice == '2':
            check_dependencies() and test_generation()
        elif choice == '3':
            setup_detection()
        elif choice == '4':
            import_external_data()
        elif choice == '5':
            check_dependencies()
        elif choice == '6':
            print("\n👋 Goodbye!")
            break
        else:
            print("Invalid option. Please choose 1-6.")


if __name__ == '__main__':
    main()
