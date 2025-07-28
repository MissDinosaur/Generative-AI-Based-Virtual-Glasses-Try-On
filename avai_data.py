"""
Quick script to show what data is available in your database.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from database_config import db


def show_available_data():
    """Show all available data in a nice format."""
    print("🔍 Exploring Your Database")
    print("=" * 50)
    
    # Test connection
    if not db.test_connection():
        print("❌ Database connection failed!")
        return
    
    # Get counts
    counts = db.get_data_counts()
    print(f"📊 Total Data:")
    print(f"   Selfies: {counts['selfies']}")
    print(f"   Glasses: {counts['glasses']}")
    
    # Show diverse selfies
    print(f"\n📸 Available Selfies (random sample):")
    selfies = db.get_selfies(limit=15, random_order=True)
    
    if not selfies.empty:
        for i, (_, row) in enumerate(selfies.iterrows()):
            print(f"   {i+1:2d}. ID: {row['id']:<6} | {row['filename']:<25} | {row['image_width']}x{row['image_height']}")
    else:
        print("   No selfies found")
    
    # Show diverse glasses  
    print(f"\n🕶️ Available Glasses (random sample):")
    glasses = db.get_glasses(limit=15, random_order=True)
    
    if not glasses.empty:
        for i, (_, row) in enumerate(glasses.iterrows()):
            brand = (row['brand'] or 'Unknown')[:12]
            title = (row['title'] or 'Unknown')[:30]
            shape = row.get('frame_shape', 'N/A') or 'N/A'
            print(f"   {i+1:2d}. {brand:<12} | {title:<30} | {shape:<10}")
    else:
        print("   No glasses found")
    
    # Show example commands
    print(f"\n💡 Try These Commands:")
    if not selfies.empty and not glasses.empty:
        # Show 3 different selfie examples
        print(f"   🎲 Random combinations (different each time):")
        print(f"   python simple_pipeline.py --mode single")
        print(f"   python simple_pipeline.py --mode single") 
        print(f"   python simple_pipeline.py --mode single")
        
        print(f"\n   📸 Try specific selfies:")
        for _, row in selfies.head(3).iterrows():
            print(f"   python simple_pipeline.py --mode single --selfie-id {row['id']}")
        
        print(f"\n   🕶️ Try specific glasses:")
        for _, row in glasses.head(3).iterrows():
            glasses_id = str(row['id'])
            print(f"   python simple_pipeline.py --mode single --glasses-id \"{glasses_id}\"")
        
        print(f"\n   📦 Batch processing (multiple random combinations):")
        print(f"   python simple_pipeline.py --mode batch --batch-size 10")
    
    print(f"\n🎮 Interactive Demo:")
    print(f"   python demo/simple_demo.py")


if __name__ == "__main__":
    show_available_data()