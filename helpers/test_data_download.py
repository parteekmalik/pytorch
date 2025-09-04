#!/usr/bin/env python3
"""
Test data download functionality.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add utils to path
sys.path.append('/Users/parteekmalik/github/pytorch')

def test_data_download():
    """Test data download functionality."""
    print("🧪 TESTING DATA DOWNLOAD")
    print("=" * 50)
    
    try:
        from utils import download_binance_klines_data, download_binance_vision_data, load_multiple_months_data
        
        # Test 1: Binance Vision download
        print("📥 Test 1: Binance Vision download...")
        df_vision = download_binance_vision_data("BTCUSDT", "5m", "2021", "01")
        
        if df_vision is not None:
            print(f"✅ Binance Vision: Downloaded {len(df_vision)} rows")
            print(f"   Columns: {list(df_vision.columns)}")
            print(f"   Date range: {df_vision['Open time'].min()} to {df_vision['Open time'].max()}")
        else:
            print("❌ Binance Vision download failed")
        
        # Test 2: Binance API download
        print(f"\n📥 Test 2: Binance API download...")
        df_api = download_binance_klines_data("BTCUSDT", "5m", "2021", "01")
        
        if df_api is not None:
            print(f"✅ Binance API: Downloaded {len(df_api)} rows")
            print(f"   Columns: {list(df_api.columns)}")
            print(f"   Date range: {df_api['Open time'].min()} to {df_api['Open time'].max()}")
        else:
            print("❌ Binance API download failed")
        
        # Test 3: Multiple months download
        print(f"\n📥 Test 3: Multiple months download...")
        df_multi = load_multiple_months_data(
            "BTCUSDT", "5m", "2021", ["01", "02"], 
            max_rows=10000  # Uses Binance Vision only
        )
        
        if df_multi is not None:
            print(f"✅ Multiple months: Downloaded {len(df_multi)} rows")
            print(f"   Columns: {list(df_multi.columns)}")
            print(f"   Date range: {df_multi['Open time'].min()} to {df_multi['Open time'].max()}")
        else:
            print("❌ Multiple months download failed")
        
        # Summary
        print(f"\n📊 Download Test Summary:")
        print(f"   Binance Vision: {'✅' if df_vision is not None else '❌'}")
        print(f"   Binance API: {'✅' if df_api is not None else '❌'}")
        print(f"   Multiple months: {'✅' if df_multi is not None else '❌'}")
        
        success_count = sum([df_vision is not None, df_api is not None, df_multi is not None])
        print(f"   Success rate: {success_count}/3")
        
        return success_count > 0
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def main():
    """Run data download tests."""
    print("🧪 TESTING DATA DOWNLOAD FUNCTIONALITY")
    print("=" * 60)
    
    success = test_data_download()
    
    if success:
        print(f"\n🎉 Data download tests completed!")
        print(f"   At least one download method is working")
        return True
    else:
        print(f"\n❌ All download methods failed")
        print(f"   Check your internet connection and try again")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Data download is working!")
        sys.exit(0)
    else:
        print("\n❌ Data download issues found.")
        sys.exit(1)
