#!/usr/bin/env python3
"""
Master test runner for all test suites.
Runs all tests and provides comprehensive reporting.
"""

import sys
import os
import subprocess
import time
from datetime import datetime

def run_test_suite(test_file, test_name):
    """Run a single test suite and return results."""
    print(f"\n{'='*60}")
    print(f"🧪 RUNNING {test_name.upper()}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        success = result.returncode == 0
        
        print(f"⏱️  Duration: {duration:.2f} seconds")
        print(f"📊 Exit code: {result.returncode}")
        
        if success:
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
            if result.stderr:
                print(f"Error output: {result.stderr}")
        
        return {
            'name': test_name,
            'success': success,
            'duration': duration,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'exit_code': result.returncode
        }
        
    except subprocess.TimeoutExpired:
        print(f"⏰ {test_name} TIMED OUT (5 minutes)")
        return {
            'name': test_name,
            'success': False,
            'duration': 300,
            'stdout': '',
            'stderr': 'Test timed out after 5 minutes',
            'exit_code': -1
        }
    except Exception as e:
        print(f"💥 {test_name} ERROR: {e}")
        return {
            'name': test_name,
            'success': False,
            'duration': 0,
            'stdout': '',
            'stderr': str(e),
            'exit_code': -2
        }

def main():
    """Run all test suites."""
    print("🚀 COMPREHENSIVE TEST SUITE RUNNER")
    print("=" * 60)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python version: {sys.version}")
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Define test suites
    test_suites = [
        ("test_data_utilities.py", "Data Utilities"),
        ("test_normalization.py", "Normalization"),
        ("test_binance_organizer.py", "Binance Data Organizer"),
        ("test_model_utils.py", "Model Utilities"),
        ("test_merged_organizer.py", "Complete Integration"),
    ]
    
    # Run all tests
    results = []
    total_start_time = time.time()
    
    for test_file, test_name in test_suites:
        test_path = os.path.join(os.path.dirname(__file__), test_file)
        if os.path.exists(test_path):
            result = run_test_suite(test_path, test_name)
            results.append(result)
        else:
            print(f"⚠️  Test file not found: {test_file}")
            results.append({
                'name': test_name,
                'success': False,
                'duration': 0,
                'stdout': '',
                'stderr': f'Test file not found: {test_file}',
                'exit_code': -3
            })
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # Generate summary report
    print(f"\n{'='*60}")
    print("📊 COMPREHENSIVE TEST RESULTS")
    print(f"{'='*60}")
    
    passed = sum(1 for r in results if r['success'])
    total = len(results)
    failed = total - passed
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {failed}/{total}")
    print(f"⏱️  Total duration: {total_duration:.2f} seconds")
    print(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Detailed results
    print(f"\n📋 DETAILED RESULTS:")
    print("-" * 60)
    
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        duration = f"{result['duration']:.2f}s"
        print(f"{status} {result['name']:<25} {duration:>8}")
        
        if not result['success'] and result['stderr']:
            print(f"    Error: {result['stderr'][:100]}...")
    
    # Performance summary
    print(f"\n⚡ PERFORMANCE SUMMARY:")
    print("-" * 60)
    
    avg_duration = sum(r['duration'] for r in results) / len(results) if results else 0
    fastest = min(results, key=lambda x: x['duration']) if results else None
    slowest = max(results, key=lambda x: x['duration']) if results else None
    
    print(f"Average duration: {avg_duration:.2f} seconds")
    if fastest:
        print(f"Fastest test: {fastest['name']} ({fastest['duration']:.2f}s)")
    if slowest:
        print(f"Slowest test: {slowest['name']} ({slowest['duration']:.2f}s)")
    
    # Success rate
    success_rate = (passed / total) * 100 if total > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")
    
    # Overall result
    print(f"\n🎯 OVERALL RESULT:")
    print("-" * 60)
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The merged BinanceDataOrganizer is working perfectly!")
        print("📊 Features verified:")
        print("   ✅ Data downloading and processing")
        print("   ✅ Feature creation and sliding windows")
        print("   ✅ Grouped normalization and scaling")
        print("   ✅ On-demand data generation")
        print("   ✅ Memory management")
        print("   ✅ Model training and prediction")
        print("   ✅ Complete end-to-end integration")
        overall_success = True
    else:
        print(f"💥 {failed} TESTS FAILED! Please check the errors above.")
        overall_success = False
    
    # Save detailed report
    report_file = os.path.join(os.path.dirname(__file__), "test_report.txt")
    with open(report_file, 'w') as f:
        f.write("COMPREHENSIVE TEST REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Python: {sys.version}\n")
        f.write(f"Working directory: {os.getcwd()}\n\n")
        
        f.write(f"SUMMARY:\n")
        f.write(f"Passed: {passed}/{total}\n")
        f.write(f"Failed: {failed}/{total}\n")
        f.write(f"Success rate: {success_rate:.1f}%\n")
        f.write(f"Total duration: {total_duration:.2f} seconds\n\n")
        
        f.write("DETAILED RESULTS:\n")
        f.write("-" * 60 + "\n")
        for result in results:
            f.write(f"\n{result['name']}:\n")
            f.write(f"  Success: {result['success']}\n")
            f.write(f"  Duration: {result['duration']:.2f}s\n")
            f.write(f"  Exit code: {result['exit_code']}\n")
            if result['stdout']:
                f.write(f"  Output:\n{result['stdout']}\n")
            if result['stderr']:
                f.write(f"  Errors:\n{result['stderr']}\n")
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
