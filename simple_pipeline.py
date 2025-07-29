#!/usr/bin/env python3
"""
Simple pipeline for virtual try-on system.
Command-line interface for batch and single processing.
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from demo.run_demo import VirtualTryOnDemo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description='Virtual Glasses Try-On Pipeline')
    parser.add_argument('--mode', choices=['single', 'batch'], required=True,
                       help='Processing mode: single or batch')
    parser.add_argument('--selfie-id', type=int, help='Specific selfie ID to use')
    parser.add_argument('--glasses-id', type=str, help='Specific glasses ID to use')
    parser.add_argument('--batch-size', type=int, default=5,
                       help='Number of combinations for batch mode (default: 5)')
    parser.add_argument('--no-save', action='store_true',
                       help='Skip saving results to disk')
    
    args = parser.parse_args()
    
    # Initialize demo
    demo = VirtualTryOnDemo()
    
    # Check data availability
    status = demo.check_data_status()
    if status['selfies'] == 0 or status['glasses'] == 0:
        logger.error("❌ Insufficient data. Run 'python demo/run_demo.py' first to setup data.")
        return
    
    save_result = not args.no_save
    
    if args.mode == 'single':
        logger.info("🎯 Running single virtual try-on...")
        result = demo.run_single_tryon(
            selfie_id=args.selfie_id,
            glasses_id=args.glasses_id,
            save_result=save_result
        )
        
        if result['success']:
            logger.info(f"✅ Success! Result saved to: {result['result_path']}")
        else:
            logger.error(f"❌ Failed: {result['error']}")
    
    elif args.mode == 'batch':
        logger.info(f"📦 Running batch processing ({args.batch_size} combinations)...")
        results = demo.run_batch_tryon(count=args.batch_size)
        
        successful = sum(1 for r in results if r['success'])
        logger.info(f"✅ Batch complete: {successful}/{args.batch_size} successful")
        
        # Show sample results
        for i, result in enumerate(results[:3]):  # Show first 3
            if result['success']:
                logger.info(f"   {i+1}. {result['result_path']}")

if __name__ == "__main__":
    main()