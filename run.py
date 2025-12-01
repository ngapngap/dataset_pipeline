# -*- coding: utf-8 -*-
"""
Dataset Pipeline CLI
Công cụ command-line để chạy pipeline tạo dataset

Sử dụng:
    python run.py                      # Chạy toàn bộ pipeline
    python run.py --steps extract      # Chỉ chạy bước extract
    python run.py --steps generate evaluate  # Chạy generate và evaluate
    python run.py --resume             # Resume từ state đã lưu
    python run.py --config my_config.yaml  # Dùng config khác
"""

import argparse
import sys
import os

# Thêm current dir vào path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import DatasetPipeline
from core.logger import get_logger


logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Dataset Pipeline - Tạo dataset Q&A từ tài liệu",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python run.py                           # Chạy toàn bộ pipeline
  python run.py --steps extract           # Chỉ extract text
  python run.py --steps generate          # Chỉ generate Q&A
  python run.py --steps evaluate export   # Evaluate và export
  python run.py --resume                  # Resume từ checkpoint
  python run.py -c custom_config.yaml     # Dùng config khác
        """
    )
    
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config.yaml",
        help="Đường dẫn tới file config (default: config.yaml)"
    )
    
    parser.add_argument(
        "-s", "--steps",
        nargs="+",
        choices=["extract", "generate", "evaluate", "export"],
        help="Các bước cần chạy (default: tất cả)"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume từ state đã lưu"
    )
    
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Thử lại các chunks thất bại trong bước generate"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Hiển thị log chi tiết"
    )
    
    args = parser.parse_args()
    
    # Check config exists
    if not os.path.exists(args.config):
        print(f"❌ Không tìm thấy file config: {args.config}")
        print("   Hãy tạo file config.yaml hoặc chỉ định đường dẫn với -c")
        sys.exit(1)
    
    try:
        # Initialize pipeline
        print(f"📋 Loading config: {args.config}")
        pipeline = DatasetPipeline(args.config)
        
        if args.retry_failed:
            # Retry failed chunks
            print("🔄 Retry failed chunks...")
            pipeline.generator.retry_failed()
            
        elif args.resume:
            # Resume from saved state
            print("▶️ Resuming pipeline...")
            result = pipeline.resume()
            
        else:
            # Run pipeline
            print("🚀 Starting pipeline...")
            result = pipeline.run(steps=args.steps)
        
        # Print summary
        print("\n" + "="*50)
        print("📊 PIPELINE SUMMARY")
        print("="*50)
        
        if isinstance(result, dict):
            print(f"  Project: {result.get('project', 'N/A')}")
            print(f"  Documents: {result.get('documents', 0)}")
            print(f"  Q&A Generated: {result.get('qa_pairs_generated', 0)}")
            print(f"  Good Q&A: {result.get('good_qa', 0)}")
            print(f"  Bad Q&A: {result.get('bad_qa', 0)}")
            print(f"  Success Rate: {result.get('success_rate', 0):.1f}%")
        
        print("\n✅ Pipeline completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline bị dừng bởi user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Pipeline lỗi: {e}")
        logger.exception("Pipeline error")
        sys.exit(1)


if __name__ == "__main__":
    main()
