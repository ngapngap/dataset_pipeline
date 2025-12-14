<!-- b5b73a58-bd77-4ad9-a1cc-f2f40e61764c b3360e6e-2ed3-4401-a728-a40caf1eb2fb -->
# Cleanup Dataset Pipeline Project

## Scope

Dự án này là **Dataset Pipeline** - tạo dataset Q&A từ văn bản pháp luật.
Fine-tune là project riêng biệt.

## Danh sách đã xóa

- [x] `nul` - File rác từ git
- [x] `rescue_bad_qa.py` - Legacy script
- [x] `data/` - Folder rỗng
- [x] `output/raw_qa/` - Folder rỗng
- [x] `pipeline_v2.py` - File legacy

## Còn lại cần xóa

- [ ] `output/tokenized/` - Folder rỗng

## Không xóa

- `__pycache__/` - Auto-generated, đã có trong `.gitignore`
- Tất cả các file/folder khác đang được sử dụng

### To-dos

- [x] Evaluate project and rewrite README.md
- [ ] Check other documentation files (optional)
- [x] Tạo cấu trúc thư mục finetune/ và các file cơ bản
- [x] Implement config.yaml và core/config.py
- [x] Implement core/data_loader.py - load dataset từ pipeline_v2
- [x] Implement core/trainer.py - training logic với Unsloth
- [x] Implement core/evaluator.py - đánh giá và báo cáo
- [x] Implement finetune.py - entry point
- [x] Tạo config presets cho 14B và 7B
- [x] Viết README.md hướng dẫn sử dụng