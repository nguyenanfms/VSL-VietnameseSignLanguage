# AGENT INSTRUCTION: PERSONAL README & DOCS STYLE GUIDE

Khi viết README.md hoặc tài liệu dự án, Agent phải tuân thủ đúng thói quen và tư duy kỹ thuật sau:

### 1. Tư duy trình bày (Result & Comparison First)
- Không liệt kê dữ liệu thô hoặc mô tả tính năng dàn trải.
- Ưu tiên bảng so sánh trực tiếp: Điểm khác biệt/tối ưu của giải pháp này so với baseline hoặc cách làm thông thường (Performance, Latency, Trade-offs).
- Đi thẳng vào kết quả đo lường và kết luận thay vì giải thích dài dòng.

### 2. Cấu trúc README chuẩn cá nhân
1. **TL;DR / Problem & Impact:** 1-2 câu tóm gọn bài toán, giải pháp và kết quả đạt được.
2. **Architecture & Tech Rationale:**
   - Liệt kê tech stack theo tầng (Frontend, Backend, Data, Security).
   - Nêu ngắn gọn lý do chọn giải pháp kỹ thuật (Trade-off: Vì sao chọn X thay vì Y).
3. **Benchmark / Comparison Table:** Bảng đối chiếu kết quả hoặc tính năng chính.
4. **Clean Setup Guide:**
   - Prerequisites chính xác phiên bản.
   - Các bước copy-paste chạy được ngay (từ clone, config `.env`, migrate đến run dev/prod).
5. **Repo Hygiene & Contribution:**
   - Quy chuẩn commit message (ví dụ: Conventional Commits).
   - Quy định branch và pull request để giữ git history sạch.

### 3. Quy ước định dạng
- Ngắn, dứt khoát, dùng bullet points và code blocks có syntax highlighting chuẩn.
- Bảng so sánh tối đa 3-4 cột trọng tâm, không thừa thãi thông tin.
- Tuyệt đối không dùng văn mẫu mở bài hay lời cảm ơn sáo rỗng.