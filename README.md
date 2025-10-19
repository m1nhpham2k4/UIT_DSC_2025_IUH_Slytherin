# UIT_DSC_2025_IUH_Slytherin

---

## ⚠️Do phương pháp sử dụng LLM nên kết quả đôi khi bị lệch 1 vài sample (nếu chạy đi chạy lại)

---

## Overview

Dự án này là một hệ thống phát hiện ảo giác (hallucination) trong các mô hình ngôn ngữ tiếng Việt.
Hệ thống được xây dựng dựa trên mô hình nền Qwen3-4B-Base, được fine-tune bằng kỹ thuật LoRA (Low-Rank Adaptation) để tối ưu bộ nhớ và tốc độ huấn luyện.

Đặc biệt, dự án kết hợp chiến lược chỉnh sửa lm_head – chỉ giữ lại các trọng số tương ứng với các token biểu diễn nhãn số (1, 2, 3) – giúp mô hình tập trung vào nhiệm vụ phân loại thay vì sinh văn bản.
Sau khi fine-tune, phần lm_head được khôi phục lại đầy đủ để mô hình có thể thực hiện suy luận (inference) chính xác.

Giai đoạn suy luận (inference) được hiệu chỉnh thêm bằng phép hiệu chỉnh bias động (bias calibration), tìm bộ bias tối ưu nhằm cân bằng xác suất giữa ba lớp:

- **`class 1 – no`**: câu trả lời nhất quán với ngữ cảnh, không thêm thông tin ngoài.

- **`class 2 – intrinsic`**: câu trả lời mâu thuẫn hoặc bóp méo thông tin trong ngữ cảnh.

- **`class 3 – extrinsic`**: câu trả lời thêm thông tin ngoài ngữ cảnh, dù có thể đúng trong thực tế.