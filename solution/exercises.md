# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?** (2–3 câu)
> * Khi tăng temperature từ 0.0 → 1.8, câu trả lời trở nên ngày càng đa dạng và sáng tạo hơn. Ở 0.0, phản hồi ổn định và mang tính sự thật; khoảng 0.7–1.2 vẫn mạch lạc nhưng có cách diễn đạt phong phú hơn. Đến 1.8, mô hình dễ thêm các chi tiết suy diễn hoặc diễn đạt lan man, nên chất lượng và tính mạch lạc bắt đầu giảm.*

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> * Trợ lý soạn thảo hợp đồng pháp lý: temperature = 0.0–0.2 để đảm bảo câu trả lời nhất quán, chính xác và hạn chế sáng tạo không cần thiết. Trợ lý viết slogan quảng cáo: temperature = 1.0–1.2 để tạo ra nhiều ý tưởng mới, câu chữ đa dạng và giàu tính sáng tạo.Sự khác biệt là hợp đồng pháp lý ưu tiên độ chính xác và tính ổn định, còn slogan quảng cáo ưu tiên sự sáng tạo và tính độc đáo.*

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
> * Với 20.000 người dùng/ngày, mỗi người gọi API 2 lần và mỗi lần sinh trung bình 500 output tokens, hệ thống tạo ra khoảng 20 triệu output tokens/ngày (tương đương 20.000 đơn vị 1.000 tokens). Theo bảng giá, chi phí ước tính là:GPT-4o: khoảng 200 USD/ngày. GPT-4o-mini: khoảng 12 USD/ngày Model lớn (GPT-4o) phù hợp khi cần khả năng suy luận và độ chính xác cao, chẳng hạn trợ lý AI cho y tế, pháp lý hoặc phân tích dữ liệu phức tạp. Model nhỏ (GPT-4o-mini) là lựa chọn phù hợp cho các ứng dụng có lượng truy cập lớn và tác vụ đơn giản như chatbot FAQ, hỗ trợ khách hàng hoặc tóm tắt văn bản, giúp tiết kiệm đáng kể chi phí vận hành.*

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)
> * Hai phản hồi có sự khác biệt rõ rệt về phong cách và mức độ kỹ thuật. Với persona nhà thơ, câu trả lời giàu hình ảnh ví von, ngôn ngữ mềm mại và hầu như không sử dụng thuật ngữ chuyên môn. Với persona kỹ sư phần mềm senior, câu trả lời chính xác, có cấu trúc rõ ràng, sử dụng các khái niệm kỹ thuật và có thể kèm ví dụ code minh họa khi phù hợp. Điều này cho thấy system prompt có thể điều khiển phong cách diễn đạt, giọng văn, mức độ chi tiết, mức độ chuyên môn, cách trình bày và định dạng của phản hồi.*

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**
> * Khi so sánh trên một đoạn văn tiếng Việt khoảng 150 từ, số token do tiktoken đếm thường khác với giá trị ước lượng theo công thức số từ / 0.75. Thông thường, sai lệch vào khoảng 5–15% (tùy nội dung và số lượng dấu câu, từ ghép, ký tự tiếng Việt). Nếu dùng công thức ước lượng thô để dự toán ngân sách API cho ứng dụng tiếng Việt, nhiều trường hợp sẽ dự toán thiếu, vì token thực tế còn phụ thuộc vào cách tokenizer tách dấu, ký tự Unicode và dấu câu, không chỉ dựa trên số từ. Do đó, khi cần tính chi phí chính xác nên sử dụng tiktoken thay vì ước lượng bằng số từ*

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> * Streaming mang lại lợi ích lớn nhất cho (b) trợ lý giọng nói vì hệ thống có thể bắt đầu đọc từng phần phản hồi ngay khi được sinh ra, giúp giảm đáng kể độ trễ người dùng cảm nhận. (a) Chatbot văn bản cũng hưởng lợi vì người dùng thấy câu trả lời xuất hiện dần thay vì phải chờ toàn bộ nội dung được tạo xong. Ngược lại, (c) pipeline dịch tài liệu chạy ngầm ban đêm hầu như không cần streaming vì không có người dùng tương tác trực tiếp; điều quan trọng là độ chính xác, tính ổn định và hoàn thành toàn bộ tác vụ hơn là phản hồi tức thời.*

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**
> * Khi API quá tải và nhiều client cùng retry, exponential backoff sẽ tăng dần thời gian chờ sau mỗi lần thất bại, giúp giảm số lượng yêu cầu gửi lại cùng lúc và tạo thời gian để máy chủ phục hồi. Nếu sử dụng delay cố định, các client thường retry đồng thời, dễ gây thêm quá tải và kéo dài thời gian khôi phục. Kỹ thuật jitter bổ sung một khoảng trễ ngẫu nhiên vào mỗi lần retry để tránh hiện tượng nhiều client gửi lại yêu cầu cùng thời điểm (retry storm hoặc thundering herd), từ đó phân tán lưu lượng và tăng khả năng thành công của các lần thử lại.*

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**
> * System prompt:"Bạn là trợ giảng thân thiện của khóa học AI. Hãy trả lời bằng tiếng Việt, ngắn gọn, chính xác, giải thích dễ hiểu và ưu tiên đưa ví dụ minh họa khi cần." Hai chỗ nếu xóa sẽ làm thay đổi hành vi rõ rệt:"trả lời bằng tiếng Việt"Nếu bỏ phần này, trợ lý có thể trả lời bằng tiếng Anh hoặc ngôn ngữ khác tùy theo câu hỏi của người dùng hoặc ngữ cảnh.Hành vi thay đổi rõ rệt vì ngôn ngữ phản hồi không còn được cố định."ngắn gọn" Nếu bỏ phần này, trợ lý sẽ có xu hướng đưa ra các câu trả lời dài hơn, giải thích chi tiết hơn và có thể bổ sung nhiều thông tin ngoài yêu cầu.Hành vi thay đổi ở độ dài và mức độ chi tiết của câu trả lời.*

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
> * Tình huống: Người dùng trò chuyện nhiều hơn 4 lượt.Ví dụ:Lượt 1: "Tên tôi là Quang." .Lượt 2: "Tôi học ngành Công nghệ thông tin."Lượt 3: "Tôi đang làm dự án AI."Lượt 4: "Tôi thích Python."Lượt 5: "Tôi đang ôn phỏng vấn."Lượt 6: "Tên tôi là gì?"Do chương trình chỉ giữ 4 lượt hội thoại cuối (8 message), thông tin ở lượt 1 đã bị loại khỏi history. Vì vậy trợ lý sẽ không còn nhớ tên "Quang" và có thể trả lời rằng không biết hoặc yêu cầu người dùng cung cấp lại thông tin.Cách khắc phục:Thay vì chỉ giữ 4 lượt gần nhất, có thể tóm tắt các lượt hội thoại cũ trước khi xóa. Ví dụ lưu một đoạn như:"Người dùng tên Quang, học ngành Công nghệ thông tin, đang làm dự án AI."Sau đó đưa phần tóm tắt này vào system hoặc history cùng với các lượt hội thoại mới. Cách này giúp giảm số lượng token nhưng vẫn giữ được các thông tin quan trọng của cuộc trò chuyện. *

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)
