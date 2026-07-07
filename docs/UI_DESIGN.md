# UI DESIGN — app.py (Streamlit) cho GĐ4
> Bản thiết kế UI (native Streamlit, tối giản kiểu Notion/Linear) + REVIEW NOTES của tôi (4 fix + cách xử state). Builder GĐ4 theo file này.

> ⚠️ **LANGUAGE (locked):** App CHROME = **English** — UI labels, buttons, status messages, sidebar headers, `review.xlsx` column headers + instructions, file names, README/docs (international Capstone). Ví dụ: nút "Mine"/"Explain"/"Expand"; placeholder "Enter a word, sentence, or request..."; status "Agent is analyzing..."; sidebar "NETWORK STATS / WORKFLOW / EXPORT"; HITL caption "Edit status in Excel, then click Commit".
> **NHƯNG hội thoại với agent = BẤT KỲ ngôn ngữ nào** — user gõ query tiếng nào, agent trả lời tiếng đó (LLM đa ngôn ngữ; ĐỪNG ép/giới hạn ngôn ngữ cuộc trò chuyện). Nội dung thẻ Anki / từ vựng = English (ngôn ngữ đích). Video demo nên dùng query English hoặc phụ đề cho giám khảo.

## 1. Theme — `.streamlit/config.toml`
```toml
[theme]
primaryColor = "#051C2C"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F4F6F8"
textColor = "#222222"
font = "sans serif"
```
Layout `layout="wide"`, cột đệm `st.columns([1.5, 7, 1.5])` — nội dung chính ở cột giữa (7).

## 2. Wireframe
**Trạng thái A — chờ nhập:**
```
SIDEBAR (#F4F6F8)        | MAIN (cột giữa 7.0)
NETWORK STATS            |   VOCABGRAPH-AGENT
  Nodes: 42              |   [ Nhập từ/câu/yêu cầu... ]   (text_input, label collapsed)
  Edges: 108             |   [ ] Attach Media  -> hiện file_uploader khi tick
WORKFLOW ACTIONS         |   [ Mine ]  [ Explain ]  [ Expand ]   (st.columns(3)+button)
  Open Excel             |
  Commit Approved        |
EXPORT DATA              |
  Download .apkg         |
```
**Trạng thái B — kết quả (sau khi chạy):**
```
SIDEBAR                  | MAIN
NETWORK STATS            |   VOCABGRAPH-AGENT  + input + 3 nút
  Nodes: 45 (+3)         |   ----- divider -----
  Edges: 112 (+4)        |   [st.status: "Hoàn tất" — gập gọn]   <- trajectory thật
WORKFLOW ACTIONS         |   **Executive Summary: trích 3 node mới**
  Open Excel             |   +-----------------------------+
  Commit Approved        |   |     PYVIS GRAPH (height>=600)|  <- điểm nhấn thị giác
EXPORT DATA              |   +-----------------------------+
  Download .apkg         |   ----- divider -----
                         |   > Anki Drafts (expander -> dataframe)
```

## 3. Bảng vùng-UI → widget Streamlit (native only)
| Vùng | Widget | Ghi chú |
|---|---|---|
| Cột đệm | `st.columns([1.5,7,1.5])` | nội dung vào cột giữa |
| Input | `st.text_input` | `label_visibility="collapsed"`, placeholder |
| Media | `st.checkbox` + `st.file_uploader` | `if checkbox: uploader` (giấu khi không cần) |
| 3 nút | `st.columns(3)` + `st.button` | `use_container_width=True` |
| Phân cách | `st.divider` | tối đa 2 |
| **Trajectory** | **`st.status`** | `expanded=True` khi chạy; `.update(state="complete", expanded=False)` khi xong |
| Summary | `st.markdown` | in đậm 1 dòng |
| **Graph** | `st.components.v1.html` | `height>=600`, đọc `output/<run>/graph.html` (nền trắng) |
| Anki drafts | `st.expander` + `st.dataframe` | xem trước thẻ |
| Sidebar stats | `st.sidebar.text` | KHÔNG `st.metric`; in text thuần `Nodes: 42` |
| HITL | `st.sidebar.button` | "Open review.xlsx"; "Commit Approved" `type="primary"`; caption hướng dẫn |
| Download | `st.download_button` | .apkg |

## 4. Trajectory (st.status) — mẫu
```python
with st.status("Agent đang phân tích...", expanded=True) as status:
    for step in result["trajectory"]:          # <- DỮ LIỆU THẬT, không script cứng (fix 1)
        st.write(f"Tool: {step['tool']}  {step.get('args','')}")
    status.update(label="Hoàn tất", state="complete", expanded=False)
```

## 5. UX notes (giới hạn Streamlit native)
- Khoảng trắng: dùng `st.write("")`/`st.empty()` (KHÔNG `unsafe_allow_html`) (fix 4).
- Mở Excel: `os.startfile(path)` (Windows) / `subprocess` (mac). UI không đổi state, đợi user sửa file rồi bấm Commit.
- Typography: `###` tên app, **đậm** cho tiêu đề sidebar, text thường cho nội dung.

---

## REVIEW NOTES (của reviewer — BẮT BUỘC theo khi build)
**4 chỉnh so với design gốc:**
1. **Trajectory render từ `out["trajectory"]` THẬT** (post-hoc), không script cứng → mới chứng minh "agent tự chọn tool".
2. **Nút "EXTRACT" → "Mine"/"Tạo từ video"** (là cả pipeline ingest→extract→enrich→graph+anki, không phải tool extract đơn lẻ).
3. **Phân vai rõ:** ô text = free query → `run_agent` (LLM tự chọn tool); 3 nút Mine/Explain/Expand = `run_intent` (fallback cố định).
4. Bỏ `unsafe_allow_html`, dùng `st.write("")`/`st.empty()`.

**Xử lý STATE khi reload (câu hỏi của designer) — disk-as-truth:**
- `data/personal_graph.json` = nguồn sự thật. Stats = `load_graph()` đọc lại MỖI rerun (luôn tươi).
- `st.session_state` chỉ giữ con trỏ nhẹ (run_id, path graph.html). KHÔNG giữ data nặng.
- **Commit Approved** → `read_xlsx(review.xlsx)` → hàng `status=approved` → `upsert(Node)` → `save_graph()` → **chạy lại `build_render_graph`** trên graph mới → ghi graph.html mới → cập nhật session_state → rerun → graph mới hiện. (Điểm ghi graph DUY NHẤT.)
- No-key: nút deterministic (recall/wordnet/expand) chạy; mine/explain cần key → báo rõ.
