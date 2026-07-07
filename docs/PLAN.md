# PLAN — Working Flow (authoritative)
> Bản kế hoạch CHỐT cho dự án Capstone. Nếu mâu thuẫn file cũ → ưu tiên file này. Cập nhật khi có quyết định mới.

## Định vị (CHỐT)
**"Agent học từ vựng đồng hành: xem phim/audio → dựng graph từ vựng cá nhân (mọc dần) + thẻ Anki, và hỏi/mở rộng mọi kiến thức đã học."**
- Track: **Agents for Good (Education)**. Solo, ~13 ngày từ 2026-06-23.
- Input v1: **video/audio** (tái dùng pipeline tool cũ — đã test).
- AI agent **sống ở lớp TƯƠNG TÁC** (hỏi/mở rộng/recall), không phải lớp ingest. Ingest = pipeline tất định (OK).

## V1 vs Vision
- **V1:** video/audio · extract→enrich→graph+anki · personal graph mọc dần · agent hỏi/mở rộng (ý định giới hạn, có rào).
- **Vision:** chatbot tự do + LLM điều phối hoàn toàn · multi-input (PDF/ảnh) · đa ngôn ngữ · i+1/known-words · vòng phản hồi điểm · shadowing.

## Kiến trúc (Agent = Model + Harness)
```
TOOLS (8): recall · ingest_transcript · extract_vocab · wordnet_lookup
           · enrich · build_render_graph · make_anki · explain
AGENT    : LLM tool-calling loop (đọc SKILL.md + mô tả tool), bounded (cap 5-8 vòng, hỏi-khi-mơ-hồ)
HARNESS  : AGENTS.md (luật) · PersonalGraph JSON (memory, mọc dần) · HITL review.xlsx · .env (no key)
```
Nguyên tắc: **deterministic-first** (WordNet cho cạnh chắc → AI chỉ chọn nghĩa + cột không chắc, flag) · **schema-driven** (validate Pydantic) · **mỗi tool no-crash** · **code là đồ bỏ, spec là vàng**.

## Concept khóa học phủ (cổng ≥3, mình 5)
MCP · Agent Skills (SKILL.md) · Agent=Model+Harness/Context-Eng (AGENTS.md) · Security (context hygiene + .env) · Spec-Driven (schema + Gherkin).

## Bám rubric (100)
Pitch 30 (concept 10 + video 10 + writeup 10) · Implementation 70 (technical 50 + docs 20). Dự phóng nếu làm ổn ~78-88. Sàn (agent→fallback nút) ~60+.

## Build phases (spec-first, timebox, fallback)
- **GĐ0 (N1):** `schema.py` + `docs/TOOLS.md` + file vệ sinh (.gitignore/.env.example/requirements/README stub) + chốt framework=Claude tool-use.
- **GĐ1 (N2):** `AGENTS.md` + `SKILL.md` + copy module cũ (không sửa).
- **GĐ2 (N3-6):** code tool: recall·extract(tái dùng)·wordnet·enrich → build_render_graph·make_anki(Cloze+Basic+dictation). **Mốc N6: pipeline chạy tay.**
- **GĐ3 (N7-8):** MCP + agent-loop (hỏi/mở rộng/recall). **FALLBACK N8: chưa ổn → nút intent.**
- **GĐ4 (N9-10):** personal graph bền + HITL + security hygiene + trajectory log + test + UI + deploy (GitHub+README+setup) + sơ đồ.
- **Đệm N11 · Video N12 · Writeup+nộp N13.**

## Thứ tự viết file (phụ thuộc)
1. `schema.py` (TRƯỚC — hợp đồng) → 2. TOOLS.md → 3. AGENTS.md → 4. SKILL.md.
Nhóm vệ sinh (.gitignore/.env.example/requirements/README) = độc lập, viết kèm lúc nào cũng được.

## Quy tắc vừa-code-vừa-note (QUAN TRỌNG)
Mỗi phase xong → note ngay vào draft:
- Quyết định/đánh đổi → `docs/WRITEUP_DRAFT.md §7`
- Lệnh setup thật, cây thư mục → `docs/README_DRAFT.md`
- Cảnh demo chạy → `docs/PITCH_SCRIPT_DRAFT.md §2:00` + `WRITEUP §8`
→ Cuối chỉ gọt final, không viết từ đầu.

## make_anki (CHỐT)
Cloze (đục từ, native) + Basic (từ→nghĩa) + Dictation (nghe→gõ, type-in) · media = **screenshot + audio** (ASCII filename) · GUID ổn định.

## Fallback nhiều tầng
agent-loop chưa ổn → nút intent · graph mọc dần quá sức → graph per-film · input phim → đã test.

---

## COURSE-ALIGNED ADDITIONS (sau khi đọc 5 day material + livestream)
Tài liệu gốc đã convert: `course_md/` (Day 1-5). Chốt bổ sung bám chuẩn khóa học:

**Cấu trúc repo (Spec-Driven, Day 5):**
```
capstone/
├── schema.py                      # ✅ done
├── AGENTS.md                      # ✅ done (static context, harness)
├── skills/                        # Agent Skills (Day 3) — progressive disclosure
│   ├── building-vocab-graph/SKILL.md       # ✅ done
│   └── expanding-vocab-knowledge/SKILL.md  # ✅ done
├── specs/                         # Spec-Driven (Day 5)
│   ├── DESIGN.md                  # narrative thiết kế
│   ├── features/*.gherkin         # Given-When-Then: ffmpeg_clip, enrich, error_handling, hitl
│   └── config/*.yaml              # schema/pipeline/execution_policy (YAML cho config >3 tầng)
├── docs/TOOLS.md                  # 8 tool: description+schema+scope+cost+error (Day 2/MCP)
├── tools/  agent/  app.py ...     # code (GĐ2+)
└── .env.example · .gitignore · requirements.txt · README.md
```

**Skills (Day 3):** 2 skill `building-vocab-graph` + `expanding-vocab-knowledge`. Mỗi SKILL.md = frontmatter (description=routing, 3 trigger+3 anti-trigger) + body (When use/NOT/Workflow/Examples/Output/Anti-patterns). Quy tắc: 1 skill 1 việc · skill phải có test.

**MCP (Day 2):** transport **stdio** (đơn giản nhất). Mỗi tool: description (routing) + JSON schema in/out (từ schema.py) + scope (read/write) + token-cost + error-per-tool + log. Tool đọc (recall/wordnet/explain) = read-only; tool ghi (ingest/enrich/build/anki) = scoped local.

**Security (Day 4, nhẹ — đủ điểm, không overkill):**
- sandbox subprocess + timeout + no-network cho code sinh ra
- `pip-audit`/whitelist → chống **slopsquatting** (lib ảo giác)
- JIT `.env` deny-by-default (chỉ ghi trong project)
- context hygiene: mask key/PII trong log
- **Bỏ:** red/blue/green teams, AgBOM, hardware MFA, 7-pillar đầy đủ.

**Evals (Day 1+4 — "thiếu eval = vẫn vibe coding"):**
- **tests** (tất định): schema valid, graph không node mồ côi, normalize_timestamp...
- **trajectory log** JSON (mỗi tool-call) + inspect tay vài phiên
- **LLM-as-judge**: agent chọn nghĩa đúng %? (golden set ~10-20 từ)
- ⭐ **mining user corrections (K-Means)** từ các dòng `rejected`/`needs_revision` trong review.xlsx → top failure modes (tái dùng HITL, ăn điểm Day 4)

**Execution modes (Day 5) dùng khi code:** Architect (scaffold repo, pin version) → Builder (theo style cũ, show diff) → Forensic (fix bằng evidence/log, không dọn linh tinh) → Author (đồng bộ docs).

**Ranh giới (Day 3):** Skill = *how* (know-how) · MCP/tools = *access* (hands) · AGENTS.md = project conventions (always-on). Test: xóa skill mà model vẫn làm được (vụng) = ranh giới sạch.
