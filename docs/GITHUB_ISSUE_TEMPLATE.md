# 📸 Add Project Screenshots to Documentation

## Description
This issue tracks the addition of high-quality screenshots to the project README and documentation. Screenshots showcase the key features of the Hallucination Detector web UI and help users understand the system's functionality at a glance.

## Screenshots to Add

### 1. **Landing Page** (`01-landing-page.png`)
- **Current Location**: Localhost:5000 hero section
- **Purpose**: Shows the main UI with title, subtitle, and initial claim generation interface
- **Size**: Full viewport width, ~800px height
- **Content**: 
  - Title: "Hallucination Detector"
  - Subtitle: "Multi-model NLI verification pipeline..."
  - Topic input field with "Generate" button

### 2. **Generate Claims Form** (`02-generate-claims.png`)
- **Current Location**: Claims generation section
- **Purpose**: Demonstrates how users input a topic
- **Content**:
  - Topic input field (e.g., "Apple")
  - Claim count selector (3 claims)
  - "Generate" button
  - Example chips below (SpaceX Mars Mission, Albert Einstein, etc.)

### 3. **Generated Claims Table** (`03-generated-claims-table.png`)
- **Current Location**: Below generation form
- **Purpose**: Shows generated claims in tabular format
- **Content**:
  - Claims table with columns: #, Claim, Status, Consensus Verdict, Details
  - 3 example claims about Apple
  - "Fact-Check All" button in top right

### 4. **Multi-Model Comparison** (`04-multi-model-comparison.png`)
- **Current Location**: Results section (after verification)
- **Purpose**: Core feature - shows 3 different NLI models side-by-side
- **Content**:
  - 3 cards: BART Large MNLI, RoBERTa Large MNLI, DistilBERT MNLI
  - Each card shows:
    - Entailment score (green bar)
    - Contradiction score (red bar)
    - Neutral score (amber bar)
    - Best Similarity score
    - Final verdict badge (SUPPORTED or HALLUCINATED)
  - Consensus warning banner: "⚠️ Models disagree — review individual results below"

### 5. **Aggregated Metrics Table** (`05-aggregated-metrics.png`)
- **Current Location**: Below multi-model comparison
- **Purpose**: Shows aggregated verification data
- **Content**:
  - Table with columns: Model, Final Score, Avg Entailment, Max Contradiction, Similarity
  - Rows for each of the 3 models

### 6. **Retrieved Evidence Table** (`06-retrieved-evidence.png`)
- **Current Location**: Evidence section (bottom)
- **Purpose**: **NEW FEATURE** - Shows retrieved evidence with Retriever Score column
- **Content**:
  - Table with columns: #, Title, Text Extract, **Retriever Score** (NEW!)
  - 6 rows of evidence from Wikipedia
  - Retriever scores in blue monospace font (e.g., 0.773, 0.856, 0.666)

---

## Implementation Tasks

- [ ] Take screenshot of landing page (`01-landing-page.png`)
- [ ] Take screenshot of generate claims form (`02-generate-claims.png`)
- [ ] Take screenshot of generated claims table (`03-generated-claims-table.png`)
- [ ] Take screenshot of multi-model comparison (`04-multi-model-comparison.png`)
- [ ] Take screenshot of aggregated metrics (`05-aggregated-metrics.png`)
- [ ] Take screenshot of retrieved evidence table (`06-retrieved-evidence.png`)
- [ ] Create `docs/screenshots/` directory in repo
- [ ] Add all 6 PNG files to `docs/screenshots/`
- [ ] Update README.md with screenshot references (in progress)
- [ ] Add screenshot gallery table to README
- [ ] Commit and push changes

---

## Directory Structure

```
docs/
├── screenshots/
│   ├── 01-landing-page.png
│   ├── 02-generate-claims.png
│   ├── 03-generated-claims-table.png
│   ├── 04-multi-model-comparison.png
│   ├── 05-aggregated-metrics.png
│   └── 06-retrieved-evidence.png
└── GITHUB_ISSUE_TEMPLATE.md
```

---

## Acceptance Criteria

✅ All 6 screenshots captured and saved in `docs/screenshots/`  
✅ README.md updated with image references in appropriate sections  
✅ Screenshots are clear, readable, and represent the UI accurately  
✅ Image links work correctly on GitHub  
✅ Changes committed and pushed to main branch  

---

## Related Issues/PRs

- Linked to: "Add Retriever Score column to Retrieved Evidence table in UI" (#PR-01)

---

## Notes

- Screenshots were generated on **macOS** with the project running at `http://localhost:5000`
- Recommended file format: **PNG** for lossless quality
- Recommended resolution: **1440x900px** or higher for clarity
- All screenshots show the dark theme UI

---

**Created**: April 20, 2026  
**Assignee**: @Shariq80  
**Labels**: `documentation`, `screenshots`, `ui`
