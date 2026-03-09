# EscalationLab

A hands-on lab that teaches the complete progression from connecting language models to domain data through model adaptation — governed by one principle: **escalation of effort must be justified by evidence**.

Uses the [Basic Fantasy RPG](https://www.basicfantasy.org/) rulebook as a realistic test domain to demonstrate how grounding LLMs with retrieved context dramatically improves answer quality, and what to do when retrieval alone isn't enough.

## Structure

| Section | Directory | Topic |
|---------|-----------|-------|
| 00 | `00_Orientation/` | Orientation and guardrails — set the mental model before touching code |
| 01 | `01_CustomerScenario/` | Frame a realistic customer engagement scenario |
| 02 | `02_Baseline/` | Configure API access, test connectivity, establish baseline responses |
| 03 | `03_Ingestion/` | Convert PDF documents to Markdown using [Docling](https://github.com/DS4SD/docling) |
| 04 | `04_RAG/` | Token counting, chunking, embeddings, vector search, and full RAG pipeline |
| 05 | `05_Evaluation/` | Compare baseline vs. RAG using manual review and LLM-as-judge scoring |
| 06 | `06_Scaling/` | Scale ingestion to multiple documents with source attribution |
| 07 | `07_InferenceTimeScaling/` | Best-of-N sampling — test whether smarter inference closes the gap |
| 08 | `08_SyntheticDataGen/` | Generate structured training data from source documents using [SDG Hub](https://github.com/red-hat-ai-innovation/sdg-hub) |
| 09 | `09_ModelAdaptation/` | Fine-tune with QLoRA using the synthetic data from Section 08 |
| 10 | `10_ComprehensiveEvaluation/` | Evaluate fine-tuned models across architectures and sizes |
| 11 | `11_Synthesis/` | Translate lab results into customer-facing language (facilitated discussion) |

## The Escalation Ladder

The lab follows a deliberate sequence. Each step produces evidence that informs whether the next step is justified:

```
Baseline → Ingestion → Retrieval → Evaluation → Scaling
    ↓ (only if evidence warrants)
Inference-Time Scaling → Synthetic Data → Fine-Tuning → Comprehensive Evaluation
```

You do not fine-tune because it sounds impressive. You fine-tune because your evaluation told you the model is the bottleneck, and you can point to exactly where and why.

## Getting Started

### Prerequisites

- Python 3.12+
- JupyterLab
- API key and endpoint for a MaaS (Model as a Service) instance running Granite models
- NVIDIA GPU (L40S with 46 GB recommended; L4 with 24 GB minimum for fine-tuning sections)

### Setup

1. Create a `.env` file in the project root:

   ```
   API_KEY=your-api-key
   ENDPOINT_BASE=https://your-maas-endpoint/v1
   ```

2. Install dependencies within the notebooks as needed (each notebook installs its own requirements).

3. Work through the notebooks in order, starting with `00_Orientation/00_Orientation_and_Guardrails.ipynb`.

## Key Technologies

- **IBM Granite 3.2 8B Instruct** — Language model for generation
- **Granite Embedding 30M English** — Local embedding model for semantic search
- **ChromaDB** — In-process vector store
- **Docling** — PDF-to-Markdown document conversion
- **ITS Hub** — Inference-time scaling (Best-of-N sampling)
- **SDG Hub** — Synthetic data generation pipelines
- **Training Hub / Unsloth** — QLoRA fine-tuning
- **sentence-transformers** — Embedding model runtime

## Project Layout

```
EscalationLab/
├── config.py              # Loads .env and exposes API_KEY / ENDPOINT_BASE
├── .env                   # API credentials (not tracked in git)
├── docs/                  # Source documents (PDFs and converted Markdown)
├── models/                # Pre-downloaded embedding models
├── prebuilt/              # Pre-generated results for offline use
├── utils/                 # Utility notebooks (e.g. GPU diagnostics)
├── extras/                # Supplementary materials (What's Next)
├── 00_Orientation/        # Section 00
├── 01_CustomerScenario/   # Section 01
├── 02_Baseline/           # Section 02
├── 03_Ingestion/          # Section 03
├── 04_RAG/                # Section 04
├── 05_Evaluation/         # Section 05
├── 06_Scaling/            # Section 06
├── 07_InferenceTimeScaling/ # Section 07
├── 08_SyntheticDataGen/   # Section 08
├── 09_ModelAdaptation/    # Section 09
├── 10_ComprehensiveEvaluation/ # Section 10
└── 11_Synthesis/          # Section 11
```

## Environment

- Python 3.12
- PyTorch 2.7 + CUDA 12.8
- NVIDIA L40S (46 GB) or L4 (24 GB)
- OpenShift AI (RHODS)
