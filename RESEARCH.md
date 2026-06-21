# Multimodal Video Analysis — Research Notes

Research conducted 2026-06-21. Goal: identify the most relevant open-source models and tools for building a multimodal video analyzer (film scene, music video, or YouTube video).

## Modalities in a typical video

1. **Visual frames** (24-60 fps image sequence)
2. **Audio waveform** (music, speech, ambient)
3. **Speech transcript** (ASR output)
4. **On-screen text** (captions, signs, overlays)
5. Derived: scene composition, facial emotion, body pose, music structure

A "multimodal analysis" combines 2+ of these on a single timeline.

## Top open-source models by task (mid-2026)

### Vision-language (image + text + video)
Sorted by HuggingFace downloads (signal of community adoption):

| Model | Downloads | Size | Notes |
|---|---|---|---|
| google/gemma-4-26B-A4B-it | 12.6M | 26B | Google's latest, multi-modal |
| Qwen/Qwen3.5-4B | 9.6M | 4B | Qwen text-only, small |
| **Qwen/Qwen2.5-VL-7B-Instruct** | **8.1M** | **7B** | **Best choice: handles video natively, 1+ hour videos, event localization** |
| Qwen/Qwen3-VL-8B-Instruct | 7.3M | 8B | Newer Qwen3 vision |
| Qwen/Qwen2.5-VL-3B-Instruct | 5.3M | 3B | Lighter, lower VRAM |
| VideoLLaMA3 (Boqiang Zhang et al., 2025) | 459 citations | 7B | Frontier video understanding |
| VideoLLaMA 2 (Cheng et al., 2024) | 770 citations | 7B | Audio + video |
| InternVideo2 | 2024 | various | Strong on action recognition |

**Pick: Qwen2.5-VL-7B-Instruct** — best video capabilities, Apache-2.0, strong community, can run on 16GB VRAM with quantization.

### Audio analysis

| Model | Task | Notes |
|---|---|---|
| **laion/clap-htsat-fused** | Zero-shot audio classification | 16.6M downloads, dominant. Tags audio with natural language |
| openai/whisper-large-v3 | ASR | Industry standard, MIT license |
| openai/whisper-large-v3-turbo | Fast ASR | 6x faster, slight accuracy drop |
| pyannote/speaker-diarization-3.1 | Who speaks when | 8.5M downloads |
| m-a-p/MERT-v1-330M | Music representation | Pre-trained music encoder |

### Music generation / analysis
| Model | Task |
|---|---|
| facebook/musicgen-medium | Text-to-music generation |
| stabilityai/stable-audio-3-medium | Text-to-audio |
| google/magenta-realtime-2 | Real-time music gen |

### Video classification / shot detection
| Model | Task | Stars |
|---|---|---|
| **soCzech/TransNetV2** | **Shot boundary detection** | **970 ⭐ on GitHub** — THE standard |
| facebook/vjepa2-vitg-fpc64-256 | Self-supervised video features | 290K downloads on HF |
| MCG-NJU/videomae-base | Video masked autoencoder | 235K downloads |
| microsoft/xclip-base-patch32 | Video-text alignment | 84K |

**Pick: TransNetV2** for shot boundaries. State-of-the-art F1 on multiple benchmarks (77.9 on ClipShots, 96.2 on BBC Planet Earth).

### Audio-visual joint
- **CLAP (laion/clap-htsat-fused)** is the standard for joint audio-text embeddings. Use it to compare audio events to natural-language queries.
- For music-video cross-modal alignment: extract CLIP visual embeddings + CLAP audio embeddings, learn a projection, or use cosine similarity directly.

## Existing reference projects (GitHub, sorted by stars)

| Project | Stars | What it does |
|---|---|---|
| soCzech/TransNetV2 | 970 | Shot boundary detection |
| jordanrendric/claude-video-vision | 814 | Claude API + frame extraction |
| 1038lab/ComfyUI-QwenVL | 787 | Qwen-VL as ComfyUI node |
| declare-lab/multimodal-sentiment-analysis | 126 | Sentiment in user-generated videos |
| win4r/VideoFinder-Llama3.2-vision-Ollama | 174 | FastAPI + Ollama + OpenCV, upload + ask |
| Aseiel/VideoHighlighter | 43 | OpenCV scene + Whisper + YOLO highlights |
| tanbryan/ai-mv-generator | 104 | Multi-agent music video generator |
| Synesthesia-AI-Video-Director | 50 | Audio analysis + LLM director |

## Research papers found (key ones)

- **Qwen2-VL** (Wang et al., 2024, 4314 citations) — foundation of Qwen-VL series
- **VideoLLaMA 2** (Cheng et al., 2024, 770 citations) — audio + video
- **VideoLLaMA 3** (Zhang et al., 2025, 459 citations) — frontier multimodal
- **InternVideo2** (2024) — multimodal video foundation
- **CLAP** (Contrastive Language-Audio Pretraining) — standard audio-language alignment
- **MAGNET** (2025) — multi-agent framework for audio-visual reasoning over video haystacks

## Recommended tech stack for multimodal_analysis project

**Core pipeline:**
1. **ffmpeg** — extract frames + audio track
2. **TransNetV2** — shot boundary detection
3. **Qwen2.5-VL-7B-Instruct** — frame captioning, scene understanding (via Ollama)
4. **Whisper-large-v3** — ASR (via faster-whisper for speed)
5. **CLAP** — audio event tagging, music-audio similarity
6. **librosa** — tempo, key, energy (music features)
7. **MediaPipe** — face/pose landmarks (optional)
8. **PyTorch + transformers** — model hosting

**Dashboard (mirroring taylor-swift-lyrics-nlp pattern):**
- Plotly interactive timeline with synchronized traces for each modality
- HTML escape, dynamic prose, all the lessons from the 11 audit rounds
- requirements.txt, SHA-pinning if data is downloaded, source hash

## Open architectural questions

1. **One video or many?** One video = deeper per-second analysis. Many = comparative patterns.
2. **Realtime or offline?** Offline is much easier; 7B+ vision models fine.
3. **Web app or static dashboard?** Static HTML dashboard transfers the taylor-swift pattern.
4. **What makes this novel?** The cross-modal synchronization story (e.g., "do cuts happen on beats?") is more interesting than yet another captioning demo.
