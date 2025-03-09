# 🤖 GenAI & LLM System Design: 500+ Production Case Studies

[![GitHub stars](https://img.shields.io/github/stars/themanojdesai/genai-llm-ml-case-studies?style=social)](https://github.com/themanojdesai/genai-llm-ml-case-studies/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/themanojdesai/genai-llm-ml-case-studies?style=social)](https://github.com/themanojdesai/genai-llm-ml-case-studies/network/members)
[![GitHub issues](https://img.shields.io/github/issues/themanojdesai/genai-llm-ml-case-studies)](https://github.com/themanojdesai/genai-llm-ml-case-studies/issues)
[![GitHub license](https://img.shields.io/github/license/themanojdesai/genai-llm-ml-case-studies)](https://github.com/themanojdesai/genai-llm-ml-case-studies/blob/main/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/themanojdesai/genai-llm-ml-case-studies/blob/main/CONTRIBUTING.md)

> The largest collection of 500+ real-world Generative AI & LLM system design case studies from 130+ companies. Learn how industry leaders design, deploy, and optimize large language models and generative AI systems in production.

*First published: June 14, 2023. Last updated: March 08, 2025*

## 🔍 Quick Navigation
- [What's Inside](#-whats-inside)
- [Featured LLM Case Studies](#-featured-llm-case-studies)
- [Browse by Industry](#-browse-by-industry)
- [Browse by Use Case](#-browse-by-use-case)
- [Browse by Company](#-browse-by-company)
- [GenAI Architectures](#-genai-architectures)
- [Contributing](#-contributing)

## 📚 What's Inside

This repository documents how companies build and deploy production-grade Generative AI and LLM systems, focusing on:

- **Architecture decisions** for RAG, fine-tuning, and multi-modal systems
- **Scaling strategies** for billion-parameter models
- **Optimization techniques** for latency, cost, and performance
- **Evaluation frameworks** for LLM outputs and hallucination mitigation
- **Deployment patterns** across industries

**Perfect for:**
- AI/ML Engineers implementing LLM-powered features
- Engineering teams designing scalable GenAI architectures
- Leaders planning generative AI initiatives
- Technical interviews on LLM system design

## 🏆 Featured LLM Case Studies

### RAG & Knowledge Retrieval
- [Ramp: From RAG to Richness: How Ramp Revamped Industry Classification](case-studies/by-company/ramp/from-rag-to-richness-how-ramp-revamped-industry-classification.md) - Enterprise RAG implementation
- [GitLab: Developing GitLab Duo: How we validate and test AI models at scale](case-studies/by-company/gitlab/developing-gitlab-duo-how-we-validate-and-test-ai-models-at-scale.md) - Testing LLM quality at scale
- [Picnic: Enhancing Search Retrieval with Large Language Models](case-studies/by-company/picnic/enhancing-search-retrieval-with-large-language-models-llms.md) - LLM-powered search

### GenAI Applications
- [Slack: How We Built Slack AI To Be Secure and Private](case-studies/by-company/slack/how-we-built-slack-ai-to-be-secure-and-private.md) - Enterprise LLM security
- [Discord: Developing rapidly with Generative AI](case-studies/by-company/discord/developing-rapidly-with-generative-ai.md) - Generative AI platform
- [GoDaddy: LLM From the Trenches: 10 Lessons Learned Operationalizing Models](case-studies/by-company/godaddy/llm-from-the-trenches-10-lessons-learned-operationalizing-models-at-godaddy.md) - LLM production lessons


## 📊 Browse by Industry

- [Tech](case-studies/by-industry/tech.md) (90 case studies) - **24 LLM case studies**
- [E-commerce and retail](case-studies/by-industry/e-commerce-and-retail.md) (119 case studies) - **21 GenAI case studies**
- [Media and streaming](case-studies/by-industry/media-and-streaming.md) (44 case studies) - **18 LLM case studies**
- [Social platforms](case-studies/by-industry/social-platforms.md) (57 case studies) - **15 GenAI case studies**
- [Fintech and banking](case-studies/by-industry/fintech-and-banking.md) (31 case studies) - **12 LLM implementations**
- [Delivery and mobility](case-studies/by-industry/delivery-and-mobility.md) (108 case studies) - **10 GenAI applications**

## 💡 Browse by LLM/GenAI Use Cases

- [LLM implementation](case-studies/by-use-case/llm.md) (92 case studies)
- [Generative AI applications](case-studies/by-use-case/generative-ai.md) (98 case studies)
- [RAG systems](case-studies/by-use-case/rag.md) (42 case studies)
- [LLM-powered search](case-studies/by-use-case/search.md) (60 case studies)
- [NLP & text processing](case-studies/by-use-case/nlp.md) (48 case studies)
- [LLM evaluation](case-studies/by-use-case/llm-evaluation.md) (36 case studies)
- [Fine-tuning approaches](case-studies/by-use-case/fine-tuning.md) (22 case studies)
- [LLM inference optimization](case-studies/by-use-case/inference-optimization.md) (19 case studies)
- [Multi-modal systems](case-studies/by-use-case/multi-modal.md) (17 case studies)
- [Content personalization](case-studies/by-use-case/content-personalization.md) (15 case studies)

## 🔍 Top Companies with LLM & GenAI Case Studies

- [OpenAI](case-studies/by-company/openai/) (8 case studies)
- [Anthropic](case-studies/by-company/anthropic/) (7 case studies)
- [Microsoft](case-studies/by-company/microsoft/) (16 case studies)
- [Google](case-studies/by-company/google/) (15 case studies)
- [Meta](case-studies/by-company/meta/) (12 case studies)
- [Hugging Face](case-studies/by-company/hugging-face/) (9 case studies)
- [Netflix](case-studies/by-company/netflix/) (14 case studies)
- [LinkedIn](case-studies/by-company/linkedin/) (19 case studies)
- [GitHub](case-studies/by-company/github/) (7 case studies)
- [Spotify](case-studies/by-company/spotify/) (10 case studies)

## 📚 LLM System Design Patterns

- **Pattern 1: Direct LLM Integration**
  - Cost-effective for simple use cases
  - Examples: [GitHub Copilot](case-studies/by-company/github/copilot-system-design.md)

- **Pattern 2: RAG (Retrieval-Augmented Generation)**
  - Improves accuracy with domain-specific knowledge
  - Examples: [Ramp's Industry Classification](case-studies/by-company/ramp/from-rag-to-richness-how-ramp-revamped-industry-classification.md)

- **Pattern 3: Multi-Agent Systems**
  - Complex reasoning through agent collaboration
  - Examples: [AutoGPT-like architectures](case-studies/by-use-case/multi-agent.md)

- **Pattern 4: Human-in-the-Loop**
  - Critical applications requiring human oversight
  - Examples: [Content moderation systems](case-studies/by-use-case/content-moderation.md)

## 📈 LLM Evolution Timeline

- **2023 Q1-Q2**: First wave of RAG implementations
- **2023 Q3-Q4**: Fine-tuning becomes mainstream
- **2024 Q1-Q2**: Agent architectures emerge
- **2024 Q3-Q4**: Multi-modal systems gain traction
- **2025 Q1**: Real-time personalization with LLMs

## 🏗️ GenAI Architectures

### RAG (Retrieval-Augmented Generation)
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Document       │────▶│  Vector         │     │                 │
│  Corpus         │     │  Database       │────▶│                 │
│                 │     │                 │     │   LLM           │
└─────────────────┘     └─────────────────┘     │   Generation    │
                                                │                 │
┌─────────────────┐     ┌─────────────────┐     │                 │
│                 │     │                 │     │                 │
│  User           │────▶│  Query          │────▶│                 │
│  Query          │     │  Processing     │     │                 │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Fine-tuning Approaches
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Base LLM       │────▶│  Fine-tuning    │────▶│  Specialized    │
│  Model          │     │  Pipeline       │     │  Model          │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                              ▲
┌─────────────────┐           │
│                 │           │
│  Company        │───────────┘
│  Data           │
│                 │
└─────────────────┘
```

### Feature Store for LLMs
```
┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │
│  Real-time      │────▶│  Feature        │
│  Data           │     │  Computation    │
│                 │     │                 │     ┌─────────────────┐
└─────────────────┘     └─────────────────┘     │                 │
                              │                 │                 │
┌─────────────────┐           ▼                 │                 │
│                 │     ┌─────────────────┐     │                 │
│  Batch          │────▶│  Feature        │────▶│  LLM            │
│  Data           │     │  Store          │     │  Application    │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## 🤝 Contributing

Contributions are welcome! Help us document the evolving GenAI landscape:

1. Fork the repository
2. Create a new branch
3. Add your LLM/GenAI case study using the established format
4. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📄 License

This repository is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- Thanks to all the companies and engineers who shared their LLM/GenAI implementation experiences
- All original sources are linked in each case study

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=themanojdesai/genai-llm-ml-case-studies&type=Date)](https://star-history.com/#themanojdesai/genai-llm-ml-case-studies&Date)
---

⭐ Found this valuable for your GenAI/LLM work? Star the repository to help others discover it! ⭐