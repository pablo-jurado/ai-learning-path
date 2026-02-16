// NOTE: main data file building the learning path. 
const DATA = [
  {
    "name": "AI Learning Path",
    "children": [
      {
        "name": "AI Foundation",
        "icon": "",
        "color": "#2563EB",
        "colorLight": "#EFF6FF",
        "colorBorder": "#BFDBFE",
        "children": [
          {
            "name": "Basics of AI",
            "children": [
              {
                "name": "LLM",
                "description": "Large Language Models are neural networks trained on massive text datasets to understand and generate human language. They form the foundation of modern AI assistants, using transformer architectures with billions of parameters.",
                "tags": ["LLM"],
                "sources": [
                  { label: 'How LLMs work', url: 'https://www.youtube.com/watch?v=LPZh9BOjkQs' }
                ]
              },
              {
                "name": "Context Window",
                "description": "The maximum number of tokens that an LLM can process in a single input or output. Understanding context window limits is crucial for effective prompt engineering and model usage.",
                "tags": ["LLM", "Limit"],
                "sources": [
                  { "label": "Context Window", "url": "https://www.youtube.com/watch?v=-QVoIxEpFkM" },
                ]
              },
              {
                "name": "Agents",
                "description": "AI systems that can autonomously plan, reason, and take actions to accomplish goals. Agents use LLMs as their reasoning engine and interact with the world through tools like APIs, web browsers, and code execution.",
                "tags": ["Agentic AI", "Core Concept"],
                "sources": [
                  { "label": 'What are AI Agents?', "url": 'https://www.youtube.com/watch?v=F8NKVhkZZWI' },
                  { "label": 'LLM vs Agents', "url": 'https://www.youtube.com/watch?v=I9z-nrk9cw0' }
                ]
              }
            ]
          },
          {
            "name": "Model Optimization",
            "children": [
              {
                "name": "Model Optimization",
                "description": "Techniques for improving LLM performance, including few-shot prompting, chain-of-thought prompting, and system prompts. Effective optimization can significantly enhance model outputs without changing the underlying model.",
                "tags": ["Optimization", "Prompting"],
                "sources": [
                  { label: 'Optimizing AI Models', url: 'https://www.youtube.com/watch?v=zYGDpG-pTho' },
                ]
              },
              {
                "name": "Prompt Engineering",
                "description": "The practice of crafting effective instructions to get the best outputs from language models. Techniques include few-shot examples, chain-of-thought prompting, system prompts, and structured output formatting.",
                "tags": ["Skill", "Practical", "Essential"],
                "sources": [
                  { "label": "Anthropic Prompt Engineering Guide", "url": "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview" },
                  { "label": "Claude Prompt Library", "url": "https://platform.claude.com/docs/en/resources/prompt-library/library" },
                  { "label": 'Prompting Techniques Guide', "url": 'https://www.promptingguide.ai/' }
                ]
              },
            ]
          },
          {
            "name": "Advanced Optimization",
            "children": [
              {
                "name": "Structured Outputs (JSON Mode)",
                "description": "Techniques for getting LLMs to return data in specific formats like JSON, XML, or typed schemas. Essential for building reliable AI pipelines where downstream systems need to parse model outputs programmatically.",
                "tags": ["Production", "Reliability", "API"],
                "sources": [
                  { "label": "Structured Outputs", "url": "https://agenta.ai/blog/the-guide-to-structured-outputs-and-function-calling-with-llms" }
                ]
              },
              {
                "name": "Embeddings",
                "description": "Numerical vector representations of text, images, or other data that capture semantic meaning. Similar concepts end up close together in vector space, enabling similarity search, clustering, and recommendation systems.",
                "tags": ["Vectors", "Semantic", "Core"],
                "sources": [
                  { "name": 'What are embeddings?', "url": 'https://www.youtube.com/watch?v=wgfSDrqYMJ4' },
                ]
              },
              {
                "name": "Vector Databases",
                "description": "Specialized databases optimized for storing and querying high-dimensional vectors. Enable fast similarity search at scale, forming the backbone of RAG systems, semantic search, and recommendation engines.",
                "tags": ["Database", "Infrastructure"],
                "sources": [
                  { "name": 'Vector Databases Explained', "url": 'https://www.youtube.com/watch?v=gl1r1XV0SLw' }
                ]
              },
              {
                "name": "RAG",
                "description": "Retrieval-Augmented Generation combines LLMs with external knowledge retrieval. Instead of relying solely on training data, RAG fetches relevant documents at query time and includes them in the prompt for more accurate, up-to-date responses.",
                "tags": ["Architecture", "Retrieval", "Key Pattern"],
                "sources": [
                  { "label": 'What is RAG (Retrieval-Augmented Generation)?', "url": 'https://www.youtube.com/watch?v=T-D1OfcDW1M' },
                ]
              },
              {
                "name": "MCP",
                "description": "The Model Context Protocol is an open standard for connecting AI models to external data sources and tools. Think of it as a universal adapter — one protocol that lets any AI model talk to any data source or service.",
                "tags": ["Protocol", "Standard", "Integration"],
                "sources": [
                  {
                    "label": 'MCP (Model Context Protocol) Explained',
                    "url": 'https://www.youtube.com/watch?v=7j1t3UZA1TY&t',
                  },
                ]
              },
              {
                "name": "Orchestratinng Agents",
                "description": "Techniques for coordinating multiple AI agents to work together on complex tasks. This includes communication protocols, task decomposition, and shared memory systems to enable collaborative problem-solving.",
                "tags": ["Agentic AI", "Collaboration", "Advanced"],
                "sources": [
                  {
                    "label": 'Orchestrating Multiple AI Agents',
                    "url": "https://www.anthropic.com/engineering/building-effective-agents"
                  }
                ]
              }
            ]
          }
        ]
      },
      {
        "name": "AI Skills",
        "icon": "",
        "color": "#16A34A",
        "colorLight": "#ECFDF5",
        "colorBorder": "#A7F3D0",
        "children": [
             {
            "name": "Cloud Platforms",
            "children": [
              {
                "name": "Azure AI Foundry",
                "description": "Microsoft's unified platform for building AI applications on Azure. Provides access to OpenAI models, open-source models, evaluation tools, prompt flow, and enterprise-grade deployment.",
                "tags": ["Cloud", "Microsoft", "Enterprise"],
                "sources": [
                  { "label": "Azure AI Foundry", "url": "https://azure.microsoft.com/en-us/products/ai-services" },
                  { "label": "Azure AI Docs", "url": "https://learn.microsoft.com/en-us/azure/ai-studio/" }
                ]
              },
              {
                "name": "AWS Bedrock",
                "description": "Amazon's fully managed service for building with foundation models. Access Anthropic Claude, Meta Llama, Mistral, and other models through a unified API with fine-tuning, guardrails, and knowledge bases.",
                "tags": ["Cloud", "AWS", "Multi-model"],
                "sources": [
                  { "label": "AWS Bedrock", "url": "https://aws.amazon.com/bedrock/" },
                  { "label": "Bedrock Documentation", "url": "https://docs.aws.amazon.com/bedrock/" }
                ]
              },
            ]
          },
          {
            "name": "Generative AI via API - TODO",
            "children": [
              {}
            ]
          },
          {
            "name": "Creating Agents - TODO",
            "children": [{}]
          },
          {
            "name": "Connecting to MCP servers - TODO",
            "children": [{}]
          },
          {
            "name": "Creating RAG Pipelines - TODO",
            "children": [{}]
          },
                    {
            "name": "Frameworks",
            "children": [
              {
                "name": "Semantic Kernel",
                "description": "Microsoft's open-source SDK for integrating LLMs into applications. Supports C#, Python, and Java with a plugin architecture, planners, and memory connectors for enterprise AI development.",
                "tags": ["Framework", "Microsoft", "C#", "Python"],
                "sources": [
                  { "label": "Semantic Kernel Docs", "url": "https://learn.microsoft.com/en-us/semantic-kernel/" },
                  { "label": "GitHub Repository", "url": "https://github.com/microsoft/semantic-kernel" },
                  { "label": "Semantic Kernel Tutorial", "url": "https://www.youtube.com/watch?v=ewHPdDtmHj4&list=PLyqwquIuSMZpGDiocmT-M67dcDxjWmoYK&index=20" }
                ]
              },
              {
                "name": "AutoGen",
                "description": "Microsoft's framework for building multi-agent conversational systems. Agents can collaborate, debate, and work together on complex tasks with human-in-the-loop capabilities.",
                "tags": ["Framework", "Multi-Agent", "Microsoft"],
                "sources": [
                  { "label": "AutoGen Docs", "url": "https://microsoft.github.io/autogen/" },
                  { "label": "GitHub Repository", "url": "https://github.com/microsoft/autogen" },
                  { "label": "AutoGen Tutorial", "url": "https://www.youtube.com/watch?v=ewHPdDtmHj4&list=PLyqwquIuSMZpGDiocmT-M67dcDxjWmoYK&index=20" }
                ]
              },
              {
                "name": "Microsoft Agent Framework",
                "description": "Microsoft's broader agentic AI framework for building autonomous agents that can plan, reason, and take actions. Integrates with Azure AI services and enterprise tooling. Agent Framework combines AutoGen's simple agent abstractions with Semantic Kernel's enterprise features — session-based state management, type safety, middleware, telemetry — and adds graph-based workflows for explicit multi-agent orchestration.",
                "tags": ["Framework", "Enterprise", "Azure"],
                "sources": [
                  { "label": "Microsoft Agent Framework Docs", "url": "https://learn.microsoft.com/en-us/agent-framework/overview/?pivots=programming-language-csharp" },
                  { "label": "GitHub Repository", "url": "https://github.com/microsoft/agent-framework" },
                  { "label": "Microsoft Agent Framework tutorial", "url": "https://www.youtube.com/watch?v=EAeUiipzCTE"}
                ]
              },
              {
                "name": "AI-SDK (JavaScript/Vercel)",
                "description": "Vercel's TypeScript toolkit for building AI applications with React. Provides hooks and streaming utilities for chat interfaces, generative UI, and tool calling with any LLM provider.",
                "tags": ["Framework", "TypeScript", "React"],
                "sources": [
                  { "label": "AI SDK Docs", "url": "https://ai-sdk.dev/getting-started" },
                  { "label": "GitHub Repository", "url": "https://github.com/vercel/ai" },
                  { "label": "AI SDK Cookbook", "url": "https://ai-sdk.dev/cookbook" },
                  { "label": "AI SDK Guides", "url": "https://ai-sdk.dev/cookbook/guides" },
                  { "label": "AI SDK Tutorial", "url": "https://www.youtube.com/watch?v=mojZpktAiYQ" }
                ]
              },
              {
                "name": "LangChain",
                "description": "The most popular framework for building applications powered by language models. Provides composable components for chains, agents, retrieval, memory, and tool use across Python and JavaScript.",
                "tags": ["Framework", "Python", "JavaScript"],
                "sources": [
                  { "label": "LangChain Python Docs", "url": "https://python.langchain.com/" },
                  { "label": "LangChain JS Docs", "url": "https://js.langchain.com/" },
                  { "label": "LangChain Python Tutorial", "url": "https://www.youtube.com/watch?v=lG7Uxts9SXs" },
                  { "label": "LangChain JS Tutorial", "url": "https://www.youtube.com/watch?v=HSZ_uaif57o" }
                ]
              },
            ]
          },
        ]
      },
      {
        "name": "AI Certifications",
        "icon": "",
        "color": "#a31687",
        "colorLight": "#FDF2FA",
        "colorBorder": "#FBCFE8",
        "children": [
          {
            "name": "Microsoft Azure AI Certifications",
            "children": [{}]
          },
          {
            "name": "AWS AI Certifications",
            "children": [{}]
          }
        ]
      },
    ]
  }
];
