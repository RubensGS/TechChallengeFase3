# 🏥 Assistente Médico Inteligente — Tech Challenge Fase 3

Sistema de apoio clínico baseado em Inteligência Artificial, combinando
fine-tuning de LLM, RAG com LangChain e fluxos automatizados de decisão
clínica com LangGraph.

> ⚠️ **Aviso importante:** Este sistema é uma ferramenta de **apoio clínico**.
> O médico responsável tem sempre a palavra final em qualquer decisão diagnóstica
> ou terapêutica.

---

## 📋 Índice

- [Visão Geral](#-visão-geral)
- [Arquitetura](#-arquitetura)
- [Tecnologias](#-tecnologias)
- [Datasets](#-datasets)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Instalação](#-instalação)
- [Configuração](#-configuração)
- [Como Executar](#-como-executar)
- [Notebooks](#-notebooks)
- [Resultados](#-resultados)
- [Segurança e Validação](#-segurança-e-validação)
- [Limitações](#-limitações)

---

## 🎯 Visão Geral

O projeto implementa um assistente médico inteligente com três componentes
principais:

| Componente | Tecnologia | Finalidade |
|---|---|---|
| **Fine-tuning** | LLaMA 3.2 1B + QLoRA | Especializar o modelo em linguagem médica |
| **RAG** | LangChain + FAISS + BioBERT | Contextualizar respostas com documentos médicos |
| **Fluxo clínico** | LangGraph | Automatizar triagem, exames, conduta e alertas |

---

## 🏗️ Arquitetura

```
                        ┌─────────────────────────────────┐
                        │     ASSISTENTE MÉDICO IA        │
                        └─────────────────────────────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              │                         │                         │
    ┌─────────▼────────┐    ┌───────────▼──────────┐   ┌─────────▼────────┐
    │   FINE-TUNING    │    │    RAG + LANGCHAIN   │   │    LANGGRAPH     │
    │                  │    │                      │   │                  │
    │  LLaMA 3.2 1B   │    │  FAISS Vector Store  │   │  Grafo de        │
    │  QLoRA (r=16)   │    │  BioBERT Embeddings  │   │  Decisão         │
    │  Google Colab   │    │  Claude API          │   │  Clínica         │
    │                  │    │  LCEL Pipeline       │   │                  │
    └──────────────────┘    └──────────────────────┘   └──────────────────┘
              │                         │                         │
              └─────────────────────────┴─────────────────────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              │                         │                         │
    ┌─────────▼────────┐    ┌───────────▼──────────┐   ┌─────────▼────────┐
    │    DATASETS      │    │    SEGURANÇA         │   │    LOGGING       │
    │                  │    │                      │   │                  │
    │  PubMedQA        │    │  Tópicos bloqueados  │   │  Auditoria       │
    │  MedQuAD         │    │  Validação entrada   │   │  Rastreamento    │
    │  1000 exemplos   │    │  Sem prescrições     │   │  Por sessão      │
    └──────────────────┘    └──────────────────────┘   └──────────────────┘
```

### Fluxo do LangGraph

```
  [ENTRADA: Dados do Paciente]
           │
           ▼
    ┌─────────────┐
    │   TRIAGEM   │ ← Classifica nível de risco (baixo/moderado/alto/crítico)
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │   EXAMES    │ ← Sugere exames complementares por prioridade
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │   CONDUTA   │ ← Gera conduta clínica baseada em evidências
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │   ALERTA    │ ← Emite alertas para risco alto/crítico
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │  RELATÓRIO  │ ← Consolida tudo em relatório clínico estruturado
    └──────┬──────┘
           │
          FIM
```

---

## 🛠️ Tecnologias

| Biblioteca | Versão | Uso |
|---|---|---|
| `langchain-core` | 1.6.3 | Base do pipeline LangChain |
| `langchain-anthropic` | 1.7.2 | Integração com Claude API |
| `langchain-community` | 0.4.2 | FAISS e HuggingFace |
| `langchain-text-splitters` | latest | Divisão de documentos |
| `langgraph` | latest | Fluxo de decisão clínica |
| `faiss-cpu` | latest | Vector store para RAG |
| `sentence-transformers` | latest | Embeddings BioBERT |
| `unsloth` | latest | Fine-tuning eficiente (Colab) |
| `anthropic` | latest | Claude API |
| `python-dotenv` | latest | Variáveis de ambiente |
| `datasets` | latest | Carregamento PubMedQA e MedQuAD |

**Modelo LLM:** `claude-sonnet-4-6` (Claude API — Anthropic)

**Modelo Fine-tuning:** `unsloth/Llama-3.2-1B-Instruct`

**Modelo Embeddings:** `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb`

---

## 📦 Datasets

### PubMedQA
- **Fonte:** https://pubmedqa.github.io
- **Conteúdo:** Perguntas e respostas clínicas baseadas em artigos do PubMed
- **Exemplos usados:** 500
- **Formato:** Pergunta + resposta longa + decisão final (yes/no/maybe)

### MedQuAD
- **Fonte:** https://github.com/abachaa/MedQuAD
- **Conteúdo:** 47.457 pares de Q&A de saúde do NIH, CDC e FDA
- **Exemplos usados:** 500
- **Formato:** Pergunta + resposta estruturada

**Total de exemplos processados:** 1.000
- Treino: 900 exemplos (90%)
- Validação: 100 exemplos (10%)

---

## 📁 Estrutura do Projeto

```
fase3_assistente_medico/
│
├── notebooks/
│   ├── 01_preparacao_dados.ipynb    # Download e pré-processamento
│   ├── 02_fine_tuning.ipynb         # Fine-tuning QLoRA (rodar no Colab)
│   ├── 03_rag_langchain.ipynb       # Pipeline RAG com LangChain
│   └── 04_langgraph.ipynb           # Fluxo automatizado com LangGraph
│
├── src/
│   ├── data/
│   │   └── preprocessor.py          # Funções de pré-processamento
│   ├── rag/
│   │   └── vectorstore.py           # Gerenciamento do FAISS
│   ├── assistant/
│   │   ├── chain.py                 # Pipeline LCEL
│   │   └── prompts.py               # Templates de prompt
│   └── security/
│       └── logger.py                # Sistema de logging
│
├── data/
│   ├── raw/                         # Dados brutos (não commitar)
│   └── processed/
│       ├── dados_medicos_ft.jsonl   # Dados para fine-tuning
│       ├── dados_medicos_rag.csv    # Dados para RAG
│       ├── vectorstore/             # Índice FAISS salvo
│       └── modelo_medico_lora/      # Adaptadores LoRA salvos
│
├── logs/                            # Logs de auditoria (não commitar)
│
├── .env                             # Variáveis de ambiente (não commitar)
├── .env.example                     # Template de variáveis de ambiente
├── .gitignore                       # Arquivos ignorados pelo Git
├── requirements.txt                 # Dependências do projeto
└── README.md                        # Este arquivo
```

---

## ⚙️ Instalação

### Pré-requisitos

- Python 3.10 ou superior
- GPU AMD/NVIDIA (para fine-tuning — recomendado Google Colab)
- Conta na Anthropic com chave de API ativa

### 1. Clone o repositório

```bash
git clone <url-do-repositorio>
cd fase3_assistente_medico
```

### 2. Crie o ambiente virtual

```bash
python -m venv .venv

# Linux/Mac
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Instale as dependências

```bash
pip install langchain-anthropic \
            langchain-community \
            langchain-text-splitters \
            langgraph \
            faiss-cpu \
            sentence-transformers \
            anthropic \
            python-dotenv \
            datasets \
            pandas numpy
```

---

## 🔑 Configuração

### 1. Crie o arquivo `.env`

```bash
cp .env.example .env
```

### 2. Preencha sua chave da API

```
ANTHROPIC_API_KEY=sk-ant-...sua-chave-aqui...
```

> ⚠️ **Nunca commite o arquivo `.env` no Git.**
> Ele já está listado no `.gitignore`.

### 3. Verifique o carregamento

```python
from dotenv import load_dotenv
import os

load_dotenv()
print(os.getenv("ANTHROPIC_API_KEY")[:10] + "...")  # mostra só o início
```

---

## ▶️ Como Executar

Os notebooks devem ser executados **na seguinte ordem:**

### Notebook 1 — Preparação dos dados
```bash
jupyter notebook notebooks/01_preparacao_dados.ipynb
```
Baixa PubMedQA e MedQuAD, processa e salva os dados em `data/processed/`.

---

### Notebook 2 — Fine-tuning (Google Colab)
> ⚠️ Este notebook deve ser executado no **Google Colab** com GPU T4.
> GPUs AMD não são suportadas pelas bibliotecas de fine-tuning (unsloth/CUDA).

1. Acesse [colab.research.google.com](https://colab.research.google.com)
2. Menu: `Runtime → Change runtime type → T4 GPU`
3. Cole e execute as células do notebook
4. Ao final, baixe o arquivo `modelo_medico_lora.zip`
5. Descompacte em `data/processed/modelo_medico_lora/`

**Resultado esperado:** Loss final ≈ 1.34 após 3 épocas

---

### Notebook 3 — RAG com LangChain
```bash
jupyter notebook notebooks/03_rag_langchain.ipynb
```
Cria o vector store FAISS, monta o pipeline RAG e testa o assistente.

---

### Notebook 4 — LangGraph
```bash
jupyter notebook notebooks/04_langgraph.ipynb
```
Implementa e executa o fluxo automatizado de decisão clínica.

---

## 📊 Resultados

### Fine-tuning

| Métrica | Valor |
|---|---|
| Modelo base | LLaMA 3.2 1B Instruct |
| Técnica | QLoRA (r=16, alpha=16) |
| Épocas | 3 |
| Batch size efetivo | 8 (2 × acumulação 4) |
| Learning rate | 2e-4 |
| Loss final | **1.3427** |
| Parâmetros treináveis | ~1.2% do total |
| Tempo de treino (T4) | ~20 minutos |

### RAG + LangChain

| Componente | Configuração |
|---|---|
| Embeddings | BioBERT especializado em biomedicina |
| Vector store | FAISS com busca por similaridade cosseno |
| Documentos indexados | ~1.200 chunks (de 1.000 documentos) |
| Documentos recuperados por query | 4 |
| LLM | claude-sonnet-4-6 (temperatura 0.2) |
| Tempo médio de resposta | ~18 segundos |

### LangGraph

| Nó | Função |
|---|---|
| Triagem | Classifica nível de risco do paciente |
| Exames | Sugere exames por prioridade |
| Conduta | Gera conduta clínica baseada em evidências |
| Alerta | Emite alertas para risco alto/crítico |
| Relatório | Consolida relatório clínico estruturado |

---

## 🔒 Segurança e Validação

### Tópicos bloqueados
O assistente recusa automaticamente consultas que envolvam:

```python
TOPICOS_BLOQUEADOS = [
    "prescrever", "receitar", "posologia exata",
    "dose de", "mg por kg", "automedicar",
    "sem consultar", "diagnóstico definitivo"
]
```

### Logging e auditoria
Todas as consultas são registradas em `logs/assistente_YYYYMMDD.log`:

```
2024-01-15 14:32:01 | INFO | [dr_silva] Nova consulta: Quais são os critérios...
2024-01-15 14:32:29 | INFO | [dr_silva] Resposta em 28637ms | Fontes: ['MedQuAD', 'PubMedQA']
2024-01-15 14:33:10 | WARNING | [dr_silva] Bloqueado: 'receitar' detectado
```

### Explainability
Cada resposta cita as fontes utilizadas na recuperação:
```
📚 Fontes utilizadas:
   [PubMedQA] Does diabetes mellitus influence the efficacy...
   [MedQuAD]  What are the treatments for...
```

### Limites de atuação
- ✅ Responde perguntas clínicas baseadas em evidências
- ✅ Sugere exames e encaminhamentos
- ✅ Fornece informações sobre mecanismos de ação
- ❌ Não prescreve medicamentos
- ❌ Não emite diagnósticos definitivos
- ❌ Não substitui avaliação médica presencial

---

## ⚠️ Limitações

**Dataset:** 1.000 exemplos é um volume pequeno para produção. A recuperação
pelo FAISS pode retornar documentos não diretamente relacionados à pergunta.
Em produção, recomenda-se uma base com dezenas de milhares de documentos
clínicos especializados.

**Idioma:** O fine-tuning foi realizado com dados em inglês (PubMedQA e MedQuAD).
O comportamento em português depende do system prompt — em produção, o ideal
seria fine-tuning com prontuários clínicos em português anonimizados.

**Hardware:** O fine-tuning requer GPU NVIDIA com suporte a CUDA. GPUs AMD
(como a Radeon RX 7600 usada no desenvolvimento) não são suportadas pelas
bibliotecas de quantização. A solução adotada foi usar Google Colab com T4.

**Autenticação:** O sistema não implementa autenticação de usuários. Em produção
hospitalar, seria necessária integração com sistemas de identidade (SSO, OAuth).

**Validação clínica:** O sistema não foi validado clinicamente. Antes de qualquer
uso em ambiente hospitalar real, seria necessária validação por especialistas
médicos e aprovação pelos comitês de ética competentes.

---

## 📄 Licença

Este projeto foi desenvolvido para fins acadêmicos no contexto do
Tech Challenge — PosTech FIAP. Não deve ser utilizado em ambiente
clínico real sem validação médica adequada.
