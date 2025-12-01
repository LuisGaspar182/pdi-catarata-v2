# Projeto N2 – Detecção Automática de Características na Cirurgia de Catarata

Este repositório contém o desenvolvimento do Projeto N2 da disciplina **Processamento Digital de Imagens (PDI)**.  
O objetivo é construir um sistema completo de visão computacional capaz de detectar e rastrear estruturas relevantes durante cirurgias de catarata, extraindo métricas quantitativas que auxiliam no controle de qualidade.

---

## 🎯 Objetivos

O sistema implementado identifica e acompanha automaticamente:

- Contorno da **Esclera Ocular**
- Borda interna da **Íris**
- **Região de Incisão**
- Métricas: circularidade e centramento

O projeto segue as especificações fornecidas pelo professor (UNIFEI – ECOI24, 2025).

---

## 📂 Estrutura do Repositório

project/
│
├── src/
│ ├── preprocessing.py
│ ├── find_circles_v2.py
│ ├── build_videos.py
│
├── data/
│ ├── videos/ # Vídeos originais
│ ├── preprocessed/ # Frames estabilizados e filtrados
│ ├── preprocessed_dec/ # Frames após encontrar os contornos
│ └── videos_processados/ # Saídas geradas pelo sistema
│
├── docs/
│ └── artigo_ieee/ # Artigo final no formato IEEE
│
├── .gitignore
├── README.md
└── requirements.txt


Pastas vazias incluem `.gitkeep` para manter a estrutura versionada.

---

## ⚙️ Instalação e Ambiente

Recomenda-se Ubuntu 22.04 LTS.

### 1. Clonar o repositório

```bash
git clone https://github.com/SEU_USUARIO/NOME_DO_REPO.git
cd NOME_DO_REPO

2. Criar ambiente virtual

python3 -m venv venv
source venv/bin/activate

3. Instalar dependências

pip install -r requirements.txt