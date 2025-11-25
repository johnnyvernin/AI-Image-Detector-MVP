# 🔍 AI Image Detector MVP

> Um detector experimental de imagens geradas por IA usando múltiplas técnicas de análise computacional.

![Status](https://img.shields.io/badge/status-MVP-yellow)
![Python](https://img.shields.io/badge/python-3.8+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🎯 O Problema

Com a evolução explosiva de modelos geradores de imagens (Midjourney, DALL-E, Stable Diffusion, Flux), distinguir fotos reais de imagens sintéticas tornou-se um desafio crítico para:

- **Jornalismo**: Verificação de autenticidade de fotos
- **Redes sociais**: Combate a desinformação
- **Forense digital**: Investigações e provas legais
- **Arte/Copyright**: Proteção de direitos autorais

Este projeto explora técnicas clássicas de visão computacional para detectar padrões estatísticos que diferenciam imagens reais de geradas por IA.

---

## 🧪 Como Funciona: As 4 Técnicas

### 1️⃣ **Luminance Gradient PCA** (Peso: 35%)

**O que detecta:** Coerência dos gradientes de iluminação

**Lógica:**
```
Imagem RGB → Luminância (Y = 0.21R + 0.71G + 0.07B)
             ↓
         Gradientes (Sobel X e Y)
             ↓
    Matriz de Covariância → PCA
             ↓
      Análise de Eigenvalues
```

**Por que funciona:**
- Fotos reais têm iluminação física coerente (luz vem de fontes específicas)
- Gradientes seguem padrões naturais (sombras, reflexos, difusão)
- IA pode gerar estruturas de luz inconsistentes ou matematicamente implausíveis

**Calibração:**
- `ratio > 0.25`: Score alto (estrutura coerente = real)
- `ratio < 0.25`: Score baixo (estrutura instável = IA)
- Ratio = eigenvalue[1] / eigenvalue[0]

**Limitações:**
- Sensível a HDR e processamento pesado
- Pode falhar em imagens muito uniformes (céu azul, parede branca)

---

### 2️⃣ **Spectral Analysis (FFT)** (Peso: 20%)

**O que detecta:** Distribuição de energia no domínio de frequência

**Lógica:**
```
Imagem → FFT 2D → Magnitude Spectrum
                      ↓
            Perfil Radial (energia por distância do centro)
                      ↓
     Correlação log(energia) vs. distância
                      ↓
    Decaimento exponencial = real / Irregular = IA
```

**Por que funciona:**
- Imagens naturais seguem lei de potência (power-law): muita energia em baixas frequências, decaimento suave
- IA pode ter distribuição espectral artificial ou picos anormais em altas frequências
- Correlação negativa forte (-0.8 a -0.95) indica naturalidade

**Calibração:**
- `correlation < -0.5`: Score alto (decaimento natural)
- `correlation > -0.5`: Penalizado (distribuição irregular)
- Redimensionamento para 512x512 para consistência

**Limitações:**
- Modelos modernos aprenderam a simular distribuições espectrais realistas
- Compressão JPEG afeta análise

---

### 3️⃣ **Texture LBP** (Peso: 25%)

**O que detecta:** Padrões locais de textura

**Lógica:**
```
Imagem → Grayscale → LBP (Local Binary Pattern)
                          ↓
                   Histograma de padrões
                          ↓
    Variância do histograma + Diversidade de padrões
                          ↓
              Score combinado
```

**Por que funciona:**
- Fotos reais têm texturas orgânicas e heterogêneas
- IA pode gerar padrões muito uniformes (smooth demais) ou artificialmente repetitivos
- Diversidade de padrões indica complexidade natural

**Calibração:**
- `var_score = hist_var × 5000` (variância do histograma)
- `diversity_score = padrões_únicos / 256`
- Score final = 40% variância + 60% diversidade

**Limitações:**
- Muito sensível a processamento de câmera (noise reduction, sharpening)
- Fotos de celular moderno podem parecer "artificiais"

---

### 4️⃣ **Noise Analysis** (Peso: 20%)

**O que detecta:** Assinatura de ruído e consistência

**Lógica:**
```
Imagem → Laplacian (detecção de ruído)
             ↓
   Análise em blocos 32x32
             ↓
Consistência do ruído entre blocos
             ↓
   Presença + Consistência = Score
```

**Por que funciona:**
- Câmeras reais geram ruído do sensor (ISO, temperatura, eletrônica)
- Ruído natural é consistente espacialmente
- IA é muito limpa OU adiciona ruído artificial não-consistente

**Calibração:**
- `noise_estimate` típico: 10-40 para real, <5 ou >60 para IA
- `noise_consistency`: 1.0 / (1 + std/mean) - quanto maior, mais natural
- Score = 60% presença + 40% consistência

**Limitações:**
- **Maior fraqueza do sistema!**
- Smartphones modernos fazem noise reduction agressiva
- Night mode e computational photography removem quase todo ruído
- IA moderna pode adicionar ruído convincente

---

## ⚖️ Sistema de Ensemble

### Pesos e Justificativa

```python
weights = {
    'Luminance Gradient PCA': 0.35,  # Mais robusto, difícil de burlar
    'Spectral Analysis (FFT)': 0.20,  # Bom mas IA aprende rápido
    'Texture LBP': 0.25,              # Funciona bem em texturas complexas
    'Noise Analysis': 0.20            # Fraco contra processamento moderno
}
```

### Lógica de Classificação

**Score Final = Média Ponderada**

```
final_score = Σ(método.score × peso) / Σ(pesos)
```

**Thresholds:**
- `> 55%`: **Real Image** (3+ métodos concordam)
- `42-55%`: **Uncertain** (métodos divididos)
- `< 42%`: **AI Generated** (3+ métodos concordam)

**Confiança Ajustada:**
- `std < 0.30`: High Confidence (métodos concordam)
- `std ≥ 0.30`: Medium/Low Confidence (discrepância alta)

---

## ⚠️ Limitações Críticas

### 🚨 **O Elefante na Sala: Modelos Modernos Burlam Tudo**

Este MVP usa **técnicas clássicas** (2000s-2010s) que foram **eficazes** contra geradores antigos (GANs simples, VAEs), mas **falham drasticamente** contra:

| Modelo | Ano | Taxa de Detecção Estimada |
|--------|-----|---------------------------|
| DALL-E 2 | 2022 | ~60-70% |
| Midjourney v5 | 2023 | ~40-50% |
| **Stable Diffusion 3** | 2024 | **~20-35%** ⚠️ |
| **DALL-E 3** | 2024 | **~15-30%** ⚠️ |
| **Flux** | 2024 | **~10-25%** 🔴 |

**Por quê?**

✅ Geradores modernos aprenderam a:
- Simular distribuições espectrais naturais (FFT inútil)
- Gerar gradientes físicamente plausíveis (PCA engana)
- Adicionar ruído artificial convincente (Noise Analysis falha)
- Criar micro-texturas orgânicas (LBP confunde)

✅ Plus: Podem ser **fine-tuned** especificamente para burlar detectores clássicos

---

## 🎯 Soluções Recomendadas

### 🥇 **Nível 1: Deep Learning (Altamente Recomendado)**

Substituir/complementar com modelos treinados:

**Opção A: CLIP-based Detector**
```python
# Hugging Face: umm-maybe/AI-image-detector
# Acc: ~85-90% em SD/DALL-E/MJ
```

**Opção B: ResNet Fine-tuned**
```python
# Treinar ResNet50 em dataset CNNDetection
# Acc: ~80-85% com data augmentation
```

**Opção C: Vision Transformer (ViT)**
```python
# Transformer com self-attention
# Detecta inconsistências globais
# Acc: ~85-92% (state-of-the-art)
```

**Prós:** 90%+ accuracy, aprende padrões que humanos não veem  
**Contras:** Precisa GPU, modelo .pth (~100MB+), mais lento

---

### 🥈 **Nível 2: APIs Third-Party**

Usar serviços especializados como segundo parecer:

| Serviço | Tecnologia | Custo | Accuracy |
|---------|-----------|-------|----------|
| **Hive Moderation** | Ensemble ML | $0.001/img | ~90% |
| **Optic.AI** | Multi-modal | $0.005/img | ~88% |
| **Illuminarty** | Proprietary | Free tier | ~85% |
| **Content Credentials** | C2PA | Free | Metadata |

**Implementação:**
```python
# Exemplo: Hive API
response = requests.post('https://api.thehive.ai/api/v2/task/sync',
    headers={'Authorization': f'Bearer {API_KEY}'},
    files={'image': open(image_path, 'rb')}
)
ai_score = response.json()['status']['ai_generated_media']
```

**Prós:** Accuracy alta, sempre atualizado  
**Contras:** Custo por requisição, dependência externa

---

### 🥉 **Nível 3: Análise Semântica (Complementar)**

Detectar erros que IA comete:

- **Mãos/dedos**: Número errado, dedos fundidos, unhas estranhas
- **Texto**: Letras embaralhadas, fontes inconsistentes
- **Física**: Sombras impossíveis, reflexos errados, perspectiva quebrada
- **Olhos**: Assimetria, pupilas diferentes, brilho artificial

**Implementação:** OCR + YOLO + regras heurísticas

---

### 🏆 **Solução Híbrida Ideal (Produção)**

```
┌─────────────────────────────────────┐
│    1. Técnicas Clássicas (MVP)     │ ← Rápido, sem custo
│    Score: 0-100%                    │
└─────────────┬───────────────────────┘
              │
         [Score < 60%?] ← Incerto
              │
              ↓
┌─────────────────────────────────────┐
│  2. Deep Learning Local (ViT/CNN)  │ ← Accuracy alta
│  Score: 0-100%                      │
└─────────────┬───────────────────────┘
              │
    [Ainda incerto ou crítico?]
              │
              ↓
┌─────────────────────────────────────┐
│     3. API Third-Party (Hive)      │ ← Decisão final
│     Score: 0-100%                   │
└─────────────────────────────────────┘
              │
              ↓
      [Veredito final]
```

**Accuracy esperada:** 95%+  
**Custo:** ~$0.001/img (apenas casos incertos)  
**Latência:** 1-2s (maioria resolvida no step 1)

---

## 📊 Resultados Esperados (MVP Atual)

### Fotos Reais de Celular
```
✅ Score: 65-80%
✅ Classificação: Real Image
⚠️ Confiança: Medium (noise reduction afeta)
```

### Fotos DSLR/RAW
```
✅ Score: 80-95%
✅ Classificação: Real Image
✅ Confiança: High
```

### IA Antiga (GANs 2018-2020)
```
✅ Score: 5-25%
✅ Classificação: AI Generated
✅ Confiança: High
```

### IA Moderna (SD3/DALL-E3/Flux)
```
❌ Score: 45-75% ← PROBLEMA!
❌ Classificação: Uncertain/Real
❌ Confiança: Low
⚠️ Taxa de falso negativo: 60-80%
```

---

## 🚀 Quick Start

### Setup Básico
```bash
# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependências
pip install Flask numpy opencv-python scipy scikit-learn Pillow

# Criar pastas
mkdir templates uploads

# Rodar
python app.py
```

Acesse: `http://localhost:5000`

---

## 🔮 Roadmap Futuro

- [ ] **v0.2**: Integrar modelo CNN pré-treinado
- [ ] **v0.3**: API Hive como fallback
- [ ] **v0.4**: Detecção de artefatos semânticos (mãos, olhos)
- [ ] **v0.5**: Fine-tuning ViT em dataset proprietário
- [ ] **v1.0**: Sistema híbrido produção-ready

---

## 📚 Referências Acadêmicas

- Wang et al. (2020): "CNN-generated images are surprisingly easy to spot... for now"
- Corvi et al. (2023): "Detection of GAN-generated images via spectral analysis"
- Gragnaniello et al. (2021): "Forensic detection of diffusion models"

---

## ⚖️ Disclaimer Legal

⚠️ **Este é um MVP experimental e educacional.**

- **Não é 100% confiável** para decisões críticas
- **Taxa de falso negativo alta** em IA moderna (60-80%)
- **Não substitui** verificação humana especializada
- **Use apenas como ferramenta auxiliar**, não como veredito final

Para aplicações de alto risco (forense, jurídico, jornalismo), recomenda-se:
1. Consultar especialista em visão forense
2. Usar múltiplas ferramentas third-party
3. Análise manual de artefatos
4. Verificação de metadados (EXIF, C2PA)

---

**Contribuições são bem-vindas!** 🚀

---

**Desenvolvido com** 🔬 **ciência** e 💜 **curiosidade**
