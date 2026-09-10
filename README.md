# Sistema de Apoio ao Diagnóstico de Pneumonia em Radiografias

Interface web (Streamlit) para auxiliar radiologistas na triagem de
radiografias de tórax, classificando a imagem em **Normal** ou
**Pneumonia**. Suporta múltiplos modelos, votação (ensemble) entre
eles, cadastro clínico completo do paciente e histórico persistido em
banco de dados.

## Estrutura do projeto

```
pneumonia_app/
├── app.py                  # Interface Streamlit (abas: Novo Diagnóstico / Histórico)
├── pneumonia_detector.py   # Modelos e votação — AQUI você integra seus modelos reais
├── database.py             # Camada de banco de dados (SQLite)
├── requirements.txt        # Dependências
├── models/                 # Coloque aqui os arquivos dos seus modelos treinados
├── uploads/                # Imagens de radiografias salvas automaticamente (criado em runtime)
├── pneumonia_app.db        # Banco SQLite (criado automaticamente na 1ª execução)
└── README.md
```

## Como rodar

```bash
pip install -r requirements.txt
streamlit run app.py
```

A interface abrirá no navegador (geralmente em `http://localhost:8501`).

Por padrão, o app roda com **3 modelos simulados** (`MockPneumoniaDetector`),
cada um com um perfil de desempenho diferente — isso permite testar
toda a interface (formulário clínico, upload, animação, votação e
histórico) antes mesmo de os modelos reais estarem prontos.

## Funcionalidades

### 1. Cadastro clínico completo
Antes da análise, é possível registrar: nome, prontuário/ID do exame,
data de nascimento (idade calculada automaticamente), sexo, data do
exame, médico solicitante, indicação clínica, sintomas (multi-seleção),
comorbidades/fatores de risco, saturação de O₂, temperatura corporal,
frequência respiratória, observações e radiologista responsável.

### 2. Múltiplos modelos e votação (ensemble)
Na barra lateral, o radiologista escolhe entre:
- **Um modelo específico** — útil quando se quer, por exemplo, o
  modelo com maior *precisão* (menos falsos positivos) em um cenário
  de triagem conservadora.
- **Votação entre modelos (ensemble)** — combina vários modelos
  simultaneamente. Dois métodos de consolidação:
  - *Soft-vote* (padrão): usa a **média das probabilidades** de todos
    os modelos selecionados.
  - *Hard-vote*: usa a **classe mais votada** (maioria simples), com
    a média de probabilidades como critério de desempate.

  O resultado mostra o voto individual de cada modelo e sinaliza
  quando há **divergência** entre eles — informação importante para o
  radiologista decidir se quer revisar a imagem manualmente com mais
  atenção.

No catálogo padrão (`MODEL_CATALOG`, em `pneumonia_detector.py`) há 3
perfis de exemplo:
- `ModeloA_AltaAcuracia` — melhor acurácia geral.
- `ModeloB_AltaPrecisao` — melhor precisão (menos falsos positivos).
- `ModeloC_AltoF1` — melhor F1-score (equilíbrio precisão/recall).

### 3. Banco de dados (histórico de detecções)
Toda análise realizada é automaticamente salva em um banco **SQLite**
local (`pneumonia_app.db`), sem necessidade de instalar servidor de
banco de dados. São armazenados:
- Dados do paciente/exame (tabela `patients`).
- Resultado da detecção: modelo(s) usado(s), rótulo, confiança,
  probabilidades, detalhe de cada voto (se ensemble), concordância
  entre modelos, radiologista responsável (tabela `detections`).
- A imagem da radiografia é salva em `uploads/` e referenciada no
  banco, para consulta posterior.

A aba **Histórico de Detecções** permite buscar por nome/prontuário,
filtrar por resultado (Normal/Pneumonia) e visualizar o detalhe
completo de cada exame, incluindo a imagem e o voto individual de
cada modelo.

## Como integrar seus modelos reais

Todo o trabalho de integração acontece em **`pneumonia_detector.py`**.
A interface (`app.py`) e o banco de dados (`database.py`) não
precisam ser alterados.

1. Treine cada modelo (classificação binária: Normal vs. Pneumonia) e
   salve os pesos (ex.: `.h5` para Keras, `.pt` para PyTorch) na pasta
   `models/`.

2. Para cada modelo, implemente o carregamento e a inferência dentro
   da classe `PneumoniaDetector` (métodos `_load_model`,
   `_preprocess` e `predict`). Já existem exemplos comentados para
   **TensorFlow/Keras** e **PyTorch**.

3. Atualize o dicionário `MODEL_CATALOG` (em `pneumonia_detector.py`)
   com as métricas reais de cada modelo, calculadas em um conjunto de
   teste independente: `acuracia`, `precisao`, `recall`, `f1_score`,
   `especificidade`.

4. Em `build_model_registry()`, troque cada
   `MockPneumoniaDetector(...)` por uma instância real:

   ```python
   registry.register(
       PneumoniaDetector(
           name="ModeloA_AltaAcuracia",
           model_path="models/modelo_a.h5",
           metrics=MODEL_CATALOG["ModeloA_AltaAcuracia"],
       )
   )
   ```

   Repita para cada modelo do catálogo. Você pode ter quantos modelos
   quiser — todos aparecerão automaticamente no seletor da interface
   e poderão participar da votação.

5. Reinicie o Streamlit. A interface passará a usar seus modelos
   reais, com o mesmo fluxo de seleção, votação e histórico.

## Esquema do banco de dados

**`patients`**: nome, prontuário, data de nascimento, idade, sexo,
data do exame, médico solicitante, indicação clínica, sintomas
(JSON), comorbidades (JSON), tabagismo, saturação de O₂, temperatura,
frequência respiratória, observações, data de criação.

**`detections`**: paciente (FK), caminho da imagem, modo de análise
(modelo único ou votação), modelo(s) utilizado(s), rótulo previsto,
confiança, probabilidades (Normal/Pneumonia), detalhe dos votos
individuais (JSON, quando ensemble), concordância entre modelos,
radiologista, observações do laudo, data de criação.

## Observações importantes

- Este sistema é uma ferramenta de **apoio à decisão clínica**. O
  laudo final é sempre de responsabilidade do médico radiologista.
- Os dados são armazenados **localmente**, em arquivo SQLite. Para uso
  em ambiente hospitalar real, avalie migrar para um banco com
  controle de acesso adequado (ex.: PostgreSQL), backup e
  conformidade com a LGPD.
- Recomenda-se validar cada modelo com métricas calculadas em conjunto
  de teste representativo da população-alvo antes de qualquer uso
  assistencial, e considerar os requisitos regulatórios aplicáveis a
  software como dispositivo médico (SaMD), conforme legislação
  vigente (ex.: ANVISA).
