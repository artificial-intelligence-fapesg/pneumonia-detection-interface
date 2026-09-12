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

### Erro "Unrecognized keyword arguments" / `quantization_config`

Se você treinou o modelo com uma versão do TensorFlow/Keras mais nova
do que a instalada onde a interface roda, pode aparecer um erro como:

```
TypeError: Unrecognized keyword arguments passed to Dense: {'quantization_config': None}
```

Isso acontece porque o `.h5` guarda a configuração de cada camada, e
versões mais novas do Keras às vezes adicionam parâmetros que versões
mais antigas ainda não conhecem. **Você não precisa fazer nada** — o
`PneumoniaDetector` detecta esse erro automaticamente e recarrega o
modelo em um "modo de compatibilidade": remove da configuração apenas
os parâmetros que a versão local não reconhece e mantém os pesos
originais intactos. Uma mensagem no console avisa quando isso
acontece (`"...carregado com sucesso em modo de compatibilidade."`).

Se mesmo assim o carregamento falhar, a forma mais segura é alinhar
as versões: reexporte o modelo com a mesma versão de TensorFlow
listada em `requirements.txt` (2.16+) e salve novamente.

## Como integrar seus modelos reais (.h5, TensorFlow 2.16+)

`PneumoniaDetector` já sabe carregar e executar modelos Keras salvos em
`.h5`/`.hdf5` (ou no formato nativo `.keras`), treinados com
**TensorFlow 2.16 ou superior** — exatamente os modelos gerados pelo
notebook de treinamento deste projeto.

### Caminho mais simples (sem editar nenhum código)

1. Treine o modelo e salve-o em `.h5` com o **mesmo nome** de uma das
   chaves do `MODEL_CATALOG` (em `pneumonia_detector.py`):
   `ModeloA_AltaAcuracia.h5`, `ModeloB_AltaPrecisao.h5` ou
   `ModeloC_AltoF1.h5`.
2. Copie o arquivo para a pasta `models/`.
3. Reinicie o Streamlit.

Pronto — `build_model_registry()` detecta o arquivo automaticamente e
passa a usar o modelo real no lugar do simulado (mock). O tamanho de
entrada da imagem e o formato de saída do modelo (1 neurônio sigmoid
ou 2 neurônios softmax) são detectados automaticamente a partir do
próprio arquivo `.h5`. Você pode ter modelos reais para alguns nomes e
deixar outros como mock — cada um é resolvido de forma independente,
e todos continuam disponíveis na votação (ensemble).

### Caminho manual (nome de arquivo ou caminho customizado)

Se preferir não usar a convenção de nomes acima, registre manualmente
em `build_model_registry()`:

```python
registry.register(
    PneumoniaDetector(
        name="ModeloA_AltaAcuracia",
        model_path="models/meu_arquivo_customizado.h5",
        metrics=MODEL_CATALOG["ModeloA_AltaAcuracia"],
    )
)
```

### Observações importantes sobre pré-processamento

- Por padrão (`normalize=False`), o `PneumoniaDetector` **não** divide
  os pixels por 255 antes de enviar ao modelo — porque as arquiteturas
  do notebook de treinamento já incluem uma camada `Rescaling(1./255)`
  como primeira camada. Se o seu modelo espera entrada já normalizada
  em `[0, 1]` e **não** tem essa camada internamente, instancie com
  `PneumoniaDetector(..., normalize=True)`.
- Se o seu modelo usa camadas, losses ou métricas customizadas que o
  Keras não reconhece automaticamente ao carregar, passe-as via
  `custom_objects` — ajuste a chamada de
  `tf.keras.models.load_model(...)` dentro de
  `PneumoniaDetector._load_model()`.
- Se o carregamento de um `.h5` falhar (arquivo corrompido, formato
  incompatível, etc.), o app registra um aviso no console e usa
  automaticamente o modelo simulado (mock) no lugar, para que a
  interface continue funcional.

### Métricas do modelo

Atualize `MODEL_CATALOG` (em `pneumonia_detector.py`) com as métricas
reais de cada modelo, calculadas em um conjunto de teste independente:
`acuracia`, `precisao`, `recall`, `f1_score`, `especificidade`.

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
