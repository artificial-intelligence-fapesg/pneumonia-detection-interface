"""
pneumonia_detector.py
======================

Módulo responsável por carregar e executar os modelos de classificação
de radiografias de tórax (Normal vs. Pneumonia), incluindo suporte a:

1. Uso de um modelo específico (dentre vários cadastrados).
2. Votação (ensemble) entre múltiplos modelos, combinando modelos com
   pontos fortes distintos (ex.: um com melhor acurácia, outro com
   melhor precisão, outro com melhor F1-score).

COMO INTEGRAR SEUS MODELOS REAIS (.h5, TensorFlow 2.16+)
-----------------------------------------------------------
A classe `PneumoniaDetector` já sabe carregar e executar modelos Keras
salvos no formato HDF5 (`.h5` / `.hdf5`), treinados em **TensorFlow
2.16 ou superior** (Keras 3) — inclusive os modelos gerados pelo
notebook de treinamento (`modelo_alta_acuracia.h5`,
`modelo_alta_precisao.h5`, `modelo_alto_f1.h5`, etc). Também aceita o
formato nativo `.keras`.

O jeito mais simples de integrar (sem editar nenhum código):

1. Salve o modelo treinado em `.h5` com o **mesmo nome** de uma das
   chaves de `MODEL_CATALOG` (ex.: `ModeloA_AltaAcuracia.h5`).
2. Copie o arquivo para a pasta `models/`.
3. Reinicie o app. `build_model_registry()` detecta o arquivo
   automaticamente e passa a usar o modelo real no lugar do simulado
   (`MockPneumoniaDetector`) — a interface não precisa de nenhuma
   alteração.

Se preferir registrar manualmente (outro nome de arquivo, caminho
customizado, etc.), instancie `PneumoniaDetector` diretamente dentro
de `build_model_registry()`:

    registry.register(
        PneumoniaDetector(
            name="ModeloA_AltaAcuracia",
            model_path="models/meu_arquivo.h5",
            metrics=MODEL_CATALOG["ModeloA_AltaAcuracia"],
        )
    )

Enquanto nenhum arquivo `.h5` correspondente for encontrado em
`models/`, o registro usa `MockPneumoniaDetector`, que simula o
modelo (útil para testar a interface antes de o treinamento estar
pronto).
"""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


CLASSES = ["Normal", "Pneumonia"]


# ---------------------------------------------------------------------------
# Estruturas de resultado
# ---------------------------------------------------------------------------
@dataclass
class DiagnosisResult:
    """Resultado padronizado de um único modelo."""

    model_name: str
    label: str                     # "Normal" ou "Pneumonia"
    confidence: float              # confiança da classe prevista (0-1)
    probabilities: dict            # {"Normal": 0.12, "Pneumonia": 0.88}


@dataclass
class EnsembleResult:
    """Resultado consolidado da votação entre múltiplos modelos."""

    label: str
    confidence: float
    probabilities: dict                    # probabilidades médias (soft-vote)
    individual_results: list[DiagnosisResult] = field(default_factory=list)
    votes: dict = field(default_factory=dict)      # {"Normal": 1, "Pneumonia": 2}
    unanimous: bool = False


PROCESSING_STEPS = [
    "Validando formato e integridade da imagem...",
    "Normalizando e redimensionando a radiografia...",
    "Extraindo características da imagem (rede convolucional)...",
    "Calculando probabilidades por classe...",
    "Consolidando laudo assistido por IA...",
]


# ---------------------------------------------------------------------------
# Compatibilidade entre versões do Keras
# ---------------------------------------------------------------------------
# Modelos .h5 salvos com uma versão do Keras mais nova que a instalada podem
# trazer, na configuração de cada camada, parâmetros que a versão local
# ainda não conhece (ex.: `quantization_config` adicionado ao `Dense` em
# versões recentes do Keras 3). Nesse caso, `tf.keras.models.load_model`
# falha com um `TypeError: Unrecognized keyword arguments passed to <Camada>`.
#
# As funções abaixo implementam um carregamento alternativo: leem a
# configuração (JSON) armazenada dentro do .h5, removem automaticamente
# qualquer parâmetro que o construtor da camada instalada localmente não
# aceite, reconstroem a arquitetura já "limpa" e, por fim, carregam os
# pesos do próprio arquivo .h5 nessa arquitetura. Isso resolve o problema
# tanto para `quantization_config` quanto para qualquer outro parâmetro
# novo que versões futuras do Keras venham a adicionar.
def _strip_unknown_layer_kwargs(config, layers_module) -> None:
    """Percorre recursivamente um dict de configuração de modelo Keras e
    remove, de cada camada, os parâmetros que o construtor da camada
    instalada localmente não reconhece."""
    import inspect

    if isinstance(config, dict):
        if "class_name" in config and isinstance(config.get("config"), dict):
            layer_cls = getattr(layers_module, config["class_name"], None)
            if layer_cls is not None:
                try:
                    accepted = set(inspect.signature(layer_cls.__init__).parameters.keys())
                except (TypeError, ValueError):
                    accepted = None
                if accepted:
                    # kwargs padrão aceitos por toda camada Keras (Layer base),
                    # mesmo que não apareçam explicitamente na assinatura do __init__
                    accepted |= {"name", "trainable", "dtype"}
                    layer_config = config["config"]
                    for key in list(layer_config.keys()):
                        if key not in accepted:
                            layer_config.pop(key, None)
        for value in config.values():
            _strip_unknown_layer_kwargs(value, layers_module)
    elif isinstance(config, list):
        for item in config:
            _strip_unknown_layer_kwargs(item, layers_module)


def _load_h5_with_sanitized_config(model_path: Path):
    """
    Carrega um .h5 removendo, da configuração das camadas, parâmetros que a
    versão local do Keras não reconhece (ex.: `quantization_config`).

    Em vez de reconstruir o modelo com `model_from_json` — que usa o
    deserializador "novo" do Keras 3 e não entende o formato legado salvo
    dentro do .h5 —, gravamos a configuração já higienizada de volta em uma
    cópia temporária do próprio arquivo .h5 e deixamos o `load_model`
    normal fazer o carregamento. Isso reaproveita o parser legado correto
    (compatível com o que gerou o arquivo), alterando apenas os parâmetros
    de camada desconhecidos — pesos e demais metadados permanecem intactos.
    """
    import json
    import os
    import shutil
    import tempfile
    import h5py
    import tensorflow as tf

    with h5py.File(str(model_path), "r") as f:
        raw_config = f.attrs.get("model_config")
        if raw_config is None:
            raise RuntimeError(
                "O arquivo .h5 não contém a chave 'model_config' — ele não "
                "parece ter sido salvo com `model.save(...)` do Keras."
            )
        if isinstance(raw_config, bytes):
            raw_config = raw_config.decode("utf-8")

    config_dict = json.loads(raw_config)
    _strip_unknown_layer_kwargs(config_dict, tf.keras.layers)
    sanitized_json = json.dumps(config_dict)

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".h5")
    os.close(tmp_fd)
    try:
        shutil.copyfile(str(model_path), tmp_path)
        with h5py.File(tmp_path, "r+") as f:
            del f.attrs["model_config"]
            f.attrs["model_config"] = sanitized_json

        return tf.keras.models.load_model(tmp_path, compile=False)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Classe base — todo modelo (real ou simulado) implementa esta interface
# ---------------------------------------------------------------------------
class BasePneumoniaModel:
    name: str = "modelo-base"
    metrics: dict = {}

    def predict(self, image: Image.Image) -> DiagnosisResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Implementação para modelos reais (.h5 / .keras — TensorFlow 2.16+ / Keras 3)
# ---------------------------------------------------------------------------
class PneumoniaDetector(BasePneumoniaModel):
    """
    Wrapper de um modelo real de classificação de radiografias, treinado
    em TensorFlow/Keras e salvo em `.h5`, `.hdf5` ou `.keras`.

    Compatível com modelos treinados em TensorFlow 2.16+ (Keras 3): o
    carregamento usa `tf.keras.models.load_model`, que reconhece
    automaticamente tanto o formato HDF5 legado (`.h5`) quanto o
    formato nativo do Keras 3 (`.keras`).

    O tamanho de entrada e o formato de saída (1 neurônio com sigmoid,
    ou 2 neurônios com softmax) são detectados automaticamente a partir
    do próprio modelo — não é necessário informá-los manualmente na
    maioria dos casos.
    """

    SUPPORTED_EXTENSIONS = {".h5", ".hdf5", ".keras"}

    def __init__(
        self,
        name: str,
        model_path: str,
        metrics: dict,
        input_size: Optional[tuple] = None,
        normalize: bool = False,
    ):
        """
        Args:
            name: nome de exibição do modelo (deve casar com uma chave
                de MODEL_CATALOG).
            model_path: caminho para o arquivo .h5/.hdf5/.keras.
            metrics: dicionário de métricas (acurácia, precisão, recall, f1).
            input_size: (altura, largura) esperada pelo modelo. Se None
                (padrão), é detectado automaticamente a partir de
                `model.input_shape`.
            normalize: se True, divide os pixels por 255 antes de
                enviar ao modelo. Deixe **False** (padrão) se o seu
                modelo já contém uma camada `Rescaling(1./255)` como
                primeira camada — é o caso dos modelos gerados pelo
                notebook de treinamento deste projeto. Se o seu modelo
                foi treinado esperando entrada já normalizada em
                [0, 1] mas SEM camada de Rescaling interna, mude para
                True.
        """
        self.name = name
        self.model_path = Path(model_path)
        self.metrics = metrics
        self.input_size = input_size
        self.normalize = normalize
        self.model = None
        self.output_mode: Optional[str] = None  # "sigmoid" (1 neurônio) ou "softmax" (2 neurônios)
        self._load_model()

    # 1) Carregamento do modelo -------------------------------------------------
    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Arquivo do modelo '{self.name}' não encontrado em: {self.model_path}\n"
                "Verifique se o caminho está correto e se o arquivo foi copiado "
                "para a pasta 'models/'."
            )

        if self.model_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Extensão '{self.model_path.suffix}' não suportada para o modelo "
                f"'{self.name}'. Formatos aceitos: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}."
            )

        try:
            import tensorflow as tf
        except ImportError as e:
            raise ImportError(
                "TensorFlow não está instalado. Instale com "
                "`pip install tensorflow>=2.16` para carregar modelos .h5/.keras "
                "(veja requirements.txt)."
            ) from e

        try:
            # compile=False: para inferência não é necessário recompilar o
            # modelo (evita erros de desserialização do otimizador/loss
            # usados apenas durante o treino).
            self.model = tf.keras.models.load_model(str(self.model_path), compile=False)
        except TypeError as e:
            # Erro típico de incompatibilidade de versão: o .h5 foi salvo
            # com uma versão do Keras mais nova que introduziu um novo
            # parâmetro em alguma camada (ex.: `quantization_config` no
            # Dense), que a versão local ainda não reconhece. Tentamos
            # reconstruir o modelo removendo automaticamente esses
            # parâmetros desconhecidos, mantendo os pesos originais.
            if "Unrecognized keyword arguments" in str(e):
                print(
                    f"[modelos] '{self.name}': o .h5 foi salvo com uma versão do "
                    "Keras diferente da instalada (parâmetro de camada não "
                    "reconhecido). Tentando carregar em modo de compatibilidade..."
                )
                try:
                    self.model = _load_h5_with_sanitized_config(self.model_path)
                    print(f"[modelos] '{self.name}': carregado com sucesso em modo de compatibilidade.")
                except Exception as fallback_error:
                    raise RuntimeError(
                        f"Falha ao carregar o modelo '{self.name}' a partir de "
                        f"'{self.model_path}', mesmo em modo de compatibilidade.\n"
                        f"Erro original: {e}\n"
                        f"Erro no modo de compatibilidade: {fallback_error}\n\n"
                        "Sugestão: reexporte o modelo com a mesma versão de "
                        "TensorFlow/Keras instalada neste ambiente (veja "
                        "requirements.txt) e salve novamente."
                    ) from fallback_error
            else:
                raise RuntimeError(
                    f"Falha ao carregar o modelo '{self.name}' a partir de "
                    f"'{self.model_path}'.\nErro original: {e}\n\n"
                    "Dicas de compatibilidade (TensorFlow 2.16+ / Keras 3):\n"
                    "  - Se o .h5 foi salvo com uma versão muito antiga do "
                    "TensorFlow/Keras, reexporte o modelo com TF 2.16+ e salve "
                    "novamente (.h5 ou .keras).\n"
                    "  - Se o modelo usa camadas, losses ou métricas customizadas, "
                    "passe-as via `custom_objects` em "
                    "`tf.keras.models.load_model(path, compile=False, "
                    "custom_objects={...})` (ajuste `_load_model`)."
                ) from e
        except Exception as e:
            raise RuntimeError(
                f"Falha ao carregar o modelo '{self.name}' a partir de "
                f"'{self.model_path}'.\nErro original: {e}\n\n"
                "Dicas de compatibilidade (TensorFlow 2.16+ / Keras 3):\n"
                "  - Se o .h5 foi salvo com uma versão muito antiga do "
                "TensorFlow/Keras, reexporte o modelo com TF 2.16+ e salve "
                "novamente (.h5 ou .keras).\n"
                "  - Se o modelo usa camadas, losses ou métricas customizadas, "
                "passe-as via `custom_objects` em "
                "`tf.keras.models.load_model(path, compile=False, "
                "custom_objects={...})` (ajuste `_load_model`)."
            ) from e

        self._infer_io_shapes()

    def _infer_io_shapes(self):
        """Detecta automaticamente o tamanho de entrada e o formato de saída do modelo."""
        input_shape = getattr(self.model, "input_shape", None)
        if isinstance(input_shape, list):  # modelos com múltiplas entradas
            input_shape = input_shape[0]

        if self.input_size is None and input_shape and len(input_shape) == 4:
            height, width = input_shape[1], input_shape[2]
            if height and width:
                self.input_size = (int(height), int(width))

        if self.input_size is None:
            # fallback: tamanho usado nas arquiteturas de referência deste projeto
            self.input_size = (252, 252)

        output_shape = getattr(self.model, "output_shape", None)
        if isinstance(output_shape, list):  # modelos com múltiplas saídas
            output_shape = output_shape[0]
        n_outputs = output_shape[-1] if output_shape else 1
        self.output_mode = "sigmoid" if n_outputs == 1 else "softmax"

    # 2) Pré-processamento -------------------------------------------------------
    def _preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB").resize(self.input_size)
        array = np.asarray(image, dtype=np.float32)
        if self.normalize:
            array = array / 255.0
        return np.expand_dims(array, axis=0)

    # 3) Inferência ---------------------------------------------------------------
    def predict(self, image: Image.Image) -> DiagnosisResult:
        if self.model is None:
            raise RuntimeError(f"Modelo '{self.name}' não carregado.")

        processed = self._preprocess(image)
        raw_output = self.model.predict(processed, verbose=0)
        raw_output = np.asarray(raw_output).reshape(-1)

        if self.output_mode == "softmax" and raw_output.size >= 2:
            # Saída com 2 neurônios: [prob_normal, prob_pneumonia].
            # Se as classes do seu treino estiverem na ordem inversa,
            # troque os índices abaixo.
            prob_normal, prob_pneumonia = float(raw_output[0]), float(raw_output[1])
        else:
            # Saída com 1 neurônio (sigmoid) = probabilidade de Pneumonia.
            prob_pneumonia = float(raw_output[0])
            prob_normal = 1 - prob_pneumonia

        probabilities = {"Normal": prob_normal, "Pneumonia": prob_pneumonia}
        label = "Pneumonia" if prob_pneumonia >= 0.5 else "Normal"
        confidence = probabilities[label]

        return DiagnosisResult(
            model_name=self.name,
            label=label,
            confidence=confidence,
            probabilities=probabilities,
        )


# ---------------------------------------------------------------------------
# Implementação simulada (para testes de UI enquanto os modelos reais
# não estão prontos). Cada modelo simulado tem um "perfil" que o torna
# levemente mais conservador/agressivo, para que a votação produza
# divergências realistas entre os modelos.
# ---------------------------------------------------------------------------
class MockPneumoniaDetector(BasePneumoniaModel):
    def __init__(self, name: str, metrics: dict, bias: float = 0.0, noise: float = 0.12):
        self.name = name
        self.metrics = metrics
        self.bias = bias      # desloca a probabilidade média do modelo
        self.noise = noise    # dispersão em torno da probabilidade "real" simulada

    def predict(self, image: Image.Image) -> DiagnosisResult:
        time.sleep(0.25)  # simula custo computacional da inferência

        # Gera uma probabilidade "verdadeira" determinística a partir do
        # conteúdo da imagem (mesma imagem -> mesmo resultado-base),
        # permitindo comparar modelos de forma consistente durante testes.
        image_bytes = image.tobytes()
        digest = hashlib.sha256(image_bytes).hexdigest()
        base_seed = int(digest[:8], 16)
        base_rng = random.Random(base_seed)
        base_prob = base_rng.uniform(0.05, 0.95)

        # Cada modelo aplica seu próprio viés/ruído sobre a base, simulando
        # diferenças reais de desempenho entre arquiteturas/modelos.
        model_rng = random.Random(base_seed + hash(self.name) % (2**16))
        noisy_prob = base_prob + self.bias + model_rng.uniform(-self.noise, self.noise)
        prob_pneumonia = min(max(noisy_prob, 0.01), 0.99)
        prob_normal = 1 - prob_pneumonia

        probabilities = {"Normal": round(prob_normal, 4), "Pneumonia": round(prob_pneumonia, 4)}
        label = "Pneumonia" if prob_pneumonia >= 0.5 else "Normal"
        confidence = probabilities[label]

        return DiagnosisResult(
            model_name=self.name,
            label=label,
            confidence=confidence,
            probabilities=probabilities,
        )


# ---------------------------------------------------------------------------
# Registro de modelos (Model Registry)
# ---------------------------------------------------------------------------
class ModelRegistry:
    """
    Mantém a lista de modelos disponíveis na interface e implementa a
    lógica de seleção de modelo único ou votação (ensemble).
    """

    def __init__(self):
        self._models: dict[str, BasePneumoniaModel] = {}

    def register(self, model: BasePneumoniaModel):
        self._models[model.name] = model

    def list_models(self) -> list[str]:
        return list(self._models.keys())

    def get_metrics(self, name: str) -> dict:
        return self._models[name].metrics

    def get_all_metrics(self) -> dict[str, dict]:
        return {name: m.metrics for name, m in self._models.items()}

    def predict_single(self, model_name: str, image: Image.Image) -> DiagnosisResult:
        if model_name not in self._models:
            raise ValueError(f"Modelo '{model_name}' não está registrado.")
        return self._models[model_name].predict(image)

    def predict_ensemble(
        self,
        image: Image.Image,
        model_names: Optional[list[str]] = None,
        method: str = "soft",
    ) -> EnsembleResult:
        """
        Executa a votação entre múltiplos modelos.

        method="soft"  -> decide pela média das probabilidades (mais
                           sensível a quão confiante cada modelo está).
        method="hard"  -> decide pela classe mais votada (maioria simples).
        Em ambos os casos, o detalhe do voto de cada modelo é retornado
        para transparência do laudo.
        """
        names = model_names or self.list_models()
        individual_results = [self._models[name].predict(image) for name in names]

        votes = {"Normal": 0, "Pneumonia": 0}
        for r in individual_results:
            votes[r.label] += 1
        unanimous = votes["Normal"] == 0 or votes["Pneumonia"] == 0

        avg_prob_pneumonia = sum(r.probabilities["Pneumonia"] for r in individual_results) / len(
            individual_results
        )
        avg_prob_normal = 1 - avg_prob_pneumonia
        probabilities = {"Normal": round(avg_prob_normal, 4), "Pneumonia": round(avg_prob_pneumonia, 4)}

        if method == "hard":
            label = "Pneumonia" if votes["Pneumonia"] > votes["Normal"] else "Normal"
            # em empate, usa o soft-vote (probabilidade média) como desempate
            if votes["Pneumonia"] == votes["Normal"]:
                label = "Pneumonia" if avg_prob_pneumonia >= 0.5 else "Normal"
        else:  # soft (padrão)
            label = "Pneumonia" if avg_prob_pneumonia >= 0.5 else "Normal"

        confidence = probabilities[label]

        return EnsembleResult(
            label=label,
            confidence=confidence,
            probabilities=probabilities,
            individual_results=individual_results,
            votes=votes,
            unanimous=unanimous,
        )


# ---------------------------------------------------------------------------
# Catálogo de modelos disponíveis
# ---------------------------------------------------------------------------
# ATUALIZE este catálogo com as métricas reais de cada modelo, calculadas
# em um conjunto de teste independente. Os valores devem estar entre 0 e 1.
MODEL_CATALOG = {
    "ModeloA": {
        "descricao": "Otimizado para melhor acurácia geral. Bom equilíbrio entre falso-positivos e falso-negativos.",
        "acuracia": 0.94,
        "precisao": 0.91,
        "recall": 0.93,
        "f1_score": 0.92,
        "especificidade": 0.94,
    },
    "ModeloB": {
        "descricao": "Otimizado para minimizar falsos positivos (maior precisão). Mais conservador ao indicar pneumonia.",
        "acuracia": 0.91,
        "precisao": 0.96,
        "recall": 0.85,
        "f1_score": 0.90,
        "especificidade": 0.97,
    },
    "ModeloC": {
        "descricao": "Otimizado para o melhor equilíbrio entre precisão e recall (F1-score). Bom para triagem geral.",
        "acuracia": 0.92,
        "precisao": 0.91,
        "recall": 0.94,
        "f1_score": 0.925,
        "especificidade": 0.90,
    },
}

ENSEMBLE_LABEL = "Votação (Ensemble de Modelos)"

MODELS_DIR = Path(__file__).parent / "models"


def _find_model_file(name: str) -> Optional[Path]:
    """
    Procura, na pasta `models/`, um arquivo de modelo real cujo nome
    (sem extensão) case com `name`. Aceita .h5, .hdf5 e .keras, nessa
    ordem de prioridade.
    """
    if not MODELS_DIR.exists():
        return None
    for ext in (".h5", ".hdf5", ".keras"):
        candidate = MODELS_DIR / f"{name}{ext}"
        if candidate.exists():
            return candidate
    return None


def build_model_registry() -> ModelRegistry:
    """
    Monta o registro de modelos usado pela interface.

    Para cada modelo do `MODEL_CATALOG`, procura automaticamente um
    arquivo `models/<nome_do_modelo>.h5` (ou `.hdf5`/`.keras`). Se
    encontrar, carrega o modelo real (`PneumoniaDetector`); caso
    contrário, usa um modelo simulado (`MockPneumoniaDetector`) para
    que a interface continue funcionável.

    Ou seja: basta treinar o modelo, salvá-lo como
    `models/ModeloA_AltaAcuracia.h5` (mesmo nome da chave do
    catálogo) e reiniciar o app — nenhuma edição de código é
    necessária.

    Se preferir um caminho de arquivo diferente do padrão, registre
    manualmente, por exemplo:

        registry.register(
            PneumoniaDetector(
                name="ModeloA_AltaAcuracia",
                model_path="models/meu_arquivo_customizado.h5",
                metrics=MODEL_CATALOG["ModeloA_AltaAcuracia"],
            )
        )
    """

    registry = ModelRegistry()

    registry.register(
                    PneumoniaDetector(
                        name="VitaRX700m",
                        model_path="models/modelo252x252_7x7_700Mparams_30epoc92acc63f1sc87prec97rec.h5",
                        metrics={
        "descricao": "Otimizado para o melhor equilíbrio entre precisão e recall (F1-score). Bom para triagem geral.",
        "acuracia": 0.92,
        "precisao": 0.87,
        "recall": 0.97,
        "f1_score": 0.63,
        "especificidade": 0.88,
    },
                    )
                )

    registry.register(
                        PneumoniaDetector(
                            name="VitaRX19M",
                            model_path="models/VitaRX19B.h5",
                            metrics={
            "descricao": "Otimizado para o melhor equilíbrio entre precisão e recall (F1-score). Bom para triagem geral.",
            "acuracia": 0.91,
            "precisao": 0.89,
            "recall": 0.96,
            "f1_score": 0.78,
            "especificidade": 0.90,
        },
                        )
                    )

    # Perfis usados apenas pelo modelo simulado (mock), para diferenciar
    # o comportamento de cada "modelo" enquanto o real não está disponível.
    mock_profiles = {
        "ModeloA": {"bias": 0.0, "noise": 0.10},
        "ModeloB": {"bias": -0.08, "noise": 0.10},
        "ModeloC": {"bias": 0.05, "noise": 0.12},
    }

    for name, metrics in MODEL_CATALOG.items():
        model_file = _find_model_file(name)

        if model_file is not None:
            try:
                registry.register(
                    PneumoniaDetector(name=name, model_path=model_file, metrics=metrics)
                )
                print(f"[modelos] '{name}': modelo real carregado de '{model_file.name}'.")
                continue
            except Exception as e:
                print(
                    f"[modelos] Aviso: falha ao carregar o modelo real '{name}' "
                    f"a partir de '{model_file}': {e}\n"
                    f"          Usando modelo simulado (mock) no lugar."
                )
                profile = mock_profiles.get(name, {"bias": 0.0, "noise": 0.12})
                registry.register(MockPneumoniaDetector(name=name, metrics=metrics, **profile))
                continue

        profile = mock_profiles.get(name, {"bias": 0.0, "noise": 0.12})
        registry.register(MockPneumoniaDetector(name=name, metrics=metrics, **profile))
        print(f"[modelos] '{name}': nenhum arquivo em 'models/{name}.h5' — usando modelo simulado (mock).")

    return registry
