# Reconhecimento Facial — OpenCV

Estudo de reconhecimento facial com OpenCV, de outubro de 2023.

| Arquivo | O que faz |
|---|---|
| `main.py`, `Exemplo1.py`, `exemplo2.py`, `exemplo3.py` | Os exercicios, do mais simples ao mais completo |
| `Cascades/` | Classificadores Haar para detectar o rosto |
| `classificadorFischer.yml`, `classificadorLBPH.yml` | Modelos ja treinados (Fisherfaces e LBPH) |
| `fotos/`, `fotos1/`, `pessoas/` | Base de imagens usada no treino |

```bash
pip install opencv-contrib-python
python main.py
```

Os dois classificadores existem de proposito: **Fisherfaces** exige imagens do
mesmo tamanho e e sensivel a iluminacao, enquanto **LBPH** tolera variacao e
aceita acrescentar pessoas sem retreinar tudo. Comparar os dois era o ponto do
exercicio.

## Atencao: imagens de pessoas

As pastas de fotos contem rostos. Confira de quem sao antes de reutilizar este
material — biometria facial e dado pessoal sensivel pela LGPD.
