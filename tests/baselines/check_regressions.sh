#!/usr/bin/env bash
# Gate de regressão contra o baseline de falhas conhecidas.
#
# ⚠️ A versão ingênua deste gate era VÁCUA no cenário que mais importa:
#
#     pytest ... | grep '^FAILED' | sort > agora.txt
#
# Se a suíte abortasse na COLETA, o pipe devolvia zero linhas e isso era lido
# como "zero regressões" — o gate ficava mais verde justamente quando as coisas
# iam pior. Duas correções: contar `ERROR` além de `FAILED`, e checar o código
# de saída do pytest (o do pipe é o do `grep`, não o do pytest).
set -uo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAIZ="$(cd "$BASE_DIR/../.." && pwd)"
PY="${PYTHON:-python3}"
BASELINE="$BASE_DIR/known_failures.txt"
BRUTO="$(mktemp)"; AGORA="$(mktemp)"
trap 'rm -f "$BRUTO" "$AGORA"' EXIT

cd "$RAIZ"
"$PY" -m pytest tests/ -q --tb=no \
  --ignore=tests/vector_stores --ignore=tests/llms --ignore=tests/embeddings \
  > "$BRUTO" 2>&1
RC=$?

# 0 = tudo passou · 1 = houve falhas (esperado, temos baseline).
# Qualquer outro código é interrupção, erro de uso, coleta abortada — casos em
# que a lista de falhas NÃO é confiável e não se pode concluir nada dela.
if [ "$RC" -ne 0 ] && [ "$RC" -ne 1 ]; then
  echo "GATE INCONCLUSIVO: pytest saiu com $RC (não é 0 nem 1)."
  echo "A lista de falhas não é confiável neste estado — investigue antes de seguir."
  tail -15 "$BRUTO"
  exit 2
fi

grep -E '^(FAILED|ERROR) ' "$BRUTO" | sed -E 's/^(FAILED|ERROR) //; s/ - .*//' \
  | sort -u > "$AGORA"

BASE_LIMPO="$(mktemp)"; trap 'rm -f "$BRUTO" "$AGORA" "$BASE_LIMPO"' EXIT
grep -v '^#' "$BASELINE" | sort -u > "$BASE_LIMPO"

echo "  ⚠ ignorando vector_stores/, llms/ e embeddings/ (não coletam neste venv)"
NOVAS="$(comm -13 "$BASE_LIMPO" "$AGORA")"
SUMIRAM="$(comm -23 "$BASE_LIMPO" "$AGORA")"

echo "  baseline : $(wc -l < "$BASE_LIMPO") · agora: $(wc -l < "$AGORA") · pytest rc=$RC"

if [ -n "$SUMIRAM" ]; then
  echo "  consertadas (bom — tire-as do baseline):"
  echo "$SUMIRAM" | sed 's/^/    /'
fi

if [ -n "$NOVAS" ]; then
  echo "  REGRESSÕES NOVAS:"
  echo "$NOVAS" | sed 's/^/    /'
  exit 1
fi

echo "  nenhuma regressão nova."
