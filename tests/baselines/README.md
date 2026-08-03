# Baseline de falhas conhecidas

`known_failures.txt` lista os testes que **já falhavam antes** do trabalho de
partição em três componentes (medido em 02/08/2026: 89 falhas, 1414 passes,
excluindo `vector_stores/`, `llms/` e `embeddings/`, que nem coletam sem os
extras de provider).

## Por que isto existe

O plano dos três componentes usa "suíte verde" como gate de cada passo. Ela
**não está verde**, e uma suíte com 89 vermelhos crônicos não serve de gate: uma
regressão nova ficaria indistinguível no meio deles.

O baseline troca o critério de *"zero falhas"* para *"zero falhas NOVAS"*, que é
verificável hoje. É contorno, não solução — a dívida segue aberta e está
registrada em `reviews/result-passo0-r1.md` do repositório do stack.

## Uso

```bash
pytest tests/ -q --tb=no --ignore=tests/vector_stores \
  --ignore=tests/llms --ignore=tests/embeddings 2>&1 \
  | grep '^FAILED' | sed 's/FAILED //; s/ - .*//' | sort > /tmp/agora.txt

# o que importa é o DELTA; falha nova aparece aqui
comm -13 tests/baselines/known_failures.txt /tmp/agora.txt
```

Saída vazia = nenhuma regressão nova. Qualquer linha = investigar antes de seguir.

⚠️ **Este arquivo só encolhe.** Acrescentar um teste a ele para "ficar verde"
inverteria o propósito: o baseline existe para tornar a dívida visível, não para
absorvê-la em silêncio. Se um teste novo falha, ou se conserta ou vira decisão
explícita — nunca uma linha a mais aqui.
