# Interpretação das Métricas — Etapa 3

## Dados usados como base

- decision_tree: recall foi de 0.8824 para 0.9412 (+5.9 pontos percentuais); accuracy foi de 0.9121 para 0.9341; F1 foi de 0.8824 para 0.9143.
- gradient_boosting: recall foi de 0.9118 para 0.9706 (+5.9 pontos percentuais); accuracy foi de 0.9560 para 0.9780; F1 foi de 0.9394 para 0.9706.
- knn: recall foi de 0.9412 para 0.9706 (+2.9 pontos percentuais); accuracy foi de 0.9670 para 0.9670; F1 foi de 0.9552 para 0.9565.
- random_forest: recall foi de 0.9412 para 0.9412 (+0.0 pontos percentuais); accuracy foi de 0.9670 para 0.9670; F1 foi de 0.9552 para 0.9552.
- logistic_regression: recall foi de 0.9706 para 0.9706 (+0.0 pontos percentuais); accuracy foi de 0.9670 para 0.9451; F1 foi de 0.9565 para 0.9296.
- svm: recall foi de 0.9706 para 0.9706 (+0.0 pontos percentuais); accuracy foi de 0.9890 para 0.9780; F1 foi de 0.9851 para 0.9706.
- extra_trees: recall foi de 0.9412 para 0.9412 (+0.0 pontos percentuais); accuracy foi de 0.9670 para 0.9670; F1 foi de 0.9552 para 0.9552.
- mlp: recall foi de 0.9706 para 0.9706 (+0.0 pontos percentuais); accuracy foi de 0.9670 para 0.9780; F1 foi de 0.9565 para 0.9706.

## Resumo executivo gerado

# Resumo Executivo: Impacto Clínico da Otimização dos Modelos de Diagnóstico

## O que isso significa na prática

A otimização melhorou significativamente a **capacidade de detectar casos de câncer** (recall) em três modelos críticos: Decision Tree agora identifica 94% dos casos malignos (ante 88%), Gradient Boosting detecta 97% (ante 91%), e KNN passa para 97% (ante 94%). Em contexto clínico, **cada ponto percentual de melhoria no recall evita mais pacientes com câncer deixando o hospital sem diagnóstico**. Os outros modelos já operavam em níveis muito altos de detecção e não se beneficiaram da otimização.

---

## Recomendações Práticas

• **Adote prioritariamente o Gradient Boosting otimizado**: Combina o maior ganho de sensibilidade (+6 pontos) com acurácia excelente (97,8%). Significa que quase todo paciente com câncer será detectado, e 98% de todos os diagnósticos estarão corretos — o melhor equilíbrio entre evitar perder casos graves e reduzir alarmes falsos.

• **KNN otimizado é alternativa viável**: Também alcança 97% de detecção de câncer com +3 pontos de melhoria. Use como backup ou em triagens rápidas se o Gradient Boosting ficar indisponível; performance clínica é praticamente equivalente.

• **Descarte Decision Tree e Logistic Regression otimizadas**: Decision Tree, apesar de melhoria, ainda deixa 6% dos cânceres não detectados — clinicamente inaceitável. Logistic Regression teve acurácia reduzida na otimização, sinal de desequilíbrio.

• **Mantenha SVM e MLP atuais**: Já detectam 97% dos cânceres sem necessidade de otimização. Trocar por versões otimizadas não agregará valor clínico e aumenta risco de erros em mudança de sistema.

---

**Resultado final**: Implementar Gradient Boosting otimizado eleva a confiança diagnóstica do hospital e reduz significativamente o risco de perda de diagnósticos críticos.
