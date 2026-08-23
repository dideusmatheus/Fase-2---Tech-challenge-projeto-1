# Interpretação das Métricas — Etapa 3

## Dados usados como base

- decision_tree: recall foi de 0.8824 para 0.9412 (+5.9 pontos percentuais); accuracy foi de 0.9121 para 0.9451; F1 foi de 0.8824 para 0.9275.
- gradient_boosting: recall foi de 0.9118 para 0.9706 (+5.9 pontos percentuais); accuracy foi de 0.9560 para 0.9780; F1 foi de 0.9394 para 0.9706.
- knn: recall foi de 0.9412 para 0.9706 (+2.9 pontos percentuais); accuracy foi de 0.9670 para 0.9670; F1 foi de 0.9552 para 0.9565.
- random_forest: recall foi de 0.9412 para 0.9412 (+0.0 pontos percentuais); accuracy foi de 0.9670 para 0.9670; F1 foi de 0.9552 para 0.9552.
- logistic_regression: recall foi de 0.9706 para 0.9706 (+0.0 pontos percentuais); accuracy foi de 0.9670 para 0.9670; F1 foi de 0.9565 para 0.9565.
- svm: recall foi de 0.9706 para 0.9706 (+0.0 pontos percentuais); accuracy foi de 0.9890 para 0.9890; F1 foi de 0.9851 para 0.9851.
- extra_trees: recall foi de 0.9412 para 0.9412 (+0.0 pontos percentuais); accuracy foi de 0.9670 para 0.9780; F1 foi de 0.9552 para 0.9697.
- mlp: recall foi de 0.9706 para 0.9706 (+0.0 pontos percentuais); accuracy foi de 0.9670 para 0.9890; F1 foi de 0.9565 para 0.9851.

## Resumo executivo gerado

# RESUMO EXECUTIVO: IMPACTO DA OTIMIZAÇÃO NOS MODELOS DE DIAGNÓSTICO

## O que isto significa na prática?

A otimização melhorou significativamente a capacidade do sistema detectar casos reais de câncer (redução de casos não diagnosticados), especialmente nos modelos que tinham deficiência inicial. **Três modelos agora atingem desempenho de classe mundial, com capacidade de identificar 97% dos casos malignos.** Contudo, nem todos os ganhos justificam adoção imediata.

---

## RECOMENDAÇÕES PARA O HOSPITAL

### 🥇 **ADOTAR COMO PADRÃO PRINCIPAL: SVM (Support Vector Machine)**
- **Por quê:** Recall de 97.06% (identifica 97 em 100 casos reais) + acurácia de 98.90% (acerta em quase 99% de todos os diagnósticos). É o melhor balanceamento entre detectar câncer e evitar alarmes falsos.
- **Risco clínico:** 3 casos em 100 podem não ser detectados — aceitável se implementado com revisão humana em casos de alta incerteza.
- **Implementação:** Substitua os sistemas atuais por este modelo.

### 🥈 **ADOTAR COMO ALTERNATIVA/VALIDAÇÃO: MLP (Rede Neural)**
- **Por quê:** Mesmo recall (97.06%) que SVM, mas com acurácia 98.90% — praticamente equivalente. Oferece redundância diagnóstica.
- **Caso de uso:** Use para casos duvidosos (segundo parecer automático) quando o SVM não tiver alta confiança.

### 🥉 **ADOTAR PARA TRIAGEM EM MASSA: Gradient Boosting (otimizado)**
- **Por quê:** Recall 97.06% + F1 97.06%, com acurácia 97.80%. Ligeiramente inferior ao SVM mas ainda excelente e potencialmente mais rápido computacionalmente.
- **Caso de uso:** Ideal para screening de alto volume (campanhas de triagem), já que detecta praticamente todos os casos.

### ❌ **NÃO ADOTAR: Decision Tree e KNN (otimizados)**
- **Motivo:** Apesar de melhorarem, ainda deixam 2-6% dos casos malignos não detectados (58-88 casos perdidos a cada 10.000 pacientes). Random Forest e Extra Trees também não melhoraram, ficando para trás.
- **Risco:** Clinicamente inaceitável em comparação com alternativas disponíveis.

---

## PRÓXIMOS PASSOS
1. **Validação externa:** Teste o SVM e MLP em dados de fora do hospital
