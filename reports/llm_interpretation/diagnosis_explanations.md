# Explicações de Diagnóstico — Etapa 3

## Câncer Perdido (Falso Negativo — Real Maligno, Previsto Benigno)

### Paciente 859983

- **Predição do modelo:** Benigno (95.0% de confiança)
- **Diagnóstico real:** Maligno
- **Medidas mais notáveis:**
  - fractal_dimension_worst: +1.10 desvios-padrão
  - texture_se: -1.06 desvios-padrão
  - symmetry_se: -0.98 desvios-padrão
  - texture_worst: -0.84 desvios-padrão
  - texture_mean: -0.84 desvios-padrão

**Explicação técnica (para o médico):**

**Interpretação da Predição - Paciente ID 859983**

O modelo SVM classificou este caso como **Benigno com 95% de confiança**. As medidas mais relevantes mostram um padrão favorável: a textura (tanto média quanto pior) e a simetria estão *abaixo* dos valores típicos de lesões malignas, o que reduz o risco de malignidade. A dimensão fractal está ligeiramente elevada (+1.10 desvios-padrão), mas não o suficiente para contrapesar os demais achados benignos. Em conjunto, essas características morfológicas apontam para uma lesão com perfil mais consistente com processos benignos. **Porém, esta é apenas a recomendação do algoritmo — a avaliação clínica, correlação com achados de imagem e possível confirmação histopatológica permanecem essenciais para a decisão final do médico.**

**Explicação para o paciente (linguagem simples):**

O modelo analisou sua mamografia e indicou resultado **benigno**, com uma confiança de 95%. Isso significa que o padrão encontrado no exame se aproxima muito mais daqueles que costumam ser lesões não cancerosas do que malignas. Algumas características da imagem — como a forma e textura — apresentaram padrões que o modelo associa a resultados benignos. No entanto, lembre-se que essa é uma indicação do modelo, não um diagnóstico confirmado. É fundamental que você converse com seu médico para que ele revise o exame junto com você, confirme esse resultado e oriente os próximos passos, se houver algum.

### Paciente 879523

- **Predição do modelo:** Benigno (44.9% de confiança)
- **Diagnóstico real:** Maligno
- **Medidas mais notáveis:**
  - texture_se: -1.51 desvios-padrão
  - texture_worst: -0.95 desvios-padrão
  - symmetry_worst: +0.81 desvios-padrão
  - fractal_dimension_worst: +0.78 desvios-padrão
  - symmetry_mean: -0.77 desvios-padrão

**Explicação técnica (para o médico):**

**Interpretação da Predição do Modelo (SVM)**

O modelo classificou o caso como **Benigno**, mas com **confiança moderada (44,9%)**, o que indica considerável incerteza. Os achados mais relevantes mostram: textura significativamente *menor* que o padrão (texture_se e texture_worst reduzidas), sugerindo padrão mais homogêneo; porém, há *maior* assimetria e dimensão fractal elevadas nas piores medidas, características que podem estar presentes tanto em lesões benignas quanto malignas. A combinação desses achados contraditórios explica por que o modelo não alcançou alta confiança. **Este resultado deve ser integrado com a avaliação clínica e radiológica completa do paciente — a baixa confiança do modelo reforça a importância do julgamento médico e, se necessário, de investigação adicional.**

**Explicação para o paciente (linguagem simples):**

O modelo indicou que a lesão tem características mais próximas de um caso benigno — mas é importante deixar claro que essa confiança é de apenas 45%, o que significa que o modelo tem muita incerteza nesse resultado. Alguns padrões no seu exame foram um pouco incomuns, o que explica por que a máquina não conseguiu chegar a uma conclusão mais segura. Isso não quer dizer que você deva se preocupar, mas também não quer dizer que tudo está resolvido — o resultado precisa ser confirmado pelo seu médico, que pode pedir exames complementares ou fazer outras avaliações antes de definir o melhor caminho a seguir.

## Acerto Maligno (Verdadeiro Positivo — Real Maligno, Previsto Maligno)

### Paciente 884948

- **Predição do modelo:** Maligno (100.0% de confiança)
- **Diagnóstico real:** Maligno
- **Medidas mais notáveis:**
  - radius_se: +2.22 desvios-padrão
  - concavity_mean: +2.20 desvios-padrão
  - area_se: +2.19 desvios-padrão
  - concave points_mean: +2.06 desvios-padrão
  - concavity_worst: +1.97 desvios-padrão

**Explicação técnica (para o médico):**

**Interpretação para o clínico:**

O modelo SVM classificou este caso como maligno com confiança de 100%, baseando-se principalmente em características morfológicas do núcleo celular significativamente alteradas. As medidas mais desviantes foram o raio do erro padrão (+2,22 DP), concavidade média (+2,20 DP) e área do erro padrão (+2,19 DP) — todas indicando irregularidades no tamanho e contorno das estruturas nucleares que o algoritmo associa a malignidade no dataset de treinamento. Os padrões de concavidade e pontos côncavos também estão notadamente elevados, reforçando a impressão de bordas irregulares características de lesões agressivas. Porém, lembre-se de que este é um auxílio diagnóstico; a avaliação clínica integrada (histórico, exame físico, achados de imagem adicionais) permanece essencial para a decisão final.

**Explicação para o paciente (linguagem simples):**

O modelo de apoio ao diagnóstico indicou um resultado de **lesão maligna** neste exame, com alta confiança. Isso significa que foram identificadas características na imagem que o sistema associa a um padrão preocupante — especialmente relacionadas ao tamanho e à forma irregular das células analisadas. No entanto, é importante deixar claro que **esta é apenas uma indicação do modelo**, não uma confirmação definitiva, pois modelos de inteligência artificial podem cometer erros. O próximo passo agora é você conversar com o seu médico responsável o quanto antes: ele vai revisar esses resultados, pode solicitar exames adicionais se necessário, e só então poderá confirmar o diagnóstico e discutir as opções de tratamento com você. Não deixe de agendar essa consulta com urgência.

### Paciente 862548

- **Predição do modelo:** Maligno (64.7% de confiança)
- **Diagnóstico real:** Maligno
- **Medidas mais notáveis:**
  - texture_se: +1.07 desvios-padrão
  - texture_worst: +0.84 desvios-padrão
  - symmetry_se: -0.69 desvios-padrão
  - concave points_worst: +0.63 desvios-padrão
  - fractal_dimension_worst: +0.55 desvios-padrão

**Explicação técnica (para o médico):**

O modelo SVM classificou este caso como **maligno com confiança de 64,7%**, principalmente porque as características de textura (texture_se e texture_worst) estão elevadas em relação à população de referência, sugerindo padrões de heterogeneidade tecidual frequentemente associados a malignidade; adicionalmente, os pontos côncavos elevados (concave points_worst) reforçam essa indicação, enquanto a simetria ligeiramente reduzida complementa o perfil. No entanto, a confiança moderada (não superior a 70%) indica que não há um padrão extremamente típico de malignidade, recomendando correlação clínica e, se necessário, confirmação por métodos adicionais. A decisão diagnóstica final permanece com o profissional de saúde responsável.

**Explicação para o paciente (linguagem simples):**

O modelo indicou um resultado preocupante nesse exame — ele sugeriu que a lesão pode ser maligna, com uma confiança de cerca de 65%. Isso significa que o modelo identificou algumas características na imagem que aparecem mais frequentemente em casos de câncer, principalmente relacionadas à textura e formato da lesão. No entanto, essa é apenas uma indicação do modelo de apoio, não uma confirmação diagnóstica — modelos podem se enganar, e apenas o seu médico pode confirmar o que está realmente acontecendo. É muito importante que você agora marque uma consulta urgente com o radiologista ou oncologista responsável para que ele analise o exame junto com você, tire suas dúvidas e defina os melhores passos seguintes.

## Alarme Falso (Falso Positivo — Real Benigno, Previsto Maligno)

### Paciente 8810158

- **Predição do modelo:** Maligno (59.6% de confiança)
- **Diagnóstico real:** Benigno
- **Medidas mais notáveis:**
  - symmetry_worst: +1.93 desvios-padrão
  - fractal_dimension_mean: +1.48 desvios-padrão
  - fractal_dimension_worst: +1.36 desvios-padrão
  - compactness_worst: +1.17 desvios-padrão
  - compactness_mean: +0.82 desvios-padrão

**Explicação técnica (para o médico):**

# Interpretação da Predição - Paciente ID 8810158

O modelo SVM classificou este paciente como **potencialmente maligno com confiança moderada (59,6%)**, baseando-se principalmente em características de **irregularidade e complexidade celular**. As medidas mais desviantes foram: simetria anormalmente elevada na pior medição (+1,93 DP), dimensão fractal aumentada tanto na média quanto na pior medição (+1,48 e +1,36 DP), e compactação celular elevada (+1,17 DP na pior medição). Esses padrões — particularmente a menor simetria e maior complexidade estrutural — são associados pelo modelo a características malignas. **Contudo, a confiança de 59,6% indica incerteza considerável**, sugerindo que este é um caso borderline que requer correlação cuidadosa com achados clínicos, imagiológicos e história do paciente. A decisão clínica final permanece com você.

**Explicação para o paciente (linguagem simples):**

O modelo analisou sua mamografia e indicou um resultado preocupante: encontrou características que, segundo seu treinamento, se associam mais frequentemente com câncer de mama. Porém, é importante saber que a confiança do modelo nessa indicação é moderada (cerca de 60%), o que significa que existe uma margem real de incerteza — o modelo pode estar certo ou errado. O que o modelo detectou foram algumas irregularidades na forma e na estrutura do tecido, mas isso não é uma confirmação de diagnóstico. É absolutamente essencial que você converse com seu médico responsável assim que possível para que ele avalie esses achados, possivelmente solicite exames adicionais e tome uma decisão sobre os próximos passos com base em sua análise clínica completa.

## Acerto Benigno (Verdadeiro Negativo — Real Benigno, Previsto Benigno)

### Paciente 865137

- **Predição do modelo:** Benigno (99.8% de confiança)
- **Diagnóstico real:** Benigno
- **Medidas mais notáveis:**
  - texture_mean: -2.00 desvios-padrão
  - texture_worst: -1.66 desvios-padrão
  - texture_se: -1.34 desvios-padrão
  - radius_se: -1.00 desvios-padrão
  - smoothness_worst: +0.97 desvios-padrão

**Explicação técnica (para o médico):**

O modelo SVM classificou este caso como **benigno com alta confiança (99,8%)**. Os achados do exame mostram padrões principalmente associados a lesões benignas: texturas significativamente *menores* que a média (texture_mean e texture_worst bem abaixo dos padrões malignos), variabilidade reduzida nas medidas de textura e raio, com apenas a suavidade na pior medição ligeiramente elevada. Esses valores combinados apontam para características morfológicas tipicamente observadas em lesões não-malignas. **Importante**: esta é uma recomendação baseada em padrões estatísticos do modelo; a avaliação clínica e correlação com achados de imagem e história clínica permanecem fundamentais para a decisão diagnóstica final.

**Explicação para o paciente (linguagem simples):**

O modelo de inteligência artificial analisou suas imagens de mamografia e indicou resultado **benigno**, com uma confiança muito alta (99,8%). Em termos práticos, isso significa que o padrão encontrado no seu exame é compatível com tecido saudável, sem características que sugiram algo preocupante. O que chamou atenção foi que algumas características das imagens — como a uniformidade e textura do tecido — apresentaram padrões bem consistentes com casos benignos já catalogados. Isso é uma indicação positiva do modelo, mas lembre-se que é só uma indicação de apoio: seu médico responsável é quem vai confirmar esse resultado, analisar o contexto clínico completo e orientar os próximos passos. Converse com ele para tirar todas as suas dúvidas e fechar o diagnóstico com segurança.

### Paciente 901303

- **Predição do modelo:** Benigno (91.8% de confiança)
- **Diagnóstico real:** Benigno
- **Medidas mais notáveis:**
  - texture_se: -1.29 desvios-padrão
  - texture_worst: -1.13 desvios-padrão
  - radius_se: -0.87 desvios-padrão
  - smoothness_se: -0.82 desvios-padrão
  - perimeter_se: -0.79 desvios-padrão

**Explicação técnica (para o médico):**

**Interpretação da predição para o paciente ID 901303:**

O modelo SVM classificou este caso como **benigno com confiança de 91.8%**, baseando-se principalmente em características que se desviam *para baixo* dos padrões típicos de malignidade no dataset. As medidas de textura (texture_se e texture_worst) estão significativamente reduzidas (≈1.3 e 1.1 desvios-padrão abaixo da média), assim como o raio e a suavidade, que são indicadores que tendem a ser elevados em lesões malignas. Essas características mais "suavizadas" e menores apontam para um padrão mais consistente com lesões benignas. Porém, lembramos que este é um resultado de modelo preditivo com boa confiança, mas **não substitui a avaliação clínica completa do médico**, que deve considerar todo o contexto clínico, imaginológico e investigativo do paciente.

**Explicação para o paciente (linguagem simples):**

O modelo de inteligência artificial analisou seu exame e indicou resultado **benigno**, com uma confiança de 91,8% nessa indicação. Isso significa que as características observadas no exame são consistentes com achados que não apresentam sinais de malignidade — basicamente, o padrão encontrado é mais comum em casos sem câncer. É importante lembrar que este é um resultado de apoio ao diagnóstico e não substitui a avaliação do seu médico, que é quem pode confirmar esse resultado, correlacionar com seus sintomas e histórico, e orientar os próximos passos. Agende uma consulta com o radiologista ou mastologista responsável para discutir esses achados e tirar todas as suas dúvidas.
