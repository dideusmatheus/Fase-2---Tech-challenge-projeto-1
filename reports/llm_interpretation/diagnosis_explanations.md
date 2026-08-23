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

**Interpretação para o médico:**

O modelo SVM classificou este caso como **benigno com 95% de confiança**. As características mais relevantes mostram um padrão ligeiramente atípico: a dimensão fractal (complexidade da borda) está discretamente elevada (+1.10 dp), enquanto as medidas de textura estão consistentemente reduzidas em relação à população de referência (entre -0,84 e -1,06 dp). Esse perfil — com textura menos heterogênea e simetria preservada — é compatível com lesões benignas no modelo treinado. Apesar da alta confiança, ressalta-se que esta é uma recomendação de apoio: a avaliação clínica, achados de imagem complementar e história do paciente permanecem determinantes para a decisão diagnóstica final.

**Explicação para o paciente (linguagem simples):**

O modelo de apoio analisou sua imagem de mamografia e indicou resultado **benigno**, com uma confiança bem alta de 95%. Basicamente, ele identificou que as características observadas no exame são mais compatíveis com algo que não é câncer — por exemplo, poderia ser um nódulo simples, uma área de densidade normal ou outra alteração comum e não maligna. É importante lembrar que essa é uma indicação da máquina para ajudar o médico, e não uma confirmação definitiva, então é essencial que você converse com o radiologista ou seu médico para que ele revise o resultado, confirme o que viu e oriente você sobre os próximos passos, se houver necessidade de acompanhamento.

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

**Interpretação da Predição para o Paciente ID 879523:**

O modelo SVM classificou o caso como **Benigno, porém com baixa confiança (44.9%)**, o que indica um resultado incerto. As características mais relevantes mostram textura notavelmente suave (texture_se e texture_worst abaixo da média), um padrão geralmente associado a lesões benignas. Por outro lado, há elevação em simetria e dimensão fractal na pior medição, características que podem sugerir malignidade, criando conflito na decisão do modelo. **Esta é uma predição frágil** — a confiança abaixo de 50% reflete essa dificuldade. Recomenda-se análise clínica adicional (avaliação radiológica detalhada, correlação clínica ou follow-up) antes de qualquer conclusão, pois o modelo sozinho não oferece segurança suficiente neste caso.

**Explicação para o paciente (linguagem simples):**

O modelo de inteligência artificial analisou seu exame e indicou resultado **benigno**, ou seja, sem sinais de câncer. No entanto, é importante você saber que a confiança do modelo nesse resultado é de cerca de 45%, o que significa que há bastante incerteza — o modelo não está muito seguro dessa conclusão. Alguns padrões encontrados no exame foram um pouco incomuns, o que explica essa dúvida. Por isso é essencial que você converse com seu médico para que ele revise o exame em detalhes, tire suas dúvidas e defina se é necessário fazer outros testes ou acompanhamento. Ele é quem pode confirmar o resultado e orientá-lo melhor sobre os próximos passos.

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

O modelo SVM classificou este caso como **maligno** baseando-se principalmente em características morfológicas do núcleo celular que apresentam desvios significativos: o raio (radius_se) está aumentado, assim como a concavidade média e os pontos côncavos, sugerindo contornos nucleares irregulares e complexos — padrões típicos de malignidade. A variabilidade anormalmente elevada na área (area_se) reforça essa impressão de instabilidade celular. Embora o modelo expresse 100% de confiança, esta é uma **recomendação técnica que deve ser integrada à avaliação clínica completa**, correlacionando com achados de imagem, história do paciente e outros exames pertinentes. A decisão diagnóstica final é responsabilidade exclusiva do médico assistente.

**Explicação para o paciente (linguagem simples):**

O modelo identificou características no seu exame que o levaram a indicar um resultado de **malignidade** — ou seja, sugestivo de câncer. Especificamente, ele detectou que certas formas e estruturas na imagem da mama estão diferentes do que seria considerado normal, com padrões que o modelo associa a tumores malignos. A confiança dele nessa indicação é muito alta, mas é importante deixar claro: **isso é uma indicação do modelo, não um diagnóstico confirmado**. Apenas um médico especialista, após análise completa do seu caso, exames adicionais se necessário e até biópsias, pode confirmar ou descartar essa possibilidade. **É essencial que você converse com urgência com o médico responsável pelo seu caso para discutir esses resultados e definir os próximos passos.**

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

O modelo SVM classificou este paciente como **Maligno com confiança de 64,7%**, baseando-se principalmente em texturas mais irregulares (texture_se e texture_worst elevadas) e presença de pontos côncavos aumentados, características associadas a lesões malignas no dataset de treinamento. A simetria levemente reduzida e a dimensão fractal ligeiramente elevada reforçaram essa tendência, embora nenhuma medida individual seja extremamente discrepante. **Ressalte-se que 64,7% é uma confiança moderada** — não é uma previsão de alto risco — e a análise clínica conjunta com achados de imagem, biópsia e contexto do paciente é fundamental para a decisão final sobre conduta.

**Explicação para o paciente (linguagem simples):**

O modelo de apoio indicou um resultado de **possível malignidade** neste exame, com uma confiança de cerca de 65% — ou seja, há uma indicação preocupante, mas não é uma certeza absoluta. O que chamou a atenção foram algumas características da imagem que o modelo associa mais frequentemente com casos que precisam de atenção: a textura do tecido apresentou variações que saíram do padrão, e algumas formas nas bordas da lesão também mostraram características que merecem investigação. Isso não é um diagnóstico confirmado — é um alerta do modelo para que você converse urgentemente com seu médico, que pode fazer uma análise mais profunda, possivelmente solicitar outros exames, e definir o melhor caminho para você. É fundamental que você marque uma consulta o quanto antes para discutir esses achados com um especialista.

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

O modelo SVM classificou este caso como **maligno com confiança de 59.6%** — um resultado moderado que requer cautela clínica. A predição foi influenciada principalmente por **elevada assimetria da lesão** (simetria_pior 1,93 desvios-padrão acima da média) e **dimensão fractal aumentada** (ambas as medidas de dimensão fractal ~1,4 desvios-padrão acima), sugerindo maior complexidade e irregularidade estrutural — características frequentemente associadas a malignidade. Adicionalmente, a **compactidade elevada** (em torno de 1,17 desvios-padrão) reforça esse padrão. No entanto, a confiança moderada (59,6%) indica sobreposição com características benignas, não permitindo certeza. **Recomenda-se análise clínica integrada com exame físico, imagenológico e eventual confirmação histológica, sendo a decisão final exclusivamente do médico assistente.**

**Explicação para o paciente (linguagem simples):**

O modelo analisou suas imagens e indicou um resultado de **câncer de mama**, com uma confiança de cerca de 60% — o que significa que há incerteza significativa nesse achado. O modelo detectou algumas características nas imagens que costumam aparecer em casos de câncer, como uma forma um pouco irregular e uma textura que saiu do padrão esperado, mas esses sinais sozinhos não são conclusivos. É muito importante que você converse com seu médico assim que possível para que ele revise essas imagens, considere seu histórico completo e decida se são necessários outros testes para confirmar ou descartar essa possibilidade.

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

**Explicação da Predição para Paciente ID 865137:**

O modelo SVM classificou o exame como **Benigno com 99,8% de confiança**. A principal razão é que as medidas de textura (texture_mean, texture_worst e texture_se) apresentam valores **significativamente menores** que o padrão do dataset — características tipicamente associadas a lesões benignas. O raio do erro padrão (radius_se) também está discretamente reduzido, reforçando esse padrão. Apenas a suavidade máxima (smoothness_worst) apresenta ligeiro aumento, mas insuficiente para contrapesar os demais achados favoráveis a benignidade. **Importante:** este é um resultado de apoio; a avaliação clínica integral e a correlação com achados de imagem e história clínica do paciente permanecem sob responsabilidade do médico assistente.

**Explicação para o paciente (linguagem simples):**

O modelo de inteligência artificial analisou seu exame e indicou resultado **benigno**, com muito alta confiança nessa indicação (99,8%). Isso significa que as características que o modelo observou na imagem — como a textura e o formato das estruturas analisadas — se assemelham mais aos padrões de lesões não cancerosas do que cancerosas. No entanto, é importante lembrar que este é um apoio ao diagnóstico, não a confirmação final: apenas seu médico pode confirmar este resultado, avaliar o contexto clínico completo e orientar os próximos passos. Agende uma consulta com o radiologista ou o médico responsável para discutir esses achados e esclarecer qualquer dúvida.

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

# Interpretação da Predição — Paciente ID 901303

O modelo SVM classificou este caso como **benigno com alta confiança (91,8%)**. As medidas que mais influenciaram essa predição mostram padrões consistentemente *abaixo da média* do dataset: textura e raio com variabilidade reduzida (desvios entre -1,3 e -0,8), além de suavidade e perímetro também menores. Esse perfil de características menos agressivas — com menos heterogeneidade e tamanhos menores — é típico de lesões benignas segundo o modelo. **Importante:** essa é uma recomendação computacional e não substitui sua avaliação clínica; correlacione sempre com imagem, história clínica e achados físicos antes de qualquer conclusão diagnóstica.

**Explicação para o paciente (linguagem simples):**

O modelo de inteligência artificial analisou seu exame e indicou resultado **benigno**, com confiança de 91,8% — o que significa que ele encontrou características que são mais frequentemente vistas em casos sem câncer. Em resumo, as características analisadas do seu tecido mamário (como sua textura e formato) não mostram padrões de alerta que costumam estar presentes em tumores malignos. **Mas lembre-se: essa é apenas uma indicação do modelo, não é um diagnóstico confirmado.** É fundamental que você converse com seu médico responsável para que ele revise o exame, considere seu histórico completo e confirme esse resultado — só assim você terá a segurança de uma avaliação médica real.
