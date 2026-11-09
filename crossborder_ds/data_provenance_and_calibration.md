# Metodología de Procedencia, Calibración Empírica y Física Estocástica

> **Documento de Auditoría Metodológica para Data Science & Quantitative FinTech**  
> **Proyecto:** Flywire Cross-Border Settlement Friction & Risk Engine (`crossborder_ds`)  
> **Versión:** 2.0 (Física Estocástica M/G/1, Ventanas RTGS y Zero-Leakage)  
> **Entorno:** Producción / Benchmark Cuantitativo  

---

## 1. Resumen Ejecutivo y Marco Regulatorio

La modelización de pagos transfronterizos (*cross-border payments*) en el sector educativo y de salud internacional (el núcleo de negocio de Flywire) exige superar modelos sintéticos aditivos o gaussianos. En la realidad bancaria global, los tiempos de liquidación no siguen una campana de Gauss simétrica: presentan **distribuciones fuertemente asimétricas con cola pesada (*heavy-tailed leptokurtic distributions*)**, bimodales debido a los cortes de fin de semana y con amplificaciones no lineales inducidas por colas de espera en los bancos corresponsales durante las temporadas de matrículas universitarias.

Este documento formaliza la **procedencia de datos, fuentes estadísticas oficiales y calibración matemática** de la suite de simulación física e ingeniería de variables (`crossborder_ds/`), en estricto cumplimiento con los estándares del **Comité de Pagos e Infraestructuras de Mercado (CPMI) del Banco de Pagos Internacionales (BIS)**, el **Banco Mundial**, **SWIFT** y los marcos regulatorios cambiarios de los principales bancos centrales de origen (**SAFE China, RBI India y Bacen Brasil**).

---

## 2. Fuentes Oficiales y Calibración de Parámetros

### 2.1. BIS CPMI (Bank for International Settlements - CPMI)
* **Referencia Oficial:** *CPMI Report to the G20: Enhancing Cross-Border Payments: Stage 3 Roadmap* (2020–2024); *CPMI Fast Payments and Settlement Interoperability Reports*.
* **Métricas de Referencia:**
  * El G20 estableció como objetivo que para 2027 el 75% de los pagos transfronterizos liquiden en menos de 1 hora.
  * Sin embargo, en el sistema bancario corresponsal heredado (*legacy correspondent banking*), el **28% al 40% de las transferencias educativas procedentes de mercados emergentes (China, India, Brasil) superan las 48 horas hábiles**, llegando hasta 5 o 7 días naturales cuando colisionan con fines de semana o revisiones documentales.
* **Calibración en el Modelo:**
  * La probabilidad de retraso global $>48$ horas del motor físico se calibra deliberadamente en el rango **30%–38%**, reflejando la realidad operativa actual antes de optimización algorítmica.
  * Se modela el número de saltos corresponsales (*intermediary hops*) entre 1 y 3 según el corredor, con penalizaciones compuestas por salto.

---

### 2.2. SWIFT GPI & Migración ISO 20022 (CBPR+)
* **Referencia Oficial:** *SWIFT Global Payments Innovation (GPI) Observer Analytics*; *ISO 20022 CBPR+ Implementation Guidelines (`pacs.008.001.08` vs legacy `MT103`)*.
* **Métricas de Referencia:**
  * **Tasas de Procesamiento Directo (STP - Straight-Through Processing):**
    * Mensajes nativos estructurados `ISO20022_PACS008` (con identificadores delimitados `<RmtInf>`): Tasa STP $>94\%$, con riesgo de error o retención $<3\%$.
    * Mensajes tradicionales `MT103_STP`: Tasa STP $\approx 82\%$, riesgo de excepción $\approx 8\%$.
    * Mensajes manuales `MT103_NON_STP`: Tasa STP se desploma al $60\%–65\%$, con un riesgo de reparación manual o retraso de hasta el $32\%$.
  * **Campo Narrativo `:70:` (Remittance Information):** En transferencias MT103 no estructuradas, la inclusión de texto libre ("tuition fee for my son please clear urgent no fees") sin identificador de matrícula genera fallos de conciliación automática en el receptor.
* **Calibración en el Modelo:**
  * Se genera sintéticamente el campo `:70:` respetando las distribuciones de caracteres de cada formato.
  * Se calcula la **Entropía de Shannon** $H = -\sum_{c \in \Sigma} p(c) \log_2 p(c)$ sobre el texto para cuantificar el desorden léxico. A mayor entropía y falta de estructura, mayor es la probabilidad de reparación STP ($8$ a $20$ horas activas) y de cribado de sanciones.

---

### 2.3. Banco Mundial: Remittance Prices Worldwide (RPW) y Comisiones Deducidas (*Lifting Fees*)
* **Referencia Oficial:** *World Bank Remittance Prices Worldwide (RPW) Database & Quarterly Reports*; *Payment Aspects of Financial Inclusion (PAFI)*.
* **Métricas de Referencia:**
  * Cláusula de gastos `:71A:`:
    * `OUR`: El emisor asume todos los costes.
    * `SHA` (Shared): Cada parte asume las comisiones de su banco.
  * En transferencias académicas procesadas bajo cláusula `SHA` con $\ge 2$ intermediarios, los bancos corresponsales intermedios deducen un canon (*lifting fee*) de **\$15 a \$35 USD** directamente del importe principal transferido.
  * **Impacto en el Negocio Escolar:** La universidad recibe una cantidad neta inferior al importe exacto de la matrícula (ej. factura de \$15.000 vs recepción de \$14.975). Esto no es un rechazo bancario formal, pero **dispara una retención administrativa (*administrative reconciliation hold*)** en la tesorería universitaria, demorando la confirmación de la matrícula entre $16$ y $36$ horas adicionales mientras el estudiante o la agencia regulariza la discrepancia.
* **Calibración en el Modelo:**
  * Simulación estocástica de `charge_bearer` (`SHA` vs `OUR`).
  * Aplicación de deducciones de \$15 a \$35 USD en transferencias `SHA` multietapa, provocando la retención administrativa correspondiente en el reloj de liquidación.

---

### 2.4. Regulaciones Cambiarias y Bancos Centrales Nacionales

| Jurisdicción | Autoridad / Sistema RTGS | Ventana de Corte (UTC) | Apertura (UTC) | Desfase ($\Delta \text{tz}$) | Marco Regulatorio y Fricción Asociada |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **China (CNY)** | **PBOC / CNAPS** | **09:00 UTC** (17:00 CST) | 01:00 UTC | $+13.0$ h vs USD | **SAFE (State Administration of Foreign Exchange):** Cuota anual personal de \$50.000 USD. Verificación documental estricta de admisión escolar (`Form I-20` / oferta universitaria). Probabilidad de inspección: ~22%, demora: 14–32h. |
| **India (INR)** | **RBI / NEFT-RTGS** | **11:30 UTC** (17:00 IST) | 03:30 UTC | $+10.5$ h vs USD | **RBI LRS (Liberalised Remittance Scheme):** Declaración Formulario A2 y retención fiscal en origen TCS (*Tax Collected at Source* de hasta el 5%–20%). Probabilidad de inspección: ~18%, demora: 12–26h. |
| **Brasil (BRL)** | **Bacen / STR** | **20:00 UTC** (17:00 BRT) | 09:30 UTC | $+2.0$ h vs USD | **Banco Central do Brasil / Receita Federal:** Tributación IOF (*Imposto sobre Operações Financeiras*) y registro de Contrato de Câmbio. Probabilidad de inspección: ~15%, demora: 12–24h. |
| **Estados Unidos (USD)** | **Federal Reserve / Fedwire** | **23:30 UTC** (18:30 EST) | 12:00 UTC | Base ($0.0$ h) | Cámara de compensación central para transferencias USD. |
| **Eurozona (EUR)** | **BCE / Target2** | **17:00 UTC** (18:00 CET) | 06:00 UTC | $+6.0$ h vs USD | Compensación RTGS europea. |
| **Reino Unido (GBP)** | **Bank of England / CHAPS** | **17:00 UTC** (18:00 BST) | 06:00 UTC | $+5.0$ h vs USD | Compensación RTGS británica para libras esterlinas. |
| **Canadá (CAD)** | **Payments Canada / Lynx** | **22:30 UTC** (17:30 EST) | 11:00 UTC | $0.0$ h vs USD | Sistema de pagos de alto valor canadiense. |

---

## 3. Formulación de la Física Estocástica

### 3.1. Red de Colas en Tándem $M/G/1$ y Fórmula de Pollaczek–Khinchine
Para cada uno de los $K$ saltos corresponsales ($k = 1, \dots, K$):
1. **Tiempo de Servicio de Cola Pesada ($S_k$):**  
   El tiempo que un banco tarda en procesar el mensaje sigue una distribución Lognormal con cola Pareto:
   $$S_k \sim \text{Lognormal}\left(\mu_S, \sigma_S = 0.50\right)$$
   con probabilidad $\approx 8\%$ de captura por una cola dura de Pareto ($\alpha = 2.0$) por revisiones de crédito u operativas.
2. **Tiempo Medio de Espera en Cola ($W_q$):**  
   Aproximado mediante la relación de Pollaczek–Khinchine / Kingman:
   $$W_{q,\text{mean}} = \left( \frac{\rho_t}{1 - \rho_t} \right) \cdot \left( \frac{1 + C_s^2}{2} \right) \cdot E[S_k]$$
   donde:
   * $\rho_t$ es el factor de saturación y utilización de la red bancaria en la fecha $t$.
   * $C_s^2 = \frac{\operatorname{Var}(S_k)}{(E[S_k])^2}$ es el coeficiente de variación al cuadrado del servicio ($C_s^2 \approx 0.75$).
3. **Realización Estocástica de la Espera:**  
   $$W_{q,k} \sim \text{Gamma}\left(k=2, \theta = \frac{W_{q,\text{mean}}}{2}\right)$$
   Esto modela fluctuaciones operativas realistas por turnos bancarios y disponibilidad de procesadores.

### 3.2. Estacionalidad No Lineal de Admisiones Universitarias
La intensidad de llegadas y saturación $\rho_t$ varía de acuerdo con el calendario global de educación superior (UNESCO / IIE Open Doors):
* **Pico Otoño (Fall Intake):** 1 de agosto al 25 de septiembre. La saturación escala desde $\rho_0 \approx 0.32$ hasta $\rho_t \approx 0.75 - 0.85$ (centrado en el 25 de agosto).
* **Pico Primavera (Spring Intake):** 10 de diciembre al 25 de enero ($\rho_t \approx 0.62 - 0.70$).
* **Periodo Ordinario:** $\rho_t \approx 0.32$.

Debido al factor $\frac{\rho_t}{1 - \rho_t}$, cuando la utilización sube de $0.32$ a $0.80$, la espera en cola se multiplica por:
$$\frac{0.80 / 0.20}{0.32 / 0.68} \approx \frac{4.0}{0.47} \approx 8.5\times$$
generando el colapso no lineal de tiempos típico de las fechas límites de pago de matrícula.

---

### 3.3. Discontinuidad de Ventanas RTGS y Abismo de Fin de Semana (*Weekend Cliff*)
El reloj de liquidación bancaria es discontinuo:
1. **Llegada tras el corte ($t_{\text{create}} \ge \text{cut\_off}$):** La transacción no entra en el ciclo diario. Salta forzosamente hasta la hora de apertura del siguiente día hábil ($\text{open\_hour\_utc}$).
2. **Abismo de Fin de Semana:** Si la orden llega un viernes por la tarde después del corte, salta automáticamente al lunes por la mañana:
   $$\Delta t_{\text{salto}} \approx 60 - 66 \text{ horas}$$
   antes de que el primer banco corresponsal comience el procesamiento activo.
3. **Pausa en Fin de Semana:** El avance de horas activas se suspende los sábados y domingos a las 00:00 UTC y se reanuda los lunes a las 07:00 UTC.

---

### 3.4. Cribado AML y Sanciones con Falsos Positivos Dinámicos
La probabilidad de retención cautelar por cribado de sanciones AML (OFAC, GAFI, PEP) tiene una tasa base empírica de $\sim 2.5\%$, pero escala dinámicamente según la calidad y claridad de la instrucción:
$$P(\text{AML Hold}) = \min\left(0.30, \, 0.025 \cdot \phi_{\text{formato}} \cdot \phi_{\text{entropía}} \cdot \phi_{\text{banco}} \cdot \phi_{\text{ref}}\right)$$
* $\phi_{\text{formato}} = 2.0$ si el mensaje es `MT103_NON_STP`.
* $\phi_{\text{entropía}} = 1.0 + 1.2 \cdot \max(0, H - 4.2)$ si el texto en `:70:` es caótico o ambiguo.
* $\phi_{\text{banco}} = 1.3$ si el banco emisor es de Nivel 3 (*Tier 3 Local Credit*).
* $\phi_{\text{ref}} = 1.2$ si no cuenta con referencia estructurada.

Si se activa, el expediente sufre una demora de inspección manual de $24$ a $48$ horas (con un 20% de probabilidad de investigación reforzada Pareto de hasta $72$–$100$ horas).

---

## 4. Contrato Arquitectónico de Cero Fuga de Información (*Zero Data Leakage*)

Para garantizar la integridad científica y que los modelos predictivos sean auditables ante reguladores financieros, se impone un **cortafuegos estricto de información**:

```
+---------------------------------------------------------------------------------+
|                   ENTORNO FÍSICO ESTOCÁSTICO (SIMULADOR BANCARIO)               |
|                                                                                 |
| Variables de Telemetría Latente (INTERNAS - Prohibidas en Feature Pipeline):   |
|  - queue_wait_hours                                                             |
|  - hop_service_hours                                                            |
|  - aml_hold_applied                                                             |
|  - aml_investigation_hours                                                      |
|  - lifting_fee_applied                                                          |
|  - lifting_fee_deducted_usd                                                     |
|  - admin_hold_hours                                                             |
|  - capital_control_hours                                                        |
|  - stp_repair_hours                                                             |
|  - active_processing_hours                                                      |
|                                                                                 |
| Etiquetas Ground-Truth (Solo para evaluación y cálculo de pérdida económica):    |
|  - actual_settlement_hours                                                      |
|  - is_delayed_gt_48h                                                            |
|  - stranded_capital_cost_usd                                                    |
|  - total_friction_cost_usd                                                      |
+---------------------------------------------------------------------------------+
                                      |
                                      | [BARRERA DE INFORMACIÓN ESTRICTA EN t=0]
                                      v
+---------------------------------------------------------------------------------+
|                PIPELINE DE INGENIERÍA DE VARIABLES (features.py)                |
|                                                                                 |
| Atributos Rigurosamente Observables al Momento de Creación (t=0):              |
|  - amount_usd_equiv (log-transformado)                                          |
|  - fx_volatility_7d (volatilidad macroeconómica histórica observable)           |
|  - origin_bank_tier_num                                                         |
|  - intermediary_hops (ruta esperada según directorio BIC/SWIFT)                 |
|  - is_iso20022 / is_mt103_non_stp                                               |
|  - is_sha_charge (código :71A: especificado en la orden)                         |
|  - shannon_entropy_f70 (calculada sobre el texto :70: provisto por el usuario)  |
|  - hours_to_rtgs_cutoff (margen horario respecto al Banco Central de origen)    |
|  - timezone_dispersion (|UTC_origen - UTC_destino|)                             |
|  - weekend_cliff_risk (interacción cuadrática temporal t=0)                    |
|  - corridor_eb_prior (suavizado Bayesiano empírico calibrado en train)          |
+---------------------------------------------------------------------------------+
```

### Pruebas Automatizadas de Garantía
1. **Test de No Fuga (`test_zero_information_leakage_in_features`):** Inspecciona exhaustivamente que ninguna de las 14 variables latentes del simulador o métricas de coste forme parte del vector $X$ de características.
2. **Test Adversarial:** Inyecta deliberadamente columnas latentes en un `DataFrame` de prueba y verifica que `CrossBorderFeaturePipeline.transform()` las purgue de manera determinista.

---

## 5. Validación Estadística y Verificación de Estabilidad

Las propiedades matemáticas del generador y el pipeline se auditan formalmente mediante tests automatizados en `pytest`:

1. **Test de Cola Pesada (`test_heavy_tailed_settlement_distribution`):**
   * Asimetría positiva: $\text{Skewness} > 1.0$ (calibrado típicamente en $\approx 1.8$–$2.2$).
   * Exceso de Curtosis leptocúrtica: $\text{Kurtosis} > 2.5$ (calibrado típicamente en $\approx 5.0$–$8.0$).
   * Esto demuestra que la física reproduce eventos raros extremos, rechazando cualquier hipótesis de normalidad gaussiana ingenua.
2. **Índice de Estabilidad de Población (`test_population_stability_index_psi`):**
   $$\text{PSI} = \sum_{b=1}^B (P_b - Q_b) \ln\left(\frac{P_b}{Q_b}\right) < 0.20$$
   Valida que la partición cronológica (Train 70%, Val 15%, Test 15%) no introduzca degradación distributiva espuria entre conjuntos de entrenamiento y prueba.
3. **Discriminación y Aporte Predictivo:**
   * LightGBM Classifier: $\text{ROC-AUC} \ge 0.85$ y $\text{Accuracy} \ge 0.75$.
   * LightGBM Regressor: $\text{MAE} < 25.0$ horas.
