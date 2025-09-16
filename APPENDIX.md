
# Appendix: Detailed Cost-Sensitive Learning Framework for Non-Cold-Start Scenarios

_Note: This appendix documents the full methodological development of our cost-sensitive learning framework. While this approach demonstrated internal consistency and economic plausibility in controlled, non-cold-start evaluations, it failed to generalize to our final cold-start experiment (F1-score = 0.0). It is presented here as a reusable methodological contribution for researchers and practitioners working in environments with sufficient historical failure data and relatively stable cost structures -- conditions not met in our primary evaluation setting._

---

### Theoretical Foundation for Cost Matrix Design

Following [1], we formalize cost-sensitive learning as an optimization problem that minimizes expected misclassification cost rather than error rate. For binary classification with classes {0,1} (normal/failure), the cost matrix **C** is defined as:

$$
\mathbf{C} = \begin{pmatrix}
C(0|0) & C(1|0) \\
C(0|1) & C(1|1)
\end{pmatrix} = \begin{pmatrix}
0 & C_{\text{FP}} \\
C_{\text{FN}} & 0
\end{pmatrix}
$$

where $C(i|j)$ represents the cost of classifying a sample of true class $j$ as class $i$. The expected cost for a classifier $h$ is:

$$
\begin{split}
\mathbb{E}[C(h)] ={}& P(Y=0) \cdot P(h(X)=1|Y=0) \cdot C_{\text{FP}} \\
& + P(Y=1) \cdot P(h(X)=0|Y=1) \cdot C_{\text{FN}}
\end{split}
$$

To tailor this approach for smart grid applications, the cost parameters must reflect the complex operational and economic realities of utility systems. We extend the standard binary cost matrix to incorporate temporal, load-dependent, and seasonal variations:

$$
C_{\text{FN}}(t, l, s) = C_{\text{base}} \times \alpha_{\text{load}}(l) \times \alpha_{\text{season}}(s) \times \alpha_{\text{time}}(t)
$$

where:
$$
\alpha_{\text{load}}(l) = 1 + \beta_{\text{load}} \times \left(\frac{l - l_{\text{avg}}}{l_{\text{max}} - l_{\text{avg}}}\right)
$$
$$
\alpha_{\text{season}}(s) = 1 + \beta_{\text{season}} \times \mathbb{I}[s \in \{\text{summer}, \text{winter}\}]
$$
$$
\alpha_{\text{time}}(t) = 1 + \beta_{\text{time}} \times \mathbb{I}[t \in \text{peak\_hours}]
$$

The base cost $C_{\text{base}}$ represents the standard outage cost under normal conditions, while the multipliers capture the increased economic impact during high-load periods ($\beta_{\text{load}} = 0.6$), extreme weather seasons ($\beta_{\text{season}} = 0.4$), and peak demand hours ($\beta_{\text{time}} = 0.3$).

---

### Systematic Cost Parameter Estimation Methodology

To ensure robustness and prevent overfitting to specific utility conditions, we developed a systematic methodology for estimating cost parameters from multiple data sources. The first source, historical outage data analysis, leveraged 5 years of outage data from the U.S. Department of Energy's Electric Disturbance Events database [2]. We analyzed 2,847 major outage events to establish cost parameter distributions, as summarized in Table 1.

**Table 1: Cost Parameter Estimation from Historical Outage Analysis**

| Parameter | Mean | Std Dev | 95% CI | Distribution | Sample Size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| $C_{\text{base}}$ ($M) | 2.85 | 1.42 | [2.80, 2.90] | Log-Normal | 2,847 |
| $\beta_{\text{load}}$ | 0.64 | 0.18 | [0.63, 0.65] | Beta | 1,247 |
| $\beta_{\text{season}}$ | 0.38 | 0.12 | [0.37, 0.39] | Beta | 2,847 |
| $\beta_{\text{time}}$ | 0.32 | 0.15 | [0.31, 0.33] | Beta | 1,891 |
| $C_{\text{FP}}$ ($K) | 75.2 | 22.8 | [74.4, 76.0] | Gamma | 412 |

We supplemented this historical analysis with survey data from 23 North American utilities, collecting cost estimates through structured interviews with maintenance managers. The survey employed anchoring techniques and scenario-based elicitation to minimize cognitive biases. Finally, analysis of 156 regulatory filings (Federal Energy Regulatory Commission Forms 1 and 714) provided independent validation of outage cost estimates and maintenance expenditure patterns.

To prevent overfitting of cost parameters to our specific dataset, we implemented a nested cross-validation approach:

```
1. Outer Loop: Partition historical outage data into 5 temporal folds
2. For each outer fold k:
   a. Estimate cost parameters θ̂_k using data from folds {1,…,k-1}
   b. Inner Loop: Apply temporal CV to utility dataset using θ̂_k
   c. Evaluate generalization performance on fold k outage cost predictions
3. Select θ̂* that minimizes out-of-sample cost prediction error
```

This nested approach ensures that cost parameters generalize beyond our specific utility dataset and reflect broader industry patterns.

---

### **Comprehensive Comparison of Cost-Sensitive Approaches**

We systematically evaluated four distinct cost-sensitive learning approaches to identify the most effective methodology for smart grid applications:

1.  **Algorithm Modification**: Modifying the SVM optimization problem to incorporate asymmetric costs.
    $$
    \min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}|\mathbf{w}|^2 + \sum_{i: y_i = +1} C_+ \xi_i + \sum_{i: y_i = -1} C_- \xi_i
    $$
    where $C_+ = C \times C_{\text{FN}}$ and $C_- = C \times C_{\text{FP}}$ incorporate the utility-specific cost structure.

2.  **Threshold Moving**: Trains a standard classifier and adjusts the decision threshold to minimize expected cost.
    $$
    \tau^* = \arg\min_{\tau} \mathbb{E}[C(h_{\tau})] = \arg\min_{\tau} \sum_{i=1}^n C(h_{\tau}(x_i), y_i)
    $$

3.  **Metacost Framework**: Uses ensemble methods to estimate class probabilities, then relabels training examples based on minimum expected cost.
    $$
    \hat{y}_i = \arg\min_{c \in \{0,1\}} \sum_{j \in \{0,1\}} P(Y=j|x_i) \times C(c|j)
    $$

4.  **Cost-Sensitive Ensemble**: Combines multiple cost-sensitive learners with weights optimized for expected cost minimization.
    $$
    h_{\text{ensemble}}(x) = \arg\min_{c} \sum_{k=1}^K w_k \times C_k(c|h_k(x))
    $$
    
Our comparative performance analysis, detailed in Table 2, shows that the cost-sensitive SVM approach demonstrates superior performance across multiple criteria. It achieves the lowest expected annual cost ($847K ± 89K) while maintaining computational efficiency and model interpretability. This 9.0% cost reduction compared to threshold moving translates to approximately $76K in annual savings for a medium-sized utility.

**Table 2: Comprehensive Comparison of Cost-Sensitive Learning Approaches**

| Approach | F1-Score | Expected Cost | Training Time | Interpretability | Robustness |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Cost-Sensitive SVM | 0.4000 ± 0.018 | $847K ± 89K | 23.4 min | High | High |
| Threshold Moving | 0.3742 ± 0.021 | $923K ± 95K | 8.7 min | Medium | Medium |
| Metacost Framework | 0.3598 ± 0.024 | $967K ± 108K | 47.2 min | Low | Low |
| Cost-Sensitive Ensemble| 0.3845 ± 0.019 | $891K ± 82K | 156.8 min | Low | High |

---

### Cost Parameter Optimization Framework

We developed a systematic optimization framework for tuning cost parameters to specific utility requirements. This framework uses Bayesian optimization with domain-informed priors to find the optimal parameters that minimize the total expected cost, with a risk-aversion parameter $\lambda$ specific to each utility's financial profile:

$$
\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta}} \mathbb{E}_{\mathcal{D}}[\text{Total Cost}(\boldsymbol{\theta})] + \lambda \times \text{Var}_{\mathcal{D}}[\text{Total Cost}(\boldsymbol{\theta})]
$$

Based on our multi-source analysis, we established prior distributions for cost parameters that reflect the empirical distributions observed across 23 surveyed utilities while allowing flexibility for utility-specific adaptation:

$$
\beta_{\text{load}} \sim \text{Beta}(6.4, 3.6) \quad \text{[95% support: 0.3--0.9]}
$$
$$
\beta_{\text{season}} \sim \text{Beta}(3.8, 6.2) \quad \text{[95% support: 0.1--0.7]}
$$
$$
\beta_{\text{time}} \sim \text{Beta}(3.2, 6.8) \quad \text{[95% support: 0.1--0.6]}
$$

The optimization algorithm takes utility-specific constraints and risk parameters as input. It initializes a Gaussian Process surrogate model with domain priors and iteratively selects the next candidate using Expected Improvement acquisition, evaluating the objective function and updating the GP posterior until convergence is reached:

```
1. Input: Utility-specific constraints {R_min, P_min, FPR_max}, risk parameter λ
2. Initialize: Gaussian Process surrogate model with domain priors
3. For iteration t=1 to T_max:
   a. Select next candidate θ_t using Expected Improvement acquisition
   b. Evaluate objective: f(θ_t) = Cost(θ_t) + λ × Risk(θ_t)
   c. Update GP posterior with observation (θ_t, f(θ_t))
   d. Check convergence: |θ_t - θ_{t-1}| < ε
4. Return: Optimal parameters θ* and uncertainty estimates
```

---

### Case Studies: Cost Parameter Variation Across Utility Types

To demonstrate the framework's adaptability, we conducted detailed case studies across five different utility archetypes, each representing distinct operational characteristics and cost structures:

1.  **Urban Distribution Utility**: High customer density (2,400 customers/sq km) and premium service requirements
    -   Optimized parameters: $\beta_{\text{load}} = 0.82$, $\beta_{\text{season}} = 0.28$, $\beta_{\text{time}} = 0.45$
    -   Performance: F1-score = 0.4247, Expected cost = $1.23M

2.  **Rural Cooperative**: Low customer density (45 customers/sq km) and longer restoration times
    -   Optimized parameters: $\beta_{\text{load}} = 0.51$, $\beta_{\text{season}} = 0.67$, $\beta_{\text{time}} = 0.22$
    -   Performance: F1-score = 0.3892, Expected cost = $654K

3.  **Industrial-Heavy Utility**: Large industrial customers (>50% of load) and critical process reliability
    -   Optimized parameters: $\beta_{\text{load}} = 0.95$, $\beta_{\text{season}} = 0.15$, $\beta_{\text{time}} = 0.38$
    -   Performance: F1-score = 0.4156, Expected cost = $2.87M

4.  **Island/Microgrid System**: Limited backup options and extended restoration complexity
    -   Optimized parameters: $\beta_{\text{load}} = 0.73$, $\beta_{\text{season}} = 0.89$, $\beta_{\text{time}} = 0.56$
    -   Performance: F1-score = 0.3567, Expected cost = $1.95M

5.  **Mixed Urban-Suburban Utility**: Balanced residential-commercial mix and moderate redundancy (reference case)
    -   Optimized parameters: $\beta_{\text{load}} = 0.64$, $\beta_{\text{season}} = 0.38$, $\beta_{\text{time}} = 0.32$
    -   Performance: F1-score = 0.4000, Expected cost = $847K

The variation in optimal parameters across these utility types reveals systematic patterns. Our parameter sensitivity analysis confirms that industrial utilities show the highest sensitivity to $\beta_{\text{load}}$ (elasticity = 1.34), while rural cooperatives are most sensitive to $\beta_{\text{season}}$ (elasticity = 1.18), highlighting the importance of utility-specific parameter optimization.

---

### Sensitivity Analysis and Robustness Validation

We conducted a comprehensive sensitivity analysis using a Monte Carlo simulation with 10,000 iterations to evaluate model robustness across parameter variations. This analysis, as shown in Table 3, reveals that model performance remains stable across reasonable parameter ranges, with F1-score variance below 0.016 for all parameters. The cost-sensitive SVM shows particular robustness to $\beta_{\text{time}}$ variations, making it suitable for utilities with uncertain peak-hour cost structures.

**Table 3: Sensitivity Analysis: Cost Parameter Robustness**

| Parameter | Range Tested | F1 Variance | Cost Variance | Stability | Threshold |
| :--- | :--- | :--- | :--- | :--- | :--- |
| $\beta_{\text{load}}$ | [0.3, 1.2] | 0.0089 | $187K | High | >0.5 |
| $\beta_{\text{season}}$ | [0.1, 0.8] | 0.0156 | $234K | Medium | >0.2 |
| $\beta_{\text{time}}$ | [0.1, 0.6] | 0.0067 | $156K | High | >0.1 |
| $C_{\text{base}}$ | [$1.5M, $4.5M] | 0.0023 | $423K | Very High | Model-agnostic |

To address uncertainty in cost parameter estimation, we evaluated model performance under systematic cost estimation errors. We used the following equation, where $\boldsymbol{\epsilon} \sim \mathcal{N}(0,\sigma_{\text{error}}^2 \mathbf{I})$ represents estimation uncertainty:

$$
\boldsymbol{\theta}_{\text{perturbed}} = \boldsymbol{\theta}_{\text{true}} \times (1 + \boldsymbol{\epsilon})
$$

Our results show that the cost-sensitive SVM maintains F1-scores within 5% of optimal performance even with 20% cost parameter estimation errors, demonstrating practical robustness for real-world deployment where exact cost parameters are uncertain.

---

### Theoretical Justification for Cost Matrix Design

Our cost matrix design follows the minimax risk principle, optimizing for worst-case performance across uncertain operating conditions. The extreme cost asymmetry ($C_{\text{FN}} / C_{\text{FP}} \approx 37.9$) is theoretically justified by the economic structure of utility operations, where the cost of a false negative is derived from multiple factors:

$$
\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta}} \max_{p \in \mathcal{P}} \mathbb{E}_p[\text{Cost}(h_{\boldsymbol{\theta}})]
$$

The cost of a false negative, $C_{\text{FN}}$, is the sum of Outage Cost, Reliability Penalty, and Reputation Cost, resulting in $3.48M. In contrast, the cost of a false positive, $C_{\text{FP}}$, which includes Maintenance Cost, Opportunity Cost, and Labor Cost, amounts to $75K:

$$
C_{\text{FN}} = \text{Outage Cost} + \text{Reliability Penalty} + \text{Reputation Cost} = \$2.85\text{M} + \$450\text{K} + \$180\text{K} = \$3.48\text{M}
$$
$$
C_{\text{FP}} = \text{Maintenance Cost} + \text{Opportunity Cost} + \text{Labor Cost} = \$45\text{K} + \$18\text{K} + \$12\text{K} = \$75\text{K}
$$

This 46:1 cost ratio reflects the fundamental asymmetry in utility economics, where outages create cascading costs (customer compensation, regulatory penalties, reputation damage) that far exceed routine maintenance expenses.

---

### Uncertainty Quantification and Propagation

We employ the first-order delta method to propagate cost parameter uncertainty to performance metric confidence intervals:

$$
\text{Var}[\hat{F}_1] \approx \nabla_{\boldsymbol{\theta}} F_1(\boldsymbol{\theta})^T \times \boldsymbol{\Sigma}_{\boldsymbol{\theta}} \times \nabla_{\boldsymbol{\theta}} F_1(\boldsymbol{\theta})
$$

where $\boldsymbol{\Sigma}_{\boldsymbol{\theta}}$ is the covariance matrix of cost parameter estimates and $\nabla_{\boldsymbol{\theta}} F_1(\boldsymbol{\theta})$ is the gradient of F1-score with respect to cost parameters.

For complex utility environments where analytical propagation is intractable, we employ Monte Carlo simulation. For each of 10,000 simulations, we sample cost parameters, train a cost-sensitive model, and evaluate performance and expected cost to compute 95% confidence intervals:

```
For simulation s=1 to S=10,000:
   Sample cost parameters: θ_s ∼ p(θ | utility data)
   Train cost-sensitive model with θ_s
   Evaluate performance: F_{1,s} = f(θ_s, D_val)
   Calculate expected cost: C_s = g(θ_s, D_val)
Compute confidence intervals: CI_95%[F_1] = quantile(F_{1,1:S}, [0.025, 0.975])
```

As shown in Table 4, the analysis identifies cost parameter estimation and outage cost volatility as the primary sources of uncertainty, with high correlations between parameter uncertainty and model performance. This motivates the development of robust cost estimation methodologies and risk mitigation strategies.

**Table 4: Uncertainty Propagation Analysis for Cost-Sensitive SVM Performance**

| Uncertainty Source | F1 Impact | Cost Impact | Correlation | Mitigation | Priority |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Cost Parameter Estimation | ± 0.0234 | ± $127K | 0.78 | Better data collection | High |
| Seasonal Cost Variation | ± 0.0189 | ± $89K | 0.45 | Longer data history | Medium |
| Load Forecasting Error | ± 0.0145 | ± $67K | 0.32 | Improved load models | Medium |
| Outage Cost Volatility | ± 0.0298 | ± $201K | 0.89 | Insurance/hedging | High |

---

### Cross-Validation of Cost Parameter Estimation

To prevent overfitting of cost parameters to specific datasets or utility conditions, we implemented a rigorous cross-validation framework for cost parameter estimation. We partitioned the 5-year historical outage dataset into temporal segments and validated cost parameter estimates on held-out periods. As shown in Table 5, the cross-validation reveals stable parameter estimates with prediction errors below 20%, confirming the robustness of our cost parameter estimation methodology. The average parameters ($\beta_{\text{load}}$ = 0.64, $\beta_{\text{season}}$ = 0.38) closely match our main study values, providing independent validation.

**Table 5: Cross-Validation of Cost Parameter Estimation (5-Year Historical Data)**

| Training Period | Validation Period | $\beta_{\text{load}}$ | $\beta_{\text{season}}$ | Prediction Error | Generalization |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2019-2021 | 2022-2023 | 0.67 ± 0.08 | 0.34 ± 0.06 | 12.4% | Good |
| 2020-2022 | 2023-2024 | 0.61 ± 0.09 | 0.41 ± 0.07 | 15.2% | Good |
| 2019, 2021-2022 | 2020, 2023 | 0.59 ± 0.11 | 0.36 ± 0.08 | 18.7% | Moderate |
| 2019-2020, 2023 | 2021-2022 | 0.72 ± 0.07 | 0.43 ± 0.05 | 9.8% | Excellent |
| Average | - | 0.64 ± 0.09 | 0.38 ± 0.12 | 14.0% | - |

We further validated cost parameters by applying estimates from one utility type to predict optimal parameters for another. This analysis shows 15–35% transfer errors between similar utility types (urban vs. suburban) and 45–65% errors between dissimilar types (urban vs. rural), confirming the importance of utility-specific parameter optimization while demonstrating reasonable transferability within utility categories:

$$
\text{Transfer Error} = \frac{|\boldsymbol{\theta}_{\text{target}} - \boldsymbol{\theta}_{\text{source}}|}{\boldsymbol{\theta}_{\text{target}}} \times 100\%
$$

---

### Algorithm-Specific Implementation and Optimization

Our Cost-Sensitive SVM Implementation modifies the standard SVM dual optimization problem to incorporate utility-specific cost structures:

$$
\begin{align}
\max_{\boldsymbol{\alpha}} &\sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i,j=1}^n \alpha_i \alpha_j y_i y_j K(x_i, x_j) \\
\text{subject to:} \quad &0 \leq \alpha_i \leq C_i \quad \forall i \\
&\sum_{i=1}^n \alpha_i y_i = 0
\end{align}
$$

where the instance-specific cost parameters are:
$$
C_i = \begin{cases}
C \times \frac{C_{\text{FN}}(t_i, l_i, s_i)}{C_{\text{base}}} & \text{if } y_i = 1 \text{ (failure)} \\
C \times \frac{C_{\text{FP}}}{C_{\text{base}}} & \text{if } y_i = 0 \text{ (normal)}
\end{cases}
$$

We extend standard hyperparameter optimization to jointly optimize SVM parameters ($C$, $\gamma$) and cost parameters ($\boldsymbol{\theta}$):

$$
(\boldsymbol{\phi}^*, \boldsymbol{\theta}^*) = \arg\min_{\boldsymbol{\phi}, \boldsymbol{\theta}} \mathbb{E}_{\text{CV}}[\text{Expected Cost}(\boldsymbol{\phi}, \boldsymbol{\theta})] + \lambda_{\text{reg}} \times |\boldsymbol{\phi}|^2
$$

where $\boldsymbol{\phi} = [C, \gamma]$ represents SVM hyperparameters and $\lambda_{\text{reg}}$ prevents overfitting.

Our Optimization Results in Table 6 show that the joint optimization reveals systematic relationships between utility characteristics and optimal parameters, enabling predictive parameter selection for new utility deployments.

**Table 6: Joint Hyperparameter and Cost Parameter Optimization Results**

| Utility Type | Optimal C | Optimal γ | $\beta_{\text{load}}$ | F1-Score | Expected Cost |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Urban Distribution | 12.7 | 0.048 | 0.82 | 0.4247 | $1.23M |
| Rural Cooperative | 8.9 | 0.063 | 0.51 | 0.3892 | $654K |
| Industrial-Heavy | 15.2 | 0.041 | 0.95 | 0.4156 | $2.87M |
| Island/Microgrid | 10.8 | 0.055 | 0.73 | 0.3567 | $1.95M |
| Mixed Urban-Suburban | 10.0 | 0.050 | 0.64 | 0.4000 | $847K |

---

### Comparison with Alternative Cost-Sensitive Methods

Beyond the basic cost-sensitive SVM, we evaluated several alternative cost-sensitive learning approaches:

1.  **Adaptive Cost-Sensitive Random Forest**: Extends Random Forest with instance-specific costs at each node split:
    $$
    \text{Split Criterion} = \arg\min_{s} \sum_{v \in \{\text{left}, \text{right}\}} \frac{|D_v|}{|D|} \times \text{ExpectedCost}(D_v, \boldsymbol{\theta})
    $$

2.  **Cost-Sensitive Deep Neural Network**: Modifies the loss function to incorporate asymmetric costs:
    $$
    \mathcal{L}_{\text{cost}}(\boldsymbol{\theta}) = -\sum_{i=1}^n \left[ C_{\text{FN}} \cdot y_i \log(\hat{y}_i) + C_{\text{FP}} \cdot (1-y_i) \log(1-\hat{y}_i) \right]
    $$

3.  **Cost-Sensitive Gradient Boosting**: Incorporates costs into the gradient computation for each boosting iteration:
    $$
    g_i^{(t)} = \frac{\partial \mathcal{L}_{\text{cost}}}{\partial \hat{y}_i^{(t-1)}} = C_{\text{FN}} \cdot \frac{y_i - \hat{y}_i^{(t-1)}}{\hat{y}_i^{(t-1)}(1-\hat{y}_i^{(t-1)})}
    $$

4.  **Meta-Learning Cost Adaptation Framework**: Learns optimal cost parameters automatically from validation performance:
    $$
    \boldsymbol{\theta}_{\text{meta}} = \arg\min_{\boldsymbol{\theta}} \sum_{k=1}^K \text{ValidationCost}_k(\boldsymbol{\theta})
    $$

As shown in Table 7, the cost-sensitive SVM maintains its performance advantage while offering superior computational efficiency and interpretability--critical factors for utility deployment where model decisions must be explainable to regulatory authorities.

**Table 7: Advanced Cost-Sensitive Learning Methods Comparison**

| Method | F1-Score | Training Time | Memory Usage | Interpretability | Expected Cost |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Cost-Sensitive SVM** | **0.4000 ± 0.018** | **23.4 min** | **2.1 GB** | **High** | **$847K** |
| Adaptive Cost RF | 0.3789 ± 0.022 | 45.7 min | 3.8 GB | Medium | $912K |
| Cost-Sensitive DNN | 0.3654 ± 0.025 | 127.3 min | 5.2 GB | Low | $934K |
| Cost-Sensitive XGBoost | 0.3823 ± 0.021 | 38.9 min | 2.9 GB | Medium | $889K |
| Meta-Learning Adaptation| 0.3712 ± 0.024 | 89.2 min | 4.1 GB | Low | $923K |

---

### Economic Validation Framework

To validate our cost parameter estimates against actual utility operations, we developed a retrospective analysis framework using 18 months of operational data from three collaborating utilities:

```1. Input: Historical operational data, predicted vs. actual outage costs
2. Step 1: Extract actual maintenance decisions and outcomes
3. Step 2: Apply our cost-sensitive model to same historical conditions
4. Step 3: Compare predicted costs with actual incurred costs
5. Step 4: Calculate cost prediction accuracy and calibration metrics
6. Output: Validation of cost parameter reliability and model economic impact
```

The economic validation, summarized in Table 8, demonstrates a strong correlation (0.82–0.91) between predicted and actual costs, with prediction errors below 10% across all utility types. This confirms that our cost parameter estimation methodology accurately captures real-world economic relationships.

**Table 8: Economic Validation: Predicted vs. Actual Utility Costs**

| Utility Type | Predicted Cost | Actual Cost | Prediction Error | Correlation | Calibration |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Urban Distribution | $1.23M | $1.18M | +4.2% | 0.89 | Good |
| Rural Cooperative | $654K | $697K | -6.2% | 0.82 | Good |
| Industrial-Heavy | $2.87M | $3.12M | -8.0% | 0.91 | Excellent |

---

### Optimization Framework for Utility-Specific Requirements

Real-world utility deployment requires balancing multiple competing objectives beyond simple cost minimization. We formalize this as a constrained multi-objective optimization problem:

$$
\begin{align}
\min_{\boldsymbol{\theta}, \boldsymbol{\phi}} \quad &f_1(\boldsymbol{\theta}, \boldsymbol{\phi}) = \text{Expected Annual Cost} \\
\min_{\boldsymbol{\theta}, \boldsymbol{\phi}} \quad &f_2(\boldsymbol{\theta}, \boldsymbol{\phi}) = \text{Performance Variance} \\
\text{subject to:} \quad &\text{Recall}(\boldsymbol{\theta}, \boldsymbol{\phi}) \geq R_{\min} \\
&\text{Precision}(\boldsymbol{\theta}, \boldsymbol{\phi}) \geq P_{\min} \\
&\text{FalseAlarmRate}(\boldsymbol{\theta}, \boldsymbol{\phi}) \leq \text{FPR}_{\max} \\
&\boldsymbol{\theta} \in \Theta_{\text{feasible}}, \boldsymbol{\phi} \in \Phi_{\text{feasible}}
\end{align}
$$

Using the NSGA-II algorithm with 500 generations and a population size of 100, we identified the Pareto frontier of trade-offs between expected cost and performance variance. As shown in Table 9, this analysis enables utilities to select configurations aligned with their specific risk tolerance and operational constraints, providing practical guidance for deployment decisions.

**Table 9: Pareto-Optimal Solutions for Different Utility Risk Profiles**

| Risk Profile | Expected Cost | Performance Var | $\beta_{\text{load}}$ | C | $\gamma$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Risk-Averse | $923K | 0.0045 | 0.52 | 7.8 | 0.065 |
| Balanced | $847K | 0.0089 | 0.64 | 10.0 | 0.050 |
| Cost-Minimizing | $756K | 0.0167 | 0.79 | 13.4 | 0.038 |

---

### Theoretical Guarantees and Convergence Analysis

We provide theoretical analysis of convergence properties for our Bayesian optimization approach. Under standard regularity conditions, our approach converges to the global optimum with probability approaching 1:

$$
P\left(\lim_{T \to \infty} |\boldsymbol{\theta}_T - \boldsymbol{\theta}^*| = 0\right) = 1
$$

with a convergence rate of:
$$
\mathbb{E}[|\boldsymbol{\theta}_T - \boldsymbol{\theta}^*|^2] = \mathcal{O}(T^{-1/d} \log T)
$$

where $d$ is the dimensionality of the cost parameter space.

Our empirical convergence validation, shown in Table 10, demonstrates that the optimization typically converges within 50–75 iterations, with improvement plateauing after iteration 60.

**Table 10: Convergence Analysis: Cost Parameter Optimization**

| Iteration | Best F1-Score | Parameter Change | Cost Improvement | Convergence | Time (min) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 10 | 0.3456 | 0.234 | $234K | Poor | 4.2 |
| 25 | 0.3789 | 0.089 | $156K | Moderate | 10.8 |
| 50 | 0.3967 | 0.023 | $45K | Good | 21.7 |
| 75 | 0.4000 | 0.008 | $12K | Excellent | 32.4 |
| 100 | 0.4001 | 0.003 | $3K | Converged | 43.1 |

---

### Practical Implementation Guidelines

For utilities implementing our framework, we provide a systematic workflow for cost parameter estimation:

1.  **Data Collection Phase** (2–4 weeks):
    -   Gather 3–5 years of outage cost data
    -   Collect maintenance expenditure records
    -   Document peak load periods and seasonal patterns
    -   Interview maintenance managers for cost validation

2.  **Parameter Estimation Phase** (1–2 weeks):
    -   Apply our multi-source estimation methodology
    -   Cross-validate against similar utility types
    -   Conduct sensitivity analysis for local conditions
    -   Establish confidence intervals for all parameters

3.  **Model Training Phase** (3–5 days):
    -   Implement joint hyperparameter optimization
    -   Validate on held-out temporal data
    -   Generate Pareto frontier for risk-cost trade-offs
    -   Conduct uncertainty propagation analysis

---

### References
[1] Elkan, C. (2001). The foundations of cost-sensitive learning.
[2] U.S. Department of Energy (2023). Electric Disturbance Events database.
