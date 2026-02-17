window.DEMLAI_SURVEY_DATA = {
    "ml":  [
               {
                   "subdomain":  "Core Mathematics",
                   "skill":  "Linear Algebra",
                   "levels":  [
                                  "Has heard of concepts like matrices or vectors but lacks working understanding. Cannot yet apply them to ML.",
                                  "Understands basic matrix operations (addition, multiplication, transposition) and concepts like dot products. Can follow tutorials using matrix math in ML libraries.",
                                  "Applies eigenvectors, SVD, and vector spaces in practice (e.g. PCA, embeddings). Can explain why these methods are used in ML.",
                                  "Understands and implements algorithms that rely heavily on linear algebra. Can debug dimensionality errors and optimize computation using matrix properties.",
                                  "Teaches and mentors others in applied linear algebra. Can design novel models or transformations informed by deep understanding of vector space geometry and tensor algebra."
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Calculus",
                   "levels":  [
                                  "Aware that ML involves derivatives but does not understand how they are used.",
                                  "Understands the role of derivatives and gradients in training (e.g. gradient descent). Can compute simple partial derivatives.",
                                  "Applies multivariable calculus in backpropagation. Understands Jacobians, Hessians, and their relevance to optimization.",
                                  "Interprets and manipulates higher-order derivatives in custom training loops or research settings. Uses calculus for model introspection and regularization.",
                                  "Develops new training techniques or architectures informed by calculus. Guides others in understanding theoretical underpinnings of differentiable models."
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Probability \u0026 Statistics",
                   "levels":  [
                                  "Recognizes that ML involves chance and randomness but cannot define probability distributions or statistical terms.",
                                  "Understands basic distributions (e.g. normal, binomial), descriptive stats (mean, variance), and basic hypothesis testing.",
                                  "Uses Bayes\u0027 theorem, likelihoods, priors, and confidence intervals in applied ML. Understands randomness in model behavior.",
                                  "Builds and evaluates probabilistic models (e.g. Naive Bayes, Bayesian networks). Understands uncertainty quantification and statistical inference.",
                                  "Leads probabilistic modeling efforts. Trains others in statistical thinking and probabilistic programming. Challenges and refines assumptions behind statistical methods in ML."
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Optimization Techniques",
                   "levels":  [
                                  "Has heard of optimization in ML (e.g., loss minimization), but cannot explain how it works.",
                                  "Understands gradient descent and basic tuning (e.g., learning rate, batch size). Can experiment with optimizers like SGD and Adam.",
                                  "Applies advanced optimizers (e.g. RMSProp, L-BFGS), regularization techniques, and understands tradeoffs. Can interpret loss surfaces and convergence behavior.",
                                  "Designs custom loss functions and optimization strategies. Understands optimization challenges like saddle points, vanishing gradients, and convergence instability.",
                                  "Innovates new optimization approaches or techniques. Educates teams on convergence theory and empirical tuning best practices. Bridges theory and production performance."
                              ]
               },
               {
                   "subdomain":  "Data Preparation \u0026 Feature Engineering",
                   "skill":  "Data Cleaning",
                   "levels":  [
                                  "Aware that data often has issues like missing values or duplicates but lacks understanding of how to handle them.",
                                  "Can identify and correct simple issues (e.g. missing data, outliers, formatting inconsistencies) using tools like pandas or SQL.",
                                  "Understands and applies domain-informed strategies for imputing, filtering, and validating data. Tracks data quality metrics.",
                                  "Designs scalable, repeatable cleaning pipelines. Anticipates downstream effects of data quality on models.",
                                  "Leads data hygiene practices across teams. Establishes standards, mentors others, and contributes to tooling or policy for data integrity in ML."
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Feature Extraction",
                   "levels":  [
                                  "Understands that raw data must be transformed into usable input, but unsure how or why.",
                                  "Can use built-in methods to extract basic features (e.g., text n-grams, image pixels, timestamps).",
                                  "Designs custom features from structured and unstructured data (e.g., aggregation, parsing, embeddings). Can justify feature choices.",
                                  "Builds reusable feature generation frameworks. Balances complexity vs. model interpretability.",
                                  "Pioneers novel feature representations and extraction methods. Guides teams in feature engineering for new problem domains."
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Feature Selection",
                   "levels":  [
                                  "Aware that not all features are useful but lacks ability to identify or remove irrelevant ones.",
                                  "Uses basic filtering techniques (e.g., correlation thresholding, variance) to reduce features.",
                                  "Applies techniques like recursive feature elimination, regularization-based selection, and information gain.",
                                  "Evaluates and automates feature selection strategies to prevent overfitting and boost generalization.",
                                  "Develops and validates custom feature scoring methods. Advises teams on balancing performance, interpretability, and compute cost."
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Normalization \u0026 Encoding",
                   "levels":  [
                                  "Knows that data needs to be formatted but is unsure how (e.g., categorical values, scales).",
                                  "Applies techniques like min-max scaling, z-score standardization, one-hot encoding.",
                                  "Selects appropriate scaling and encoding methods based on model and data type. Handles unseen categories and leakage issues.",
                                  "Designs robust preprocessing pipelines (e.g., via sklearn Pipeline, Spark transformers). Balances numerical stability with model assumptions.",
                                  "Educates others on best practices. Develops or adapts encoding and scaling strategies for novel or large-scale data contexts."
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Dimensionality Reduction",
                   "levels":  [
                                  "Aware that too many features can hurt model performance but unfamiliar with reduction techniques.",
                                  "Can apply basic techniques like PCA with default settings using libraries.",
                                  "Understands and selects methods like PCA, t-SNE, UMAP, or autoencoders based on data and task. Can interpret reduced spaces.",
                                  "Uses dimensionality reduction for interpretability, visualization, or compression. Understands trade-offs and implications.",
                                  "Designs or evaluates new dimensionality reduction approaches. Mentors others in using these techniques for feature selection, exploration, and model optimization."
                              ]
               },
               {
                   "subdomain":  "Model Development",
                   "skill":  "Regression Modeling",
                   "levels":  [
                                  "Recognizes that regression predicts numeric outcomes but has no hands-on experience.",
                                  "Can fit and evaluate linear regression models using libraries like scikit-learn. Understands concepts like coefficients and residuals.",
                                  "Uses advanced regression methods (e.g. ridge, lasso, polynomial). Understands assumptions, multicollinearity, and interprets metrics like RMSE and R�.",
                                  "Designs regression pipelines, handles feature interaction, and addresses overfitting. Explains results to technical and non-technical audiences.",
                                  "Develops novel regression approaches or tunes architectures for complex tasks. Trains others and establishes modeling standards for regression problems."
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Classification Modeling",
                   "levels":  [
                                  "Aware that classification predicts categories. No real modeling experience.",
                                  "Can train simple classifiers (e.g. logistic regression, decision tree) and interpret outputs like predicted classes.",
                                  "Applies and compares multiple classifiers (e.g. random forest, SVM, XGBoost). Understands class imbalance, precision/recall tradeoffs, and ROC/AUC.",
                                  "Optimizes pipelines, calibrates classifiers, and handles edge cases (e.g., multilabel, ordinal). Integrates model explainability tools.",
                                  "Innovates on classification techniques for new domains. Establishes classifier evaluation frameworks and mentors others on robust, fair model design."
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Clustering Techniques",
                   "levels":  [
                                  "Understands that clustering groups data but doesn\u0027t know how it works or why it�s useful.",
                                  "Uses simple clustering algorithms (e.g. K-means) to explore structure in data. Tunes basic parameters like number of clusters.",
                                  "Applies a range of methods (e.g. DBSCAN, hierarchical, GMM). Evaluates using silhouette score, inertia, or domain-informed labeling.",
                                  "Selects clustering methods based on data shape, scale, and density. Builds interpretable, repeatable workflows for unsupervised tasks.",
                                  "Designs custom similarity metrics or clustering algorithms. Leads clustering strategy for exploratory or production use, mentoring others in unsupervised ML."
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Deep Learning (MLPs)",
                   "levels":  [
                                  "Has heard of neural networks. No practical experience building or using one.",
                                  "Can train a basic feedforward network (MLP) for a supervised task using frameworks like TensorFlow or PyTorch. Understands the concept of layers and activation functions.",
                                  "Designs MLP architectures, tunes hyperparameters, and applies best practices (e.g. dropout, batch norm). Understands backpropagation and vanishing gradients.",
                                  "Tailors architectures to domains (e.g., tabular, text). Integrates MLPs into larger ML systems. Troubleshoots training instability and performance bottlenecks.",
                                  "Advances MLP usage in novel contexts or architectures. Mentors others and contributes to reusable components or learning frameworks."
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Reinforcement Learning",
                   "levels":  [
                                  "Aware that RL involves agents and environments but lacks working knowledge or intuition.",
                                  "Can run simple RL examples (e.g. cart-pole with Q-learning or DQN). Understands the reward/policy/value loop.",
                                  "Implements RL algorithms, tunes reward functions, and applies RL to basic problems. Understands exploration vs. exploitation and convergence.",
                                  "Applies deep RL in complex environments (e.g., continuous action spaces, simulators). Evaluates stability, sample efficiency, and policy performance.",
                                  "Researches or innovates in RL algorithms. Leads experimentation with RL in real-world systems and mentors others on long-horizon learning strategies."
                              ]
               },
               {
                   "subdomain":  "Model Evaluation \u0026 Diagnostics",
                   "skill":  "Evaluation Metrics",
                   "levels":  [
                                  "Recognizes that models need to be evaluated but is unfamiliar with specific metrics.",
                                  "Can compute basic metrics like accuracy, precision, recall, or RMSE using libraries.",
                                  "Understands when and why to use different metrics based on problem type (e.g., classification vs regression).",
                                  "Selects and justifies appropriate metrics for domain-specific tasks. Incorporates multiple metrics for robust evaluation.",
                                  "Defines new or composite metrics, educates teams on best practices, and sets standards for evaluation across projects."
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Cross-Validation",
                   "levels":  [
                                  "Aware that cross-validation helps improve generalization but hasn�t applied it.",
                                  "Can implement simple k-fold or train/test split cross-validation using libraries.",
                                  "Understands different CV techniques (e.g., stratified, LOOCV) and applies them appropriately.",
                                  "Integrates CV into pipelines, handles data leakage concerns, and interprets variance in results.",
                                  "Designs robust evaluation schemes for complex or time-series data. Coaches others on validation design and pitfalls."
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Error Analysis",
                   "levels":  [
                                  "Knows that model errors exist but lacks a framework to analyze them.",
                                  "Can identify misclassified or high-error examples manually.",
                                  "Analyzes patterns in errors to find model weaknesses or data quality issues.",
                                  "Structures iterative error analysis loops to inform data cleaning, feature updates, or model redesigns.",
                                  "Leads teams in systematic error analysis, linking it to business KPIs and continuous improvement processes."
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Model Bias \u0026 Fairness",
                   "levels":  [
                                  "Aware that models can be biased but lacks practical understanding.",
                                  "Can identify basic types of bias (e.g., class imbalance) and compute group-based metrics like demographic parity.",
                                  "Understands causes of algorithmic bias and interprets fairness metrics within context.",
                                  "Applies fairness-aware training, post-processing, or auditing techniques. Navigates ethical tradeoffs.",
                                  "Sets organizational standards for fairness. Leads efforts on ethical ML and regulatory compliance. Educates others on bias mitigation."
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Calibration",
                   "levels":  [
                                  "Has heard of model calibration but doesn�t know what it means or when it\u0027s used.",
                                  "Can generate and interpret calibration plots or reliability diagrams.",
                                  "Understands how to assess and improve calibration using techniques like Platt scaling or isotonic regression.",
                                  "Applies calibration in production workflows (e.g., probability thresholds, risk scoring) and evaluates model confidence systematically.",
                                  "Develops calibration-aware models, instructs others on uncertainty quantification, and innovates in trust-building techniques."
                              ]
               },
               {
                   "subdomain":  "Model Optimization \u0026 Tuning",
                   "skill":  "Hyperparameter Tuning",
                   "levels":  [
                                  "Aware that models have hyperparameters but doesn�t know what or how to tune them.",
                                  "Can modify hyperparameters manually and run simple grid searches.",
                                  "Understands the impact of key hyperparameters and applies grid/random search effectively.",
                                  "Uses automated search strategies (e.g., Bayesian, Optuna), interprets tuning results critically.",
                                  "Designs advanced tuning strategies, custom search spaces, and optimizes tuning pipelines across projects."
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Regularization",
                   "levels":  [
                                  "Has heard of overfitting and regularization but doesn�t know how they work.",
                                  "Can apply L1/L2 regularization using library defaults.",
                                  "Understands how regularization affects model complexity and generalization; can tune penalties effectively.",
                                  "Chooses and configures regularization strategies (e.g., dropout, elastic net) based on model and dataset.",
                                  "Develops regularization schemes for novel models or domains; teaches others how to balance bias-variance trade-offs."
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Model Selection",
                   "levels":  [
                                  "Recognizes that different models exist but doesn�t know how to choose between them.",
                                  "Can compare models using simple metrics (e.g., accuracy, RMSE).",
                                  "Uses evaluation metrics, validation results, and task requirements to select suitable models.",
                                  "Balances interpretability, performance, and deployment needs when selecting models.",
                                  "Leads architecture decisions across projects, drives innovation by proposing new model classes or paradigms."
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Ensembling",
                   "levels":  [
                                  "Is vaguely aware that combining models can improve performance.",
                                  "Can implement basic ensembles like majority voting or averaging.",
                                  "Understands bagging, boosting, and stacking; selects ensemble methods based on goals and data.",
                                  "Designs and tunes ensembles that improve robustness and generalization; integrates ensemble workflows into pipelines.",
                                  "Develops ensemble strategies for critical applications; teaches ensembling trade-offs and failure modes to others."
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Imbalance Handling",
                   "levels":  [
                                  "Aware that class imbalance can hurt model performance but doesn�t know what to do.",
                                  "Can apply basic techniques like oversampling or class weighting.",
                                  "Diagnoses imbalance problems using metrics and distributions; applies SMOTE, stratification, or cost-sensitive training.",
                                  "Builds custom imbalance strategies depending on business impact and context; integrates mitigation in training pipelines.",
                                  "Drives fairness and imbalance resolution policies; mentors others on building resilient models for rare-event or skewed-data problems."
                              ]
               },
               {
                   "subdomain":  "Productionalization, Operation, \u0026 Model Lifecycle",
                   "skill":  "Model Serialization",
                   "levels":  [
                                  "Understands that trained models can be saved but not how or why.",
                                  "Can save and load models using built-in library methods (e.g., joblib, pickle, torch.save).",
                                  "Applies serialization formats appropriate to deployment context and handles versioning.",
                                  "Builds reusable serialization modules; considers model portability, compatibility, and security.",
                                  "Defines organization-wide serialization standards, tools, and compatibility layers; trains others on best practices."
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Model Deployment",
                   "levels":  [
                                  "Knows models must be deployed to be used, but unfamiliar with methods.",
                                  "Can deploy a model locally or via a simple API (e.g., Flask or FastAPI).",
                                  "Uses model serving frameworks (e.g., MLflow, TorchServe, SageMaker) and integrates with backend systems.",
                                  "Designs scalable, resilient deployment strategies (e.g., containerized microservices, serverless).",
                                  "Leads architecture for ML platforms; standardizes deployment pipelines; mentors teams on production ML practices."
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "CI/CD for ML",
                   "levels":  [
                                  "Aware that automation is used in software, but unsure how it applies to ML.",
                                  "Can trigger retraining or tests manually using scripts or notebooks.",
                                  "Implements ML-specific CI/CD flows (e.g., automated retraining, testing, linting, model validation).",
                                  "Integrates CI/CD into MLOps pipelines with branch-based triggers, model registry, and approvals.",
                                  "Architects CI/CD systems for reproducibility, auditability, and collaboration across ML lifecycle. Guides others in DevOps/MLOps fusion."
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Model Monitoring",
                   "levels":  [
                                  "Understands that deployed models may drift or fail but doesn�t monitor them.",
                                  "Can log basic metrics and usage statistics post-deployment.",
                                  "Monitors performance, data drift, and service health using alerts and dashboards.",
                                  "Builds automated feedback loops, integrates observability tooling (e.g., Prometheus, Grafana, Evidently).",
                                  "Leads monitoring architecture across teams; formalizes accountability frameworks (e.g., SLA/SLOs) and incident response for ML."
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Retraining Strategy",
                   "levels":  [
                                  "Knows models need occasional updates but has no plan for it.",
                                  "Can manually retrain a model when performance declines.",
                                  "Sets up retraining triggers (e.g., periodic, drift-based) and maintains retraining scripts.",
                                  "Automates retraining with validation gates, rollback options, and metadata tracking.",
                                  "Designs adaptive learning systems, owns lifecycle governance, and coaches teams on sustainable model evolution."
                              ]
               },
               {
                   "subdomain":  "Programming \u0026 Libraries for ML",
                   "skill":  "Python Programming",
                   "levels":  [
                                  "Can read simple Python code; struggles to write or debug.",
                                  "Writes basic scripts, uses functions, and controls flow for data tasks.",
                                  "Writes modular, readable code with exception handling and logging; uses libraries effectively.",
                                  "Follows software engineering practices (e.g., unit testing, packaging, typing) for ML pipelines.",
                                  "Develops shared codebases and internal packages; mentors others on performance, design, and maintainability."
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "NumPy \u0026 Pandas",
                   "levels":  [
                                  "Can load data using Pandas but struggles with transformations or vectorization.",
                                  "Performs basic wrangling, indexing, and aggregations; uses NumPy for simple math.",
                                  "Uses advanced operations (e.g., broadcasting, groupby, joins, reshaping) effectively in pipelines.",
                                  "Optimizes code using vectorized operations, memory management, and chunking.",
                                  "Trains others in high-performance numerical computing; contributes reusable data utilities or wrappers."
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "scikit-learn",
                   "levels":  [
                                  "Aware of its use in ML, but unfamiliar with APIs.",
                                  "Trains models using .fit(), evaluates with .score(), and applies transformations.",
                                  "Uses pipelines, model selection, preprocessing, and cross-validation workflows effectively.",
                                  "Extends estimators, writes custom transformers, integrates with production workflows.",
                                  "Advocates for standardization; mentors others in scalable, composable scikit-learn design patterns."
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Deep Learning Frameworks",
                   "levels":  [
                                  "Knows names (e.g., PyTorch, TensorFlow) but unsure how to use them.",
                                  "Can build simple MLPs or CNNs using a high-level API (e.g., nn.Sequential, Keras).",
                                  "Writes custom models and training loops; uses GPU acceleration and data loaders.",
                                  "Optimizes training with callbacks, scheduling, and mixed-precision; integrates with production tooling.",
                                  "Builds reusable frameworks, mentors on trade-offs across DL libraries, and contributes to shared model infrastructure."
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Notebook prototyping",
                   "levels":  [
                                  "Uses notebooks sporadically with poor structure or output management.",
                                  "Runs linear prototypes with inline plots and print debugging.",
                                  "Uses versioned, well-commented notebooks for reproducible experiments.",
                                  "Structures notebooks with parameterization, exports outputs, and integrates with scripts or pipelines.",
                                  "Promotes best practices (e.g., modularization, reproducibility); maintains shared templates and reviewable notebooks."
                              ]
               },
               {
                   "subdomain":  "ML Systems \u0026 Scalability",
                   "skill":  "Mini-batch \u0026 GPU Training",
                   "levels":  [
                                  "Understands that ML training can be slow and GPUs can help.",
                                  "Can configure mini-batch sizes and enable GPU use in a framework like PyTorch or TensorFlow.",
                                  "Tunes batch sizes, leverages GPU memory efficiently, uses data loaders and augmentation pipelines.",
                                  "Profiles and optimizes GPU throughput, resolves bottlenecks, and uses multiple GPUs effectively.",
                                  "Designs scalable training routines, selects appropriate hardware, and teaches teams efficient deep learning practices."
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Distributed Training",
                   "levels":  [
                                  "Knows large models can be trained across machines but lacks practical experience.",
                                  "Uses high-level APIs (e.g., DataParallel, Accelerate, tf.distribute) for multi-GPU or node setups.",
                                  "Implements training across nodes using Horovod, PyTorch DDP, or custom RPC backends.",
                                  "Balances load, manages communication overhead, and ensures fault tolerance in large-scale jobs.",
                                  "Architects and oversees distributed training infrastructure across projects; mentors teams on cost/performance trade-offs."
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Pipeline Integration",
                   "levels":  [
                                  "Understands that ML is part of a broader system but sees it in isolation.",
                                  "Can save/load models and integrate them into basic workflows via scripts or APIs.",
                                  "Builds training/inference steps as modular components in orchestrated workflows (e.g., Airflow, MLflow, SageMaker Pipelines).",
                                  "Connects ML components to upstream/downstream systems (e.g., ETL, alerts, APIs) and handles dependencies.",
                                  "Leads platform integration strategy; standardizes ML pipeline design; mentors teams on resilient, maintainable workflows."
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Resource Optimization",
                   "levels":  [
                                  "Aware that compute is expensive but does not manage usage.",
                                  "Minimizes memory or disk use manually; uses smaller batch sizes or simpler models.",
                                  "Applies profiling tools, quantization, pruning, or model distillation techniques.",
                                  "Automates resource-efficient workflows, balances cost-performance, and aligns with infrastructure constraints.",
                                  "Develops resource budgeting frameworks; drives culture of efficient computing and trains others to optimize at scale."
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Serving Infrastructure",
                   "levels":  [
                                  "Knows models can be deployed, but not how they are served in production.",
                                  "Deploys models using basic APIs or frameworks like Flask, FastAPI, or Streamlit.",
                                  "Uses model serving solutions (e.g., TorchServe, TensorFlow Serving, BentoML) and handles REST/gRPC endpoints.",
                                  "Designs scalable and fault-tolerant serving systems with A/B testing, canary deployments, and autoscaling.",
                                  "Leads ML platform infrastructure; defines organization-wide best practices for serving and maintains critical production deployments."
                              ]
               },
               {
                   "subdomain":  "Use Cases \u0026 Application Design",
                   "skill":  "Problem Framing",
                   "levels":  [
                                  "Struggles to distinguish between a data task and a business problem.",
                                  "Identifies if a problem is classification, regression, or clustering; starts defining goals.",
                                  "Translates business needs into ML problems with clearly scoped objectives and constraints.",
                                  "Frames ambiguous problems into tractable ML tasks, including evaluation criteria and feasibility.",
                                  "Guides cross-functional teams in framing strategic ML initiatives; mentors others in problem scoping under uncertainty."
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Business Metrics Mapping",
                   "levels":  [
                                  "Unaware of the relationship between model outputs and business value.",
                                  "Associates common metrics (e.g., accuracy, MSE) with product impact at a basic level.",
                                  "Aligns model metrics to business KPIs, selects thresholds based on cost-benefit trade-offs.",
                                  "Designs custom metrics reflecting real-world performance (e.g., churn saved, revenue impact).",
                                  "Advises stakeholders on long-term metric strategies; trains teams on goal-oriented metric design."
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Use Case Pioritization",
                   "levels":  [
                                  "Treats all ML opportunities as equal; lacks context for decision-making.",
                                  "Understands basic factors like data availability and implementation difficulty.",
                                  "Prioritizes ML use cases based on impact, feasibility, risk, and resourcing.",
                                  "Leads cross-team roadmapping; balances short-term wins with long-term investments.",
                                  "Builds ML portfolio strategies; mentors product and engineering teams on data-driven prioritization frameworks."
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Ethical Impact Analysis",
                   "levels":  [
                                  "Unaware of ML�s ethical or social implications.",
                                  "Acknowledges possible risks (e.g., bias, privacy) but doesn�t analyze them.",
                                  "Identifies potential ethical concerns and designs mitigation strategies (e.g., anonymization, fairness audits).",
                                  "Collaborates on cross-functional reviews, supports explainability, and integrates compliance needs.",
                                  "Champions responsible AI practices; develops frameworks and training for ethical ML design across the organization."
                              ]
               },
               {
                   "subdomain":  "ML Project Management",
                   "skill":  "Identifying ML Project Conditions",
                   "levels":  [
                                  "Struggles to determine if ML is appropriate for a problem.",
                                  "Recognizes when labeled data and predictive needs suggest an ML approach.",
                                  "Evaluates technical feasibility (data volume, quality), business relevance, and risks.",
                                  "Performs thorough opportunity assessments including legal, ethical, and stakeholder alignment.",
                                  "Leads organization-wide intake for ML initiatives; trains teams to assess readiness, risk, and opportunity systematically."
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Project Planning \u0026 Estimation",
                   "levels":  [
                                  "Unfamiliar with timelines or resources needed for ML tasks.",
                                  "Can list stages (e.g., data collection, training, deployment) but struggles with accurate estimates.",
                                  "Builds realistic timelines including iteration cycles, data dependencies, and handoff points.",
                                  "Manages risk buffers, coordinates with multiple stakeholders, and adapts plans as data evolves.",
                                  "Leads ML portfolio management; institutionalizes planning practices for high-impact, reliable delivery."
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Reporting Outcomes",
                   "levels":  [
                                  "Focuses only on technical metrics (e.g., accuracy); struggles to communicate value.",
                                  "Prepares basic reports with performance plots and summary metrics.",
                                  "Communicates results in stakeholder language, connects to KPIs and decisions.",
                                  "Builds dashboards, presents trade-offs and uncertainties clearly, and guides decision-making.",
                                  "Coaches teams in outcome storytelling; advises leadership on results interpretation and strategic implications."
                              ]
               }
           ],
    "de":  [
               {
                   "subdomain":  "Data Modeling \u0026 Querying",
                   "skill":  "SQL",
                   "levels":  [
                                  "Understand SQL syntax and can write basic SELECT statements",
                                  "Use JOINs, aggregations, and nested queries",
                                  "Optimize queries with indexes and analyze query plans",
                                  "Refactor complex queries for maintainability and performance",
                                  "Design SQL standards and lead query performance tuning at scale"
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Dimensional Modeling",
                   "levels":  [
                                  "Identify dimensions and facts in a dataset",
                                  "Build basic star schemas in BI tools",
                                  "Design robust schemas for reporting across domains",
                                  "Normalize/denormalize models based on use cases and access patterns",
                                  "Architect enterprise-wide dimensional models and standards"
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Medallion Modeling",
                   "levels":  [
                                  "Can land raw, immutable data in a dedicated ingestion layer and retrieve specific batches by timestamp or file name",
                                  "Cleanses and normalizes data incrementally into a refined layer, enforcing schema and basic data?quality rules",
                                  "Integrates and enriches datasets into a conformed analytical layer, resolving keys, managing history, and documenting lineage",
                                  "Optimizes performance, cost, and reliability end?to?end with tuned storage formats, automated tests, and proactive monitoring",
                                  "Architects metadata?driven, policy?governed Medallion platforms at enterprise scale and mentors teams on reusable best practices"
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "BCNF, Normalization Forms\n(1NF - 5NF)",
                   "levels":  [
                                  "Identifies repeating groups and converts unstructured tables to First Normal Form (1NF) by enforcing atomic columns and unique row identifiers",
                                  "Eliminates partial and transitive dependencies to achieve Second and Third Normal Forms (2NF,?3NF), defining clear primary�foreign?key relationships",
                                  "Decomposes schemas into Boyce?Codd and Fourth Normal Forms (BCNF,?4NF), resolving anomaly?causing functional and multi?valued dependencies",
                                  "Applies or relaxes up to Fifth Normal Form (5NF) based on query patterns, articulating performance vs. integrity trade?offs and documenting rationale",
                                  "Establishes organization?wide normalization standards, automates dependency analysis, and mentors teams on designing scalable, anomaly?free relational models"
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Schema Design",
                   "levels":  [
                                  "Choose appropriate data types and constraints",
                                  "Create normalized schemas aligned with data usage",
                                  "Design schemas for scalability and data integration",
                                  "Manage schema evolution, compatibility, and backward/forward changes",
                                  "Define enterprise-wide schema governance processes"
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Data Discovery",
                   "levels":  [
                                  "Document datasets and add tags/descriptions",
                                  "Maintain dataset lineage and usage examples",
                                  "Collaborate with teams to define ownership and business definitions",
                                  "Implement organization-wide discovery, governance, and metadata strategy",
                                  "Lead data workshops with execs and relevant stakeholders"
                              ]
               },
               {
                   "subdomain":  "Data Integration \u0026 Transformation",
                   "skill":  "ETL/ELT Design",
                   "levels":  [
                                  "Describe ETL/ELT flow stages and tools",
                                  "Build basic pipelines using tools like dbt or Airflow",
                                  "Design reusable pipeline components with testing",
                                  "Handle dependencies, failures, and recovery in pipeline frameworks",
                                  "Architect end-to-end data movement systems across platforms"
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Python for Data Engineering",
                   "levels":  [
                                  "Use pandas to clean and merge small datasets",
                                  "Build scripts using functions and error handling",
                                  "Design reusable modules for pipeline tasks",
                                  "Write performant code with memory profiling and logging",
                                  "Configure spark environments, practice paralellism and partitioning, and balances cost, security, governance, and performance trade?offs when choosing file formats, clusters, and orchestration patterns"
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Shell Scripting",
                   "levels":  [
                                  "Create basic scripts for core file and directory operations",
                                  "Employ variables, loops, and conditionals; schedule execution with the operating system�s native job scheduler",
                                  "Develop modular, reusable scripts with robust logging and error handling",
                                  "Integrate scripts with orchestration frameworks or cloud command?line interfaces to automate infrastructure and deployments",
                                  "Architect shell?based boot?strap, health?check, and monitoring utilities for production environments"
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Data Masking \u0026 Anonymization",
                   "levels":  [
                                  "Identify sensitive fields and masking needs",
                                  "Apply column-level masking using tools or scripts",
                                  "Implement masking for multiple formats and systems",
                                  "Build re-identification-safe anonymization pipelines",
                                  "Standardize privacy-preserving transformations org-wide"
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Stored Procedures",
                   "levels":  [
                                  "Execute simple stored procedures with parameters",
                                  "Write basic procedures with control flow and error handling",
                                  "Build procedures with logic, transactions, and optimization",
                                  "Design modular, reusable procedures for pipelines",
                                  "Define standards, refactor legacy code, and understand sys tables and meta data and engine complexities"
                              ]
               },
               {
                   "subdomain":  "Distributed Data Processing",
                   "skill":  "Apache Spark",
                   "levels":  [
                                  "Run simple PySpark transformations",
                                  "Use Spark SQL, caching, and partitioning",
                                  "Tune jobs using memory and execution metrics",
                                  "Optimize DAG execution and Spark configurations",
                                  "Architect distributed compute strategies across clusters"
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Kafka Streams",
                   "levels":  [
                                  "Understand producers, topics, and consumers",
                                  "Stream and transform messages using Kafka tools",
                                  "Ensure fault tolerance and message ordering",
                                  "Manage schema evolution and compaction strategies",
                                  "Design low-latency streaming architectures at scale"
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Performance Tuning",
                   "levels":  [
                                  "Identify performance bottlenecks in basic code",
                                  "Profile Spark or Python jobs using logs and metrics",
                                  "Use partitioning, batching, and caching to optimize throughput",
                                  "Implement job retries, resource scaling, and backpressure handling",
                                  "Tune end-to-end data platforms with cost/performance tradeoffs"
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Resource Management",
                   "levels":  [
                                  "Monitor job memory and CPU usage",
                                  "Set limits and quotas for jobs and containers",
                                  "Balance load across clusters or pipeline stages",
                                  "Automate autoscaling policies in orchestration tools",
                                  "Optimize cost and SLA tradeoffs with resource budgets"
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Compute Engines",
                   "levels":  [
                                  "Runs individual jobs on a single?node or default cluster configuration, setting only basic CPU and memory parameters",
                                  "Monitors resource metrics and adjusts executor / worker sizing, parallelism, and basic autoscaling policies to keep jobs stable",
                                  "Designs distributed workflows that exploit shuffling, caching, and data locality, choosing the right compute engine (Spark, Dask, Flink, etc.) for each workload",
                                  "Tunes multi?tenant clusters for mixed batch?and?stream processing, implements checkpointing and fault recovery, and enforces security, quota, and cost?governance rules",
                                  "Architects adaptive, cloud?agnostic compute platforms that dynamically allocate resources across heterogeneous engines, automating workload placement and mentoring teams on performance engineering best practices"
                              ]
               },
               {
                   "subdomain":  "Workflow Orchestration \u0026 Automation",
                   "skill":  "DAG Scheduling",
                   "levels":  [
                                  "Lists task dependencies and produces a simple topological (topographic) order to run steps one after another",
                                  "Uses basic graph traversals like DFS or BFS to confirm the DAG is acyclic and derive an initial run sequence",
                                  "Implements formal topological?sort algorithms (e.g., Kahn�s) to surface parallel branches and generate concurrency?safe execution plans",
                                  "Optimizes schedules via critical?path analysis and longest?path heuristics, reordering tasks to maximize resource usage while honoring dependencies",
                                  "Designs adaptive DAG schedulers that dynamically re?prioritize nodes with algorithmic strategies (leveled BFS, priority queues, hybrid DFS/BFS) to balance load and guarantee deterministic results at scale"
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "CI/CD for Pipelines",
                   "levels":  [
                                  "Push and pull pipeline code using Git",
                                  "Add linting and simple tests to CI workflows",
                                  "Deploy pipelines through automation pipelines. Understand topographics",
                                  "Use secrets and multi-environment configurations",
                                  "Architect robust CI/CD workflows for pipeline stacks. Understand application of BFS and DFS and when to apply them"
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Monitoring \u0026 Alerting",
                   "levels":  [
                                  "Read logs and detect job failures",
                                  "Configure alerts on DAG failures and SLAs",
                                  "Build basic dashboard for pipeline health",
                                  "Integrate observability tooling (e.g., Prometheus, Grafana)",
                                  "Implement unified monitoring across orchestration layers"
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Dependency Management",
                   "levels":  [
                                  "Order basic tasks and define static dependencies",
                                  "Use conditional branching and sensors in DAGs",
                                  "Handle retries, backfills, and downstream effects",
                                  "Manage inter-DAG and external task dependencies",
                                  "Design global DAG dependency and alert frameworks"
                              ]
               },
               {
                   "subdomain":  "Programming \u0026 Software Engineering for Data",
                   "skill":  "Git \u0026 Version Control",
                   "levels":  [
                                  "Clone repos and push commits",
                                  "Use feature branches and resolve merge conflicts",
                                  "Apply semantic versioning and code reviews",
                                  "Enforce repo structure and commit standards",
                                  "Design branching strategy and enforce across teams"
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Unit \u0026 Integration Testing",
                   "levels":  [
                                  "Write tests for functions and expected output",
                                  "Use mocks and fixtures for simple data tests",
                                  "Test integration across pipeline components",
                                  "Automate testing in CI/CD pipelines",
                                  "Develop data test frameworks with schema and value checks"
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Modular Design",
                   "levels":  [
                                  "Craft and consistently reuse small, single?responsibility functions to speed personal development and demonstrate quick wins for clients",
                                  "Group related functions into clear modules or installable packages so teammates at the client site can import, configure, and extend them with minimal friction",
                                  "Refactor end?to?end pipelines into unit?tested, parameter?driven components that isolate client?specific logic from generic processing steps",
                                  "Publish reusable data libraries and reference processes to the consulting project�s shared repo�complete with README examples, versioning, and basic CI�so multiple squads can adopt a common approach",
                                  "Architect a plug?in?based pipeline framework with documented extension points that lets future consulting teams drop in new sources, transforms, or sinks without rewriting core orchestration or quality controls"
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Containerization",
                   "levels":  [
                                  "Build container images for local development and testing",
                                  "Run and orchestrate multi?container stacks locally to support development workflows",
                                  "Package application or data pipelines into containers for consistent deployment across environments",
                                  "Deploy and scale containers with a cluster?level orchestration platform",
                                  "Govern image registries, curate base images, and enforce container security and compliance"
                              ]
               },
               {
                   "subdomain":  "Storage Systems \u0026 Data Formats",
                   "skill":  "Cloud Storage",
                   "levels":  [
                                  "Upload and download files using CLI or SDK",
                                  "Configure permissions and lifecycle rules",
                                  "Automate data storage with versioning and encryption",
                                  "Manage regional replication and failover",
                                  "Design secure, cost-effective cloud storage systems"
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Data Warehouses",
                   "levels":  [
                                  "Query and write to data warehouses",
                                  "Design partitioning and clustering strategies",
                                  "Manage cost and performance tradeoffs",
                                  "Optimize metadata configs and materialized views",
                                  "Architect cross-region, scalable warehouse systems"
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "File Formats",
                   "levels":  [
                                  "Read and write CSV, JSON, Parquet",
                                  "Choose efficient formats for data use",
                                  "Tune compression and encoding settings",
                                  "Benchmark file performance across workflows",
                                  "Standardize format use and storage strategy org-wide"
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Partitioning \u0026 Bucketing",
                   "levels":  [
                                  "Query partitioned datasets by filter",
                                  "Design partitioning strategies (e.g., date, region)",
                                  "Optimize bucket sizes and split thresholds",
                                  "Manage file layout for parallelism and cost",
                                  "Implement partition/bucket strategies across domains"
                              ]
               },
               {
                   "subdomain":  "Data Quality, Observability \u0026 Reliability",
                   "skill":  "Data Quality Validation",
                   "levels":  [
                                  "Run row-level or column-level checks",
                                  "Use validation libraries (e.g. Great Expectations)",
                                  "Integrate tests into orchestration workflows",
                                  "Automate schema and value checks at scale",
                                  "Create shared quality definitions and test registries"
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Anomaly Detection",
                   "levels":  [
                                  "Identify outliers using visual summaries",
                                  "Use thresholds and statistical checks",
                                  "Detect schema and distribution drift",
                                  "Set up alerting on model and data health",
                                  "Build ML-based anomaly detection on pipelines"
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Logging",
                   "levels":  [
                                  "Add print/debug statements to scripts",
                                  "Use structured logging for events",
                                  "Write logs to files or systems like CloudWatch",
                                  "Centralize logs and monitor patterns",
                                  "Define logging and alerting strategy for observability stack"
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "SLA Management",
                   "levels":  [
                                  "Track pipeline run time and failures",
                                  "Configure task-level SLAs and alerts",
                                  "Analyze trends in SLA misses",
                                  "Report SLA violations and mitigations",
                                  "Automate SLA enforcement and escalation policies"
                              ]
               },
               {
                   "subdomain":  "Governance, Security \u0026 Compliance",
                   "skill":  "IAM \u0026 Access Control",
                   "levels":  [
                                  "Access data using role credentials",
                                  "Apply fine-grained permissions on datasets",
                                  "Audit permissions and secure credentials",
                                  "Implement RBAC/ABAC across services",
                                  "Design and manage org-wide data access controls"
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Encryption",
                   "levels":  [
                                  "Understand in-transit and at-rest encryption",
                                  "Use managed encryption services like KMS",
                                  "Encrypt files programmatically",
                                  "Rotate keys and manage secure access",
                                  "Design encryption policy across data domains"
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Metadata Management",
                   "levels":  [
                                  "Read table metadata and column info",
                                  "Document metadata manually",
                                  "Automate metadata population (e.g., via ingestion)",
                                  "Ensure metadata accuracy and lineage tracking",
                                  "Build enterprise metadata pipelines and governance"
                              ]
               },
               {
                   "subdomain":  "",
                   "skill":  "Compliance Standards",
                   "levels":  [
                                  "Understand common regulations (GDPR, HIPAA)",
                                  "Apply tagging or masking to PII fields",
                                  "Document controls and audits for compliance",
                                  "Monitor compliance with automated checks",
                                  "Work with Security Stakeholders / Legal to identify potential crises areas"
                              ]
               },
               {
                   "subdomain":  "DE Project Management",
                   "skill":  "Project Management",
                   "levels":  [
                                  "Captures high?level objectives and lists the basic ETL tasks, data sources, and success criteria in a simple checklist or spreadsheet.",
                                  "Breaks work into sprints or milestones, estimates effort, and tracks task status while flagging obvious data?quality or dependency risks.",
                                  "Produces a detailed project plan that maps data flows, resource needs, and cross?team hand?offs, complete with timelines, risk log, and stakeholder communication cadence.",
                                  "Integrates architecture decisions, capacity?/?cost projections, compliance checkpoints, and CI/CD gates into the roadmap, continuously re?forecasting based on velocity and metrics.",
                                  "Orchestrates multi?program portfolios that align data?platform evolution with business OKRs, sequencing parallel workstreams, governing scope change, and defining measurable value KPIs."
                              ]
               }
           ],
    "ai":  [
               {
                   "subdomain":  "Context Engineering",
                   "skill":  "Prompt Design and Clarity",
                   "levels":  [
                                  "Understands what a prompt is and can use basic, direct questions (e.g., Summarize this paragraph)",
                                  "Can write functional prompts with some tuning (e.g., changing tone or format), usually through trial and error",
                                  "Crafts clear, targeted prompts with intent (e.g., persona-based prompts, format constraints) and predicts model behavior",
                                  "Designs nuanced prompts tailored to task complexity, refines for performance, and reduces hallucinations",
                                  "Innovates new prompting strategies, teaches best practices, and actively contributes to evolving prompt methodologies"
                              ]
               },
               {
                   "subdomain":  "Context Engineering",
                   "skill":  "Prompting Strategies and Patterns",
                   "levels":  [
                                  "Aware that prompts can be written in different ways (e.g., with or without examples); may use zero-shot without knowing the term",
                                  "Understands and can explain zero-shot, one-shot, and few-shot prompting; begins experimenting with examples or role-based setups",
                                  "Chooses prompting strategies based on task (e.g., uses few-shot for classification, chain-of-thought for reasoning); evaluates performance impact",
                                  "Refines and optimizes prompting techniques (e.g., tunes few-shot examples, uses role prompting or system messages for behavior control)",
                                  "Invents or systematizes advanced techniques (e.g., self-consistency, scratchpad reasoning); mentors others and contributes to prompt strategy design"
                              ]
               },
               {
                   "subdomain":  "Context Engineering",
                   "skill":  "Using Prompt Templates and Variables",
                   "levels":  [
                                  "Can fill in simple templates with static inputs when shown a working example",
                                  "Understands the use of input/output variables and can use basic templating tools (e.g., f-strings, Jinja)",
                                  "Creates reusable prompt templates for common tasks, dynamically fills in variables from app code or workflows",
                                  "Optimizes prompt templates for performance, integrates them into tooling (e.g., LangChain/Flowise), and manages template versions",
                                  "Designs scalable prompt templating frameworks, mentors others on maintainability, and drives prompt system architecture"
                              ]
               },
               {
                   "subdomain":  "Context Engineering",
                   "skill":  "Token and Context Management",
                   "levels":  [
                                  "Aware that models have context limits (e.g., �4,000 tokens�); may not fully understand implications",
                                  "Can count tokens using tooling (e.g., tiktoken, tokenizer APIs); trims inputs manually when needed",
                                  "Designs prompts and applications to stay within context window; understands tradeoffs between input length, quality, and cost",
                                  "Implements dynamic context management (e.g., truncation strategies, selective summarization, memory buffers)",
                                  "Innovates on context compression, hierarchical memory, or long-context strategies; advises or contributes to scalable LLM deployment patterns"
                              ]
               },
               {
                   "subdomain":  "Context Engineering",
                   "skill":  "Prompt Chaining for Multi-Step Reasoning",
                   "levels":  [
                                  "Aware that large tasks can be broken down, but may not know how to design or execute chained prompts",
                                  "Can use a predefined multi-prompt flow to perform step-by-step tasks (e.g., retrieve \u003e summarize \u003e reformat)",
                                  "Designs simple prompt chains to guide reasoning or improve reliability in multi-step tasks",
                                  "Builds robust chained workflows using frameworks (e.g., LangChain agents), and handles errors and edge cases",
                                  "Invents novel reasoning patterns, teaches chaining strategies, and contributes to best practices for LLM workflow design"
                              ]
               },
               {
                   "subdomain":  "Foundational Models \u0026 Adaptation",
                   "skill":  "Understanding Transformer Architectures",
                   "levels":  [
                                  "Has heard of transformers and knows they�re used in LLMs; may recognize �attention is all you need�",
                                  "Can explain basic components (e.g., attention, embeddings, positional encoding) with some inaccuracies",
                                  "Understands transformer internals well enough to select or compare models based on architecture",
                                  "Analyzes model behavior or performance in context of architectural trade-offs (e.g., decoder-only vs encoder-decoder)",
                                  "Can teach or innovate on architecture; contributes to research, benchmarking, or new model designs"
                              ]
               },
               {
                   "subdomain":  "Foundational Models \u0026 Adaptation",
                   "skill":  "Using Open and Closed Source Models (e.g., GPT-4, LLaMA)",
                   "levels":  [
                                  "Can run basic queries through OpenAI or Hugging Face interfaces",
                                  "Knows the difference between hosted APIs and local models; can switch between GPT, Claude, LLaMA, etc",
                                  "Selects and configures appropriate models (API vs local), understands performance/cost tradeoffs",
                                  "Benchmarks model performance, deploys models across different environments, understands licensing implications",
                                  "Evaluates, modifies, or fine-tunes models at scale; helps organizations build hybrid or fallback strategies"
                              ]
               },
               {
                   "subdomain":  "Foundational Models \u0026 Adaptation",
                   "skill":  "Fine-Tuning Models (full, LoRA, PEFT)",
                   "levels":  [
                                  "Aware that models can be trained further; has heard terms like �fine-tuning� or �domain adaptation.�",
                                  "Can follow tutorials to perform basic fine-tuning tasks using libraries like Hugging Face or PEFT",
                                  "Chooses the right fine-tuning method (full vs LoRA vs prompt tuning), and runs training workflows effectively",
                                  "Optimizes fine-tuning pipelines (e.g., hyperparams, evaluation), manages training data pipelines",
                                  "Develops novel fine-tuning techniques, integrates them at scale, and contributes to open-source or research"
                              ]
               },
               {
                   "subdomain":  "Foundational Models \u0026 Adaptation",
                   "skill":  "Using Adapters and Model Quantization",
                   "levels":  [
                                  "Has heard of adapters or quantization, but may not know when or why to use them",
                                  "Can follow instructions to apply adapters or quantize models using off-the-shelf tooling",
                                  "Applies quantization or adapter strategies appropriately to optimize model deployment and cost",
                                  "Benchmarks performance vs quality trade-offs; understands implications of precision levels (e.g., int8, fp16)",
                                  "Designs deployment strategies with custom adapters/quantization; contributes to tooling or optimization frameworks"
                              ]
               },
               {
                   "subdomain":  "RAG (Retrieval-Augmented Generation)",
                   "skill":  "Text Chunking Strategies",
                   "levels":  [
                                  "Aware that long texts must be broken into parts; uses basic naive splits (e.g., every 500 words)",
                                  "Can apply simple chunking techniques (e.g., overlapping windows, paragraph-based) via prebuilt tools",
                                  "Selects chunking strategy based on context length, semantic boundaries, and task type (e.g., summarization vs Q\u0026A)",
                                  "Designs and evaluates custom chunking logic using metadata, headings, or content density; considers chunk cohesion",
                                  "Innovates new chunking methods to improve downstream RAG performance; shares tools or research with the community"
                              ]
               },
               {
                   "subdomain":  "RAG (Retrieval-Augmented Generation)",
                   "skill":  "Embedding Generation and Similarity Search",
                   "levels":  [
                                  "Can generate embeddings using provided APIs (e.g., OpenAI, Hugging Face) for a few documents",
                                  "Understands the concept of semantic similarity and vector space; can implement basic search",
                                  "Selects embedding models, tunes similarity thresholds, and handles common edge cases (e.g., null vectors)",
                                  "Benchmarks different embedding models for task fit; implements reranking or hybrid search techniques",
                                  "Designs embedding strategies (e.g., task-specific training), evaluates at scale, and contributes to model or index improvement"
                              ]
               },
               {
                   "subdomain":  "RAG (Retrieval-Augmented Generation)",
                   "skill":  "Vector Database Use (e.g., FAISS, Pinecone, Weaviate)",
                   "levels":  [
                                  "Can store and retrieve embeddings using a simple API (e.g., Pinecone quickstart, ChromaDB)",
                                  "Sets up a basic vector database, loads data, and performs similarity searches with guidance",
                                  "Selects appropriate vector DB for the task; implements filtering, metadata queries, and hybrid search",
                                  "Tunes index parameters (e.g., ANN algorithms), manages data lifecycle, and scales databases in production",
                                  "Designs distributed or multi-modal vector stores; contributes to indexing strategy or underlying vector DB performance"
                              ]
               },
               {
                   "subdomain":  "RAG (Retrieval-Augmented Generation)",
                   "skill":  "Designing Effective RAG Pipelines (Fusion/Fan-Out, Ranking, Fallback)",
                   "levels":  [
                                  "Understands that RAG connects retrieval with generation, but may struggle with implementation",
                                  "Can build basic RAG pipeline using templates or tools (e.g., LangChain retrieval chain)",
                                  "Designs RAG workflows with appropriate retrieval logic (e.g., top-k, rerankers) and integrates into apps",
                                  "Builds robust pipelines using fusion, fan-out, fallback, or multi-hop retrieval; evaluates latency vs quality",
                                  "Innovates new retrieval-generation workflows; contributes to open-source or publishes case studies/research on RAG"
                              ]
               },
               {
                   "subdomain":  "Stateful AI and Memory",
                   "skill":  "Desiging and Using Memory Systems",
                   "levels":  [
                                  "Understands that memory enables AI to retain information across interactions; recognizes stateless vs stateful behavior",
                                  "Uses basic memory modules in frameworks like LangChain or Semantic Kernel; distinguishes short-term (buffer, sliding window) and long-term memory (vector-based retrieval)",
                                  "Designs and applies memory in projects; selects appropriate strategies and tools (e.g., Redis, vector stores)",
                                  "Customizes memory architecture for specific use cases (e.g., summarization for long sessions, episodic memory for agents); balances cost, context, performance and builds reusable patterns",
                                  "Leads design of scalable memory systems; develops new patterns, mentors others, and aligns memory with governance standards"
                              ]
               },
               {
                   "subdomain":  "Stateful AI and Memory",
                   "skill":  "Implementing Conversational Memory",
                   "levels":  [
                                  "Recognizes that conversations need context to feel natural; aware that models �forget� without memory and can�t continue coherent dialogue on their own",
                                  "Uses framework-provided tools (e.g., session buffers, turn logs) to maintain conversation history; enables basic continuity and recall within a single session",
                                  "Implements memory that persists across turns and sessions; uses summarization, tagging, or segmentation to retain relevant context and user details",
                                  "Designs conversational flows that adapt based on memory (e.g., user preferences, prior topics); manages memory lifecycle to balance personalization with accuracy and cost",
                                  "Architects dynamic, context-aware memory systems for complex chat experiences; aligns memory strategy with UX goals, privacy controls, and long-term user engagement. Mentors others on creating human-like continuity in AI interactions"
                              ]
               },
               {
                   "subdomain":  "Stateful AI and Memory",
                   "skill":  "Ethical Use and Governance of AI Memory",
                   "levels":  [
                                  "Understands that storing memory raises ethical and privacy concerns; aware of basic data sensitivity",
                                  "Applies basic guardrails (e.g., not storing PII); recognizes when consent, transparency, or deletion may be needed",
                                  "Designs memory features with user consent, data minimization, and retention policies in mind",
                                  "Implements governance controls (e.g., audit logs, opt-out mechanisms); ensures alignment with internal and external compliance standards",
                                  "Leads ethical design of memory systems; defines policy, aligns architecture with regulatory frameworks (e.g., GDPR), and educates others on responsible practices"
                              ]
               },
               {
                   "subdomain":  "AI System Integration",
                   "skill":  "Calling models via API (e.g., OpenAI, Azure, Hugging Face)",
                   "levels":  [
                                  "Can call an AI model using simple REST or SDK examples (e.g., OpenAI Playground or curl snippet)",
                                  "Can call APIs programmatically from apps using authentication and basic parameterization",
                                  "Manages API versioning, rate limits, and retries; uses environment variables for secure deployment",
                                  "Abstracts and wraps model APIs into reusable components; handles multi-provider support and logging",
                                  "Designs robust multi-model infrastructure; contributes to SDKs, best practices, or enterprise AI platform integration"
                              ]
               },
               {
                   "subdomain":  "AI System Integration",
                   "skill":  "Chaining model calls with tools/workflows (e.g., LangChain, Semantic Kernel)",
                   "levels":  [
                                  "Follows tutorials to chain two or more steps in a basic agent or workflow",
                                  "Understands agent architectures; chains tools or prompts using existing components in frameworks",
                                  "Designs full pipelines for task execution, including memory, planning, and tool invocation",
                                  "Builds modular, reusable agents or chains that can handle conditional logic, tool errors, and state",
                                  "Invents new chaining patterns, extends frameworks with custom logic, and helps define best practices for tool orchestration"
                              ]
               },
               {
                   "subdomain":  "AI System Integration",
                   "skill":  "Integrating model outputs into applications (frontend, backend)",
                   "levels":  [
                                  "Can display raw model output in a basic interface (e.g., textbox ? response box)",
                                  "Uses model responses in form fields, dashboards, or chat UIs with some formatting or logic",
                                  "Designs APIs or service layers to process model responses, handle edge cases, and enable testing",
                                  "Integrates model feedback loops, streaming outputs, or real-time updates into app UX patterns",
                                  "Leads AI-driven product architecture; ensures LLM behavior aligns with business logic and user experience"
                              ]
               },
               {
                   "subdomain":  "Model Deployment and Infrastructure",
                   "skill":  "Hosting models (local, cloud, edge)",
                   "levels":  [
                                  "Can run a small model locally using example scripts (e.g., Hugging Face transformers or text-generation-webui)",
                                  "Deploys basic models to cloud services (e.g., Hugging Face Spaces, Azure ML) using default settings",
                                  "Configures model serving environments, manages hardware requirements (e.g., GPU/CPU), and selects deployment targets",
                                  "Deploys and scales models in containerized/cloud-native environments (e.g., Docker, Kubernetes, Azure AKS)",
                                  "Designs hybrid/local/cloud deployment strategies, optimizes resource allocation, and supports edge inference pipelines"
                              ]
               },
               {
                   "subdomain":  "Model Deployment and Infrastructure",
                   "skill":  "Using inference endpoints and serverless functions",
                   "levels":  [
                                  "Understands the concept of an inference endpoint and can query it via HTTP tools (e.g., Postman)",
                                  "Deploys serverless inference using low-code platforms (e.g., Azure Functions, AWS Lambda with Hugging Face endpoints)",
                                  "Wraps models as REST endpoints with validation, authentication, and async processing logic",
                                  "Orchestrates inference across serverless environments with autoscaling, observability, and traffic management",
                                  "Designs cost-optimized, multi-region serverless inference architecture and integrates with enterprise systems"
                              ]
               },
               {
                   "subdomain":  "Model Deployment and Infrastructure",
                   "skill":  "Caching, rate limiting, and scaling",
                   "levels":  [
                                  "Aware that model queries can be slow or expensive; may recognize API limits",
                                  "Uses basic caching strategies (e.g., memoization, API gateway caching) and understands rate limits",
                                  "Implements caching layers (e.g., Redis), configures rate limits, and monitors throughput",
                                  "Designs horizontal/vertical scaling strategies, optimizes for latency and concurrency.",
                                  "Builds globally distributed, auto-scaling AI services with fine-tuned traffic control, caching policies, and fallback logic."
                              ]
               },
               {
                   "subdomain":  "Model Deployment and Infrastructure",
                   "skill":  "Using MLOps pipelines (CI/CD for models)",
                   "levels":  [
                                  "Has heard of MLOps; understands it�s similar to DevOps for models",
                                  "Uses simple pipelines to retrain or redeploy models via notebooks or tools like MLflow or Azure ML Designer",
                                  "Sets up automated CI/CD workflows for models using GitHub Actions, MLflow, or Azure DevOps",
                                  "Integrates model training, evaluation, deployment, and rollback into a cohesive pipeline with monitoring",
                                  "Designs robust MLOps architecture for enterprise scale, incorporating audit trails, lineage tracking, and compliance requirements"
                              ]
               },
               {
                   "subdomain":  "Evaluation, Safety, and Monitoring",
                   "skill":  "Designing human \u0026 automated evaluation methods",
                   "levels":  [
                                  "Understands that AI outputs need to be evaluated; can give basic subjective feedback",
                                  "Uses predefined metrics (e.g., BLEU, ROUGE, accuracy) and performs basic A/B comparisons",
                                  "Designs task-specific evaluation frameworks combining human review and metrics (e.g., faithfulness, relevance)",
                                  "Implements human-in-the-loop pipelines, feedback collection, and aggregated scoring dashboards",
                                  "Innovates new evaluation methodologies for complex LLM use cases (e.g., helpfulness, bias, trustworthiness); contributes to standardization"
                              ]
               },
               {
                   "subdomain":  "Evaluation, Safety, and Monitoring",
                   "skill":  "Hallucination detection and mitigation",
                   "levels":  [
                                  "Aware that models can �make things up� but unsure how to detect or fix it",
                                  "Can identify obvious hallucinations and uses simple prompts or grounding to reduce them",
                                  "Uses retrieval grounding, prompt structure, or model choice to reduce hallucination risk",
                                  "Evaluates hallucination rates systematically, designs verification workflows or filters",
                                  "Develops mitigation techniques (e.g., response calibration, post-generation validation); shares findings or publishes benchmarks"
                              ]
               },
               {
                   "subdomain":  "Evaluation, Safety, and Monitoring",
                   "skill":  "Use of guardrails and moderation APIs",
                   "levels":  [
                                  "Knows that AI outputs may need filtering; may use UI-based moderation settings",
                                  "Applies prebuilt guardrails or content filters (e.g., OpenAI moderation API, Azure Content Safety)",
                                  "Implements structured safety checks using multiple layers (pre-/post-filtering, rule-based)",
                                  "Builds custom guardrails aligned to business or user-specific safety thresholds",
                                  "Develops reusable moderation pipelines, influences policy frameworks, or contributes to safety tooling"
                              ]
               },
               {
                   "subdomain":  "Evaluation, Safety, and Monitoring",
                   "skill":  "Logging and feedback collection",
                   "levels":  [
                                  "Can manually track responses or see history in a basic tool interface",
                                  "Captures and reviews model inputs/outputs via application logs or lightweight tooling",
                                  "Implements structured logging with metadata (e.g., user IDs, latency, token usage) for observability",
                                  "Builds feedback loops into applications; aggregates logs for analytics and error tracing",
                                  "Designs systems for longitudinal performance tracking, user feedback analysis, and dynamic model improvements"
                              ]
               },
               {
                   "subdomain":  "AI Ethics, Risk \u0026 Governance",
                   "skill":  "Understanding responsible AI principles",
                   "levels":  [
                                  "Aware that AI can introduce risks or bias; has heard terms like fairness, transparency, and accountability",
                                  "Can describe key responsible AI principles and cite examples of harm or unintended outcomes",
                                  "Applies RAI principles to guide design or decision-making in practical projects",
                                  "Leads ethical reviews, creates checklists or standards for RAI adoption within teams",
                                  "Shapes organizational policy or frameworks for responsible AI; contributes to thought leadership or academic discussion"
                              ]
               },
               {
                   "subdomain":  "AI Ethics, Risk \u0026 Governance",
                   "skill":  "Data privacy and model misuse risks",
                   "levels":  [
                                  "Knows that sensitive data should not be shared with AI tools; aware of basic privacy concerns",
                                  "Understands PII, data retention, and consent principles; avoids using private data in prompts",
                                  "Implements privacy-aware workflows (e.g., anonymization, encryption, audit logs)",
                                  "Anticipates potential misuse scenarios and develops countermeasures (e.g., prompt injection protection)",
                                  "Guides governance decisions on data use and misuse risks at scale; contributes to regulatory guidance or standards"
                              ]
               },
               {
                   "subdomain":  "AI Ethics, Risk \u0026 Governance",
                   "skill":  "Regulatory awareness (e.g., EU AI Act, NIST RMF)",
                   "levels":  [
                                  "Aware that regulations exist but not familiar with names or implications",
                                  "Recognizes key frameworks and can describe basic obligations (e.g., risk tiers in EU AI Act)",
                                  "Ensures AI systems comply with relevant laws and standards; participates in compliance reviews",
                                  "Maps technical decisions to regulatory categories and designs compliant system architectures",
                                  "Advises on or helps shape policy and regulation at the organizational or industry level"
                              ]
               },
               {
                   "subdomain":  "AI Ethics, Risk \u0026 Governance",
                   "skill":  "Documentation practices (model cards, data sheets, auditability)",
                   "levels":  [
                                  "Understands that documentation is important but may not know what model cards are",
                                  "Can read and use model documentation (e.g., model cards, data statements) to make basic decisions",
                                  "Produces model documentation, tracks data lineage, and maintains reproducibility in AI workflows",
                                  "Builds documentation processes into development pipelines; aligns artifacts with compliance needs",
                                  "Defines standards for documentation quality and consistency; advocates for transparency and traceability at scale"
                              ]
               },
               {
                   "subdomain":  "Use Case Identification",
                   "skill":  "Framing business problems for AI",
                   "levels":  [
                                  "Understands that AI is more useful when applied to business problems (not just tasks) and can identify general problem categories",
                                  "Helps clients reframe tasks or ideas as outcomes-driven problems and can distinguish between well-posed vs ambiguous use cases",
                                  "Translates client pain points into AI-suitable problem statements and avoids jumping to solutions too early",
                                  "Guides clients through structured problem-framing sessions and ensures alignment with organizational goals",
                                  "Teaches and scales problem-framing best practices across teams and contributes to shaping how clients define their AI roadmap"
                              ]
               },
               {
                   "subdomain":  "Use Case Identification",
                   "skill":  "Assessing AI fit and solution type",
                   "levels":  [
                                  "Understands that not all problems are AI problems and can identify broad AI categories (e.g., classification, generation)",
                                  "Uses simple criteria to assess fit (e.g., need for pattern recognition, ambiguous rules, data availability)",
                                  "Applies structured fit assessment methods and maps problems to appropriate AI solutions (e.g., summarization, RAG, NLP)",
                                  "Advises on trade-offs of different AI solutions given constraints, and matches fit to client maturity and risk tolerance",
                                  "Shapes methodology for assessing fit across engagements and is recognized for judgment on when to recommend or avoid AI"
                              ]
               },
               {
                   "subdomain":  "Use Case Identification",
                   "skill":  "Scoping technical feasability",
                   "levels":  [
                                  "Understands that data and infrastructure affect AI outcomes and is aware of basic prerequisites",
                                  "Can ask questions about data readiness, model access, and governance to gauge viability",
                                  "Assesses feasibility using client inputs on data quality, APIs, and compute; flags blockers early",
                                  "Works with technical teams to refine scope based on system constraints and delivery realities",
                                  "Leads technical due diligence and scopes realistic pathways forward even in low-maturity environments"
                              ]
               },
               {
                   "subdomain":  "Use Case Identification",
                   "skill":  "Estimating value and prioritizing opportunities",
                   "levels":  [
                                  "Understands that not all AI projects are equally valuable. Aware of basic business impact categories",
                                  "Uses templates to estimate potential gains (e.g., cost savings, efficiency) and classify value levels",
                                  "Prioritizes use cases based on effort vs value; considers adoption complexity and outcome alignment",
                                  "Builds lightweight business cases and recommends sequencing based on ROI, impact, and feasibility",
                                  "Shapes client roadmaps based on strategic value. Aligns opportunity assessment with executive priorities"
                              ]
               },
               {
                   "subdomain":  "Use Case Identification",
                   "skill":  "Avoiding overengineering",
                   "levels":  [
                                  "Aware that AI isn\u0027t always the right answer, and recognizes when simpler solutions might work.",
                                  "Flags potential overuse of AI (e.g., chatbots for static FAQs); can suggest simpler alternatives",
                                  "Regularly challenges solution choices with \u0027what problem are we solving\u0027 and \u0027is AI necessary\u0027 lenses",
                                  "Guides teams away from unnecessary complexity; reframes efforts toward lean, impact-driven solutions",
                                  "Influences client culture to be skeptical of AI hype; advocates for responsible, grounded solutioning"
                              ]
               },
               {
                   "subdomain":  "Use Case Identification",
                   "skill":  "Aliging use cases with client goals and readiness",
                   "levels":  [
                                  "Understands that client culture, goals, and maturity impact success, and is aware of basic readiness factors",
                                  "Identifies misalignments between use cases and client expectations or capabilities",
                                  "Proactively adjusts recommendations based on readiness, change appetite, and stakeholder dynamics",
                                  "Facilitates alignment across client roles and stages use cases based on near- and long-term value",
                                  "Shapes strategy based on deep client understanding; ensures buy-in, sustainability, and outcome alignment"
                              ]
               },
               {
                   "subdomain":  "Project Management",
                   "skill":  "Managing uncertainty and iteration",
                   "levels":  [
                                  "AI outcomes are often non-deterministic and require experimentation, which makes scope fluid",
                                  "Success depends on data quality and availability, not just building code � requires data profiling and stakeholder coordination",
                                  "AI projects must include success metrics, human feedback paths, and model evaluation workflows",
                                  "Teams must manage trade-offs between exploration (e.g., prompt tuning, model comparison) and shippable product features",
                                  "AI work spans data science, engineering, product, and sometimes compliance/legal � alignment is key"
                              ]
               },
               {
                   "subdomain":  "Project Management",
                   "skill":  "Data-aware scoping and planning",
                   "levels":  [
                                  "Aware that data quality and availability impact AI outcomes, and recognizes that planning requires more than just feature specs",
                                  "Asks questions about data sources, volume, and accessibility; surfaces early risks tied to missing or low-quality data",
                                  "Integrates data readiness into scope definitions and coordinates with data owners to assess feasibility before committing to timelines",
                                  "Anticipates downstream risks from poor or biased data, and builds in buffer time for discovery, profiling, cleaning, or augmentation tasks",
                                  "Leads data scoping alongside solution architecture and establishes best practices for aligning delivery scope with real-world data constraints; mentors teams to treat data as a first-class deliverable"
                              ]
               },
               {
                   "subdomain":  "Project Management",
                   "skill":  "Designing feedback loops and evalutation plans",
                   "levels":  [
                                  "Understands that AI systems need to be evaluated beyond �it runs and recognizes that human feedback is often required",
                                  "Tracks basic evaluation metrics (e.g., accuracy, latency) and logs model behavior; is aware that outputs need validation.",
                                  "Plans for human review or user feedback in early stages and etablishes evaluation criteria tied to the business goal (e.g., relevance, helpfulness)",
                                  "Designs feedback loops (manual or automated) into delivery plans and adjusts scope based on test outcomes or live usage signals",
                                  "Leads model evaluation strategy across engagements and builds repeatable systems for feedback-driven improvement; coaches teams on metrics, bias detection, and post-deployment monitoring"
                              ]
               },
               {
                   "subdomain":  "Project Management",
                   "skill":  "Balancing research and delivery",
                   "levels":  [
                                  "Understands that AI projects involve exploratory work and notices when team activities shift between experimentation and delivery",
                                  "Tracks exploratory work alongside deliverables, and helps capture what�s �good enough� to move forward while documenting open questions",
                                  "Plans projects with space for iteration while driving toward deadlines and keeps clients aligned when priorities shift from R\u0026D to shipping",
                                  "Manages scope creep from over-experimentation; helps teams define MVP boundaries, deliver iterative value, and timebox research efforts",
                                  "Leads projects where experimentation is an asset, not a risk; aligns R\u0026D phases to business goals, mentors others in making pragmatic trade-offs, and ensures project momentum"
                              ]
               },
               {
                   "subdomain":  "Project Management",
                   "skill":  "Orchestrating cross-disciplinary teams",
                   "levels":  [
                                  "Understands that AI projects require cross-functional input. Aware that roles and expectations may differ from traditional dev teams",
                                  "Facilitates coordination between data and product teams. Tracks dependencies and flags when disconnects arise",
                                  "Actively manages alignment across disciplines (e.g., data availability, model behavior, UX implications). Leads shared planning sessions and syncs",
                                  "Anticipates misalignment across roles and surfaces it early; designs workflows that keep engineering, ML, and business needs in sync",
                                  "Creates high-functioning, cross-disciplinary AI teams, and builds repeatable practices for aligning priorities, pacing, and language across domains; mentors others on navigating AI collaboration friction points"
                              ]
               }
           ]
};
