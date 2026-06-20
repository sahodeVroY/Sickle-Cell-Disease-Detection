# Proposed System

## 1. System Overview

The proposed system is an automated framework for detecting sickle cell morphology from peripheral blood smear images. The system begins with a traditional machine learning baseline in order to establish an initial reference for image classification performance. In this stage, the blood smear images are converted into numerical feature vectors and processed using classical algorithms such as Support Vector Machine, Random Forest, Logistic Regression, Decision Tree, and Naive Bayes. This baseline helps in understanding how well conventional machine learning methods can classify normal and sickle cell images.

Although traditional machine learning provides a useful starting point, it has limitations in handling raw image data effectively because it depends on manually prepared or flattened features and may fail to capture complex spatial patterns present in blood smear images. To overcome these limitations, a Convolutional Neural Network (CNN) is introduced. The CNN automatically learns hierarchical image features such as edges, shapes, and morphological structures directly from the input images, making it more suitable for identifying sickle-shaped cells accurately.

To further improve performance, the proposed system extends the approach using transfer learning models such as EfficientNetB0, ResNet50, and MobileNetV2. These models are pretrained on large image datasets and can extract more robust and generalized visual features. By fine-tuning these architectures on the sickle cell image dataset, the system benefits from stronger feature representation, improved classification capability, and better generalization compared to training from scratch.

Thus, the proposed system progresses from traditional machine learning baseline models to a custom CNN and finally to advanced transfer learning architectures. This layered approach not only provides a meaningful comparison between conventional and deep learning methods but also helps in identifying the most effective model for automated sickle cell morphology detection. In addition, explainable AI techniques are integrated to make the model predictions more transparent and interpretable.

The proposed system also addresses the major limitations of existing manual and conventional systems. Manual blood smear examination is time-consuming, labor-intensive, and dependent on expert interpretation, which can lead to inconsistency and observer variation. Traditional machine learning methods provide only baseline performance and are limited in capturing complex image patterns. The use of CNN overcomes this issue by learning important visual features automatically, while transfer learning models further enhance performance through pretrained knowledge and better feature extraction. Moreover, many existing AI systems do not provide sufficient interpretability; this limitation is addressed in the proposed system through Grad-CAM, LIME, and SHAP, which visually explain the reasoning behind the classification results. Therefore, the proposed system improves accuracy, consistency, transparency, and efficiency in sickle cell morphology detection.

**Table 1: Existing System vs. Proposed System**

| Limitation | Existing System | Proposed System |
| --- | --- | --- |
| Feature Extraction | Manual / handcrafted features | Automatic via CNN / Transfer Learning |
| Interpretability | None / black-box predictions | Grad-CAM, LIME, and SHAP explanations |
| Generalization | Low — trained on limited local data | Improved via ImageNet Transfer Learning |
| Evaluation Robustness | Single train/test split | Stratified 5-Fold Cross-Validation |
| Class Imbalance Handling | Not addressed | Data augmentation + stratified splits |
| Deployment | No deployment interface | Streamlit web app with real-time inference |

## 2. System Architecture

![Figure 1. System architecture overview](/mnt/c/Users/sahod/Desktop/project/converted_report_assets/image1.png)
*Figure 1. System architecture overview*

### High-Level Architecture Description

The proposed system architecture presents a comprehensive, end-to-end classification framework designed for detecting sickle cell morphology from peripheral blood smear images. Rather than relying on a single modeling paradigm, the system implements a dual-pathway approach evaluating both state-of-the-art Deep Learning pipelines against Traditional Machine Learning baselines.

Crucially, the architecture resolves the "black box" problem of advanced neural networks by terminating the pipelines natively into a layered Explainable AI (XAI) module. This design guarantees that every probabilistic prediction is accompanied by a transparent visual explanation of why the model made its decision, making the system viable for robust clinical or diagnostic contexts. The entire flow operates recursively through a Stratified 5-Fold Cross Validation generator to ensure that the evaluation metrics produced at the output layer are statistically robust and highly generalized.

### Major Components and Their Roles

### 2.1. Data Input Layer

Role: Ingestion and Organization

Description: Acts as the primary data ingestion point. It isolates and reads raw peripheral blood smear images from directories labeled as normal (or clear) and pathological sickle iterations. This component ensures independent streams of positive and negative classes are prepared without data leakage.

### 2.2. Preprocessing & Augmentation Pipeline

Role: Data Standardization and Matrix Handling

Description: Before mathematical modeling, all image batches must be strictly formatted. Images are structurally resized to 64x64 matrices and rescaled identically (1/255) to secure rapid gradient convergence. For the deep learning route, an augmentation layer (applying randomized algorithmic flips, rotations, and zooming) artificially inflates dataset variance to combat model overfitting. Conversely, for the traditional ML route, a Standard Scaler and a Principal Component Analysis (PCA - 100 components) drastically compress raw geometric pixels into lightweight mathematical features.

### 2.3. Core Modeling: Deep Learning Pipeline

Role: Primary Pathological Classification

Description: The system deploys multi-tiered neural networks to establish morphological predictions:

Custom CNN: Operates as an internal baseline with sequential Conv2D and MaxPool2D layers. It bypasses complex freezing mechanics and optimizes weights purely End-to-End.

Transfer Learning Backbones: Integrates massive, pre-trained ImageNet architectures (ResNet50, MobileNetV2, EfficientNetB0). They classify via an intensive Two-Stage Fine-Tuning mechanic: initially freezing the main backbone to learn safe feature abstractions via the top Dense Head, and subsequently unfreezing the architecture at a miniature learning rate to fine-tune the internal feature map directly for the sickle cells.

### 2.4. Core Modeling: Traditional ML Pipeline

Role: Baseline Algorithmic Comparison

Description: Bypassing convolutional tensors, this pipeline receives the 100-component PCA matrices and feeds them into established, computationally efficient machine learning paradigms: Support Vector Machines (RBF Kernel), Random Forests, naive Bayes, and Logistic Regression algorithms.

### 2.5. Explainable AI (XAI) Layer

Role: Predictive Transparency and Trust Mapping

Description: Ensures no prediction goes unexplained. It dissects the prediction algorithms via three distinct methodologies:

Grad-CAM Heatmaps: Locates the final convolutional layer of the deep networks and highlights the spatial regions mapping highest to the target class activation.

LIME Superpixels: Modifies segments of the input image locally to map and display the precise "superpixel" geometric zones driving the prediction up or down probabilistically.

SHAP Plots: Provides Shapley additive explanations primarily evaluating the overarching feature importance mathematically, particularly mapping which PCA features drove the Traditional ML estimators.

### 2.6. Output & Evaluation Module

Role: Validation, Diagnostic Outputs, and Model Persistence

Description: The final convergence node that outputs the definitive discrete class (Sickle vs. Normal). It aggregates outputs generated across the 5 independent data folds and computes finalized, robust clinical metrics such as ROC-AUC curves, F1-Scores, and comprehensive Confusion Matrices. The localized final states are saved immediately as transportable .keras weight configurations.

## 3. Dataset Description:

### 3.1. Dataset Overview

The dataset consists of microscopic images of red blood cells collected to facilitate the automated detection of sickle cell morphology. The images were processed using clinical-grade staining techniques and captured under high-magnification microscopy.

Total Positive (Sickle) Samples: 422 images.

Total Negative (Normal) Samples: 147 images.

Total Unclear Samples: 122 images (containing artefacts or poor staining; available on request).

Processing Methods: Samples were processed using Field stains and Leichman stains.

### 3.2. Origin and Data Collection

The dataset was collected from the Teso region in Eastern Uganda, specifically from the Kumi and Soroti districts.

### 3.3. Data Sources

Blood samples were provided by 140 patients from the following institutions:

Kumi Hospital

Soroti Regional Referral Hospital

Soroti University

### 3.4. Research Team and Funding

The dataset was prepared by:

Florence Tushabe, Mwesige Samuel, Kasule Vicent, David Areu, Philip Mutabazi, Emily Nsimire, Sarah Cheptoek Musani, and Emmanuel Othieno.

Funding: Supported by the Government of Uganda through the Soroti University Research and Innovation Fund (project number RIF/2022/05).

### 3.5. Dataset Structure

| Folder | Class | Image Count | Description |
| --- | --- | --- | --- |
| Positive/Labelled | Sickle (Positive) | 422 | Images with bounding boxes around visible sickle cells. |
| Positive/UnLabelled | Sickle (Positive) | (Included in 422) | Raw positive images without labels (for identification tasks). |
| Negative/Clear | Normal (Negative) | 147 | High-quality images of healthy red blood cells. |
| Not clear | Unclear | 122 | Images with artefacts, unusual colors, or staining issues. |

### 3.6. Publications and Citations

The research and dataset have been documented in the following publications:

Tushabe, F. B., Mwesige, S., Kasule, V., Nsiimire, E., Musani, S. C., et al. (2025). "An Image-Based Sickle Cell Detection Method". Engineering and Applied Sciences Journal, 2(1), 1-4. Link to Paper

Florence Tushabe., et al. (2024). "A Dataset of Microscopic Images of Sickle and Normal Red Blood Cells". Acta Scientific Microbiology 7.12: 22-29.

Conference Presentation: 5th Global Webinar on AI ML, Data Science & Robotics (April 15-16th, 2024). Paper title: "A Dataset of Microscopic Images of Sickle Cells".

## 4. Data Preprocessing

Transforming raw microscopic images into a mathematically stable format is a critical step before model training. The preprocessing pipeline for this project is bifurcated into shared geometric/intensity transformations and pathway-specific preparations depending on whether Deep Learning or Traditional Machine Learning algorithms are employed.

### 4.1. Base Image Standardization

Raw microscope images obtained from the clinical sources possess varying dimensions and noise profiles. To establish uniformity:

Resolution Resizing: All original images are aggressively down-sampled to a standard matrix size of 64 × 64 pixels. This dimension was selected to balance computational efficiency with the preservation of cellular morphology (e.g., the elongated shape of sickle cells).

Color Space: Images are processed continually in the standard 3-channel RGB color space to preserve diagnostic staining colors (Field and Leichman stains).

Intensity Rescaling: The raw pixel intensities, naturally ranging from 0 to 255, are normalized to a [0.0, 1.0] bounded range by dividing all values by 255.0. This prevents massive gradient updates and ensures stable, monotonic convergence during optimization.

### 4.2. Data Augmentation (Deep Learning Pipeline)

Given the inherent class imbalance (Sickle accounts for ~74% of the dataset) and to deter the rapid overfitting typical in Deep Convolutional Neural Networks on moderate-sized datasets, aggressive on-the-fly Data Augmentation is utilized exclusively on the training sets. Validating bounds are strictly isolated from augmentation.

The applied deterministic augmentations include:

Random Horizontal Flipping: Because red blood cell orientation on a slide is arbitrary.

Random Rotations: Application of rotational shifts (up to 20 degrees) to simulate variations in microscope slide placement.

Random Zooming: Scaling images in and out (up to 15%) to account for varying focal depths and cellular spread scaling.

Contrast & Brightness Adjustments: Random shifts in brightness ranges to make the model invariant to varied illumination and staining intensities across the Ugandan clinical hospitals.

### 4.3. Backbone-Specific Preprocessing (Transfer Learning)

When routing the dataset through the pre-trained ImageNet architectures (ResNet50, MobileNetV2, EfficientNetB0), the baseline 64x64 matrices require further specialized transformations:

Resolution Upscaling: The base matrices are up-sampled to 224 × 224 natively via an embedded Resizing layer to match the minimum input receptive fields expected by the ImageNet topologies.

Native Preprocessing Wrappers: The pixel arrays are routed through specific preprocess_input functions native to each architecture (e.g., zero-centering pixels around ImageNet means).

### 4.4. Dimensionality Reduction (Traditional ML Pipeline)

Traditional classifiers (like SVMs and Random Forests) struggle with raw spatial arrays. For these algorithms, preprocessing diverges:

Flattening: The 64x64x3 images are flattened into a single-dimensional vector of 12,288 features.

Standardization: A StandardScaler removes the mean and scales features to unit variance.

Principal Component Analysis (PCA): To eliminate multicollinearity and reduce the massive 12,288-dimensional space computationally, PCA compresses the array down to the 100 most critical principal components dictating the most variance.

## 5. Model Architecture

The classification of sickle cell morphology in this project is resolved through a multi-tiered modeling approach. Rather than relying on a singular hypothesis, the architecture benchmarks a lightweight custom Convolutional Neural Network (CNN) against heavyweight ImageNet pre-trained feature extractors, and establishes a foundational baseline using traditional Machine Learning classification algorithms.

### 5.1. Custom Convolutional Neural Network

A custom-built CNN acts as the primary deep-learning baseline. It is deliberately constructed to be mathematically lightweight, establishing a performance floor for spatial feature extraction directly from the 64x64 microscopic arrays.

Convolutional Blocks: The network utilizes three sequential blocks of convolution and downsampling.

Conv2D (32 filters, 3x3 kernel, ReLU) -> MaxPooling2D (2x2)

Conv2D (64 filters, 3x3 kernel, ReLU) -> MaxPooling2D (2x2)

Conv2D (128 filters, 3x3 kernel, ReLU) -> MaxPooling2D (2x2)

Fully Connected Head: The spatial outputs are collapsed via a Flatten layer and routed through a deep, fully connected classification head.

Dense (128 units, ReLU) -> Dropout (0.5)

Dense (64 units, ReLU) -> Dropout (0.5)

Dense (1 unit, Sigmoid)

Optimization: Trained strictly end-to-end via the Adam optimizer (Initial Learning Rate = 1e-3, leveraging a ReduceLROnPlateau callback) against a Binary Crossentropy loss function. The heavy 0.5 dropout blocks combat overfitting.

### 5.2. Transfer Learning Architectures

To leverage generalized, pre-learned geometric filters, the system implements cutting-edge Transfer Learning backbones. Three separate architectural paradigms were evaluated:

ResNet50: Utilizes deep residual skip-connections to solve the vanishing gradient problem in deep networks.

MobileNetV2: Utilizes depthwise separable convolutions to maintain high accuracy at a fraction of the parameter cost.

EfficientNetB0: Utilizes a carefully balanced compound scaling method for depth, width, and resolution.

#### 5.2.1. Unified Transfer Learning Head

All three backbones strip their original 1000-class ImageNet heads . The images are natively upscaled within the graph to 224x224, routed through network-specific preprocessing logic, and passed through the frozen backbones.

The extracted feature map feeds into a unified Custom Classification Head:

GlobalAveragePooling2D (Compressing the spatial dimensions).

Dense (128 units, ReLU) -> Dropout (0.5).

Dense (1 unit, Sigmoid).

#### 5.2.2. Two-Stage Fine-Tuning Optimization

Unlike the custom CNN, the Transfer Learning variants are trained via an advanced two-stage optimization routine:

Stage 1 - Feature Extraction: The entire backbone is completely frozen. Only the new custom classification head learns to map the pre-existing ImageNet features to the sickle cell problem (Adam lr=1e-3).

Stage 2 - Fine-Tuning: The top 20% of the backbone layers are unfrozen computationally (while critically keeping BatchNormalization layers permanently frozen to avoid destabilizing accrued statistics). The network is then trained uniformly end-to-end with a severely decayed learning rate (Adam lr=1e-5).

### 5.3. Traditional Machine Learning Baselines

To validate whether deep spatial convolutions are strictly necessary, a parallel architectural pipeline compresses the raw image arrays using Principal Component Analysis (retaining the top 100 maximum variance components).

These 100-dimensional vectors are evaluated across standard statistical estimators:

Support Vector Machine (SVM): Utilizing a Radial Basis Function (RBF) kernel to map the compressed pixels into a higher-dimensional linear separation space.

Random Forest Classifier: Utilizing an ensemble of 300 independent decision trees.

Logistic Regression / Naive Bayes / Decision Trees: Acting as rapid, interpretable probabilistic baselines.

## 6. Explainable AI (XAI) Architectures

Because neural networks natively operate as mathematical "black boxes," deploying them for critical morphological classification like Sickle Cell detection demands clinical interpretability. To ensure transparency, the system architecture terminates the opaque classification pipelines directly into three highly specialized Explainable AI (XAI) algorithms.

### 6.1. Grad-CAM (Gradient-weighted Class Activation Mapping)

Architectural Purpose: Global spatial localization within Deep Convolutional Networks.

Mechanism: Grad-CAM is architected to hook directly into the computational graph of the custom CNN and the Transfer Learning models. Using TensorFlow's automated gradient tracking , the architecture dynamically probes backward from the final classification logit down to the ultimate convolutional block (bypassing the flattened dense layers).

It calculates the gradients of the specific "Sickle" class prediction relative to the spatial feature map of that terminal Conv2D layer.

It applies Global Average Pooling across these gradients to compute distinct "importance weights" for every individual neuron filter.

It multiplies these weights against the raw feature map, applying a ReLU activation function strictly to zero out any spatial features that negatively influence the predicted target.

The remaining matrix is up-scaled to the native 64x64 coordinate space and overlaid on the original image through an OpenCV Jet Colormap, producing a high-contrast heatmap representing precisely where the model is looking.

### 6.2. LIME (Local Interpretable Model-agnostic Explanations)

Architectural Purpose: Local structural approximations via geometric perturbation.

Mechanism: Rather than relying on internal model gradients, the LIME architecture investigates the classifier externally as a true black box.

Superpixel Segmentation: When fed an image, LIME natively parses the cell geometry, breaking the image down into distinct "superpixels" (contiguous regions of similar pixels) rather than analyzing grid-based pixels independently.

Synthetic Data Perturbation: LIME generates hundreds of modified "perturbed" samples of the single image by systematically blacking out arbitrary permutations of these superpixels.

Surrogate Fitting: The primary CNN is forced to predict probabilities across this massive array of perturbed, partly hidden images.

Local Interpretable Routing: A mathematically simple, highly interpretable regression model (e.g., Ridge Regression) is trained strictly on this synthetic dataset. The linear model perfectly identifies the localized weight of each specific superpixel grouping, allowing LIME to visually highlight which components of the cell geometry explicitly dragged the probability toward the positive or negative classification class.

### 6.3. SHAP (SHapley Additive exPlanations)

Architectural Purpose: Game-theoretic absolute feature importance.

Mechanism: Based on cooperative game theory, the SHAP architecture guarantees that the computed feature importance strictly distributes the difference between the base baseline expectation and the current prediction. The architecture implements SHAP bifurcated across both the traditional and deep pipelines:

Deep Learning (GradientExplainer): The network strips away the terminal non-linear Sigmoid activation, calculating gradients directly off the raw logit outputs to ensure mathematical symmetry. It establishes a "background distribution" by analyzing a randomly sampled set of 100 images from both classes to model the baseline prediction. It computes partial-derivative Shapley values back to every raw pixel, which the system then aggregates dynamically into larger superpixel chunks to produce Waterfall and horizontal Importance Bar plots.

Traditional ML (KernelExplainer): For the standard algorithmic pipeline (SVM, Random Forest), a KernelExplainer runs permutations against the previously extracted 100 PCA Components. It calculates the exact marginal contribution of each abstract mathematical component, completely unlocking the interpretability of the traditional black-box mathematical classifiers.

## 7. Training Strategy

The training methodology is strictly standardized across the entire Sickle Cell morphology dataset to ensure that comparative evaluations between the Custom CNN, Transfer Learning Backbones, and Traditional Machine Learning models are statistically sound and free from variance anomalies.

### 7.1. Cross-Validation Protocol

To guarantee robustness and specifically counteract the aggressive class imbalance (74% Sickle vs. 26% Normal), the entire evaluation is anchored by a Stratified 5-Fold Cross-Validation strategy.

The raw dataset is iteratively partitioned into 5 independent folds, where the proportion of positive and negative classes is meticulously maintained dynamically across every single training and holdout validation split.

This approach ensures every single image is tested exactly once dynamically while training on the other 80% dynamically, providing an isolated, generalized validation output uninfluenced by "lucky" random seeds.

### 7.2. Custom CNN Optimization

The custom architecture is trained directly through standard end-to-end optimization against the augmented data stream.

Loss Function: BinaryCrossentropy, mapped perfectly against the terminal Sigmoid layer.

Optimizer: Adam optimizer, initiated dynamically with a steep 1e-3 learning rate.

Epoch Configuration: Bounded to a maximum of 30 isolated passes, allowing rapid weight convergence over the fully unthawed topology.

### 7.3. Transfer Learning: Advanced Two-Stage Fine-Tuning

Applying generalized ImageNet weights to a hyper-specific medical imaging domain requires intricate tuning to prevent gradient destruction. The ImageNet backbones (ResNet50, MobileNetV2, EfficientNetB0) are trained via a strict two-stage sequence:

Stage 1: Feature Extraction

State: The entire pre-trained transfer learning backbone is explicitly frozen. Only the new, randomized classification Dense Head (128 → 64 → 1) is permitted to learn.

Optimization: Configured rapidly (Adam, lr=1e-3, max 10 epochs). The head learns to aggressively map the pre-existing spatial features mathematically into Sickle Cell probabilities without disrupting the delicate convolutional filters beneath it.

Stage 2: Backbone Destabilization & Fine-Tuning

State: The top 20% of the terminal convolutional backbone layers are aggressively "unfrozen" to permit learning, explicitly adapting the highest level macro-features uniquely to blood cell geometry.

Normalization Safety: Vitally, all native BatchNormalization layers are explicitly kept permanently frozen. This prevents the training batches from destabilizing the statistical means and variances accrued originally on the ImageNet dataset.

Optimization: The network is completely recompiled using an intensely suppressed learning rate (lr=1e-5 for 12 maximal epochs) to allow only highly marginal, precision-focused updates to the newly unfrozen backbone depths without causing catastrophic forgetting.

### 7.4. Dynamic Callbacks & Regularization

To aggressively guard against deep learning overfitting (where models essentially memorize the clinical training data and fail generalized validation), identical dynamic callback monitors are attached identically across both the Custom CNN and the Transfer Learning variants:

ReduceLROnPlateau: Actively monitors the val_loss. If the validation error fails to improve after 3 specific epochs, it algorithmically reduces the active learning rate exponentially (by a factor of 0.3, dropping dynamically down to 1e-7) to force hyper-focused gradient descents.

EarlyStopping: Also monitors val_loss. If the network fails to improve generalizability over 5 consecutive epochs completely, the routine kills the training process prematurely and restores the mathematical weights from precisely the epoch possessing the lowest validation error.

### 7.5. Traditional Machine Learning Evaluation

The traditional Machine Learning estimators eschew deep iterative epochs. After identical dimensional reduction across the 5 independent Training folds (via the isolated spatial StandardScaler and 100-component PCA), estimators like the RandomForestClassifier (300 internal trees) and the RBF-scaled SVC compute their deterministic mathematical fits utilizing standard optimal structural hyperparameters natively mapped via Scikit-Learn.

## 8. Evaluation Metrics

Because medical diagnostic modeling inherently penalizes certain types of errors more strictly than others (e.g., misdiagnosing a diseased cell as healthy is often catastrophic compared to misdiagnosing a healthy cell as diseased), evaluating the architectures strictly on global categorical accuracy is insufficient.

To provide a comprehensive clinical appraisal, the algorithms across both the Deep Learning and Traditional Machine Learning pipelines are uniformly evaluated against a rigorous multi-dimensional suite of performance metrics.

### 8.1. Primary Algorithmic Metrics

Binary Accuracy: The core baseline measuring the raw probabilistic exactness of the model. It defines the literal percentage of correctly classified predictions (both true positive sickle and true negative normal) relative to the total test cohort.

Precision (Positive Predictive Value): Identifies the proportion of positive clinical identifications that were genuinely pathological. High precision indicates the network has a very low rate of False Positives (i.e., when the model claims a cell is sickle, it is highly reliable).

Recall (Sensitivity): Critically vital for pathological diagnostics, Recall defines the proportion of actual, true sickle cells within the slide matrix that were successfully identified by the algorithm. High recall ensures the network functionally minimizes False Negatives (failing to detect existing sickle cells).

F1-Score: Because the Ugandan data cohort contains an explicit class asymmetry (74% positive vs 26% negative), traditional raw accuracy can become highly deceptive. The F1-Score is deployed as the harmonic mathematically weighted mean of Precision and Recall. It specifically penalizes models attempting to achieve high accuracy simply by classifying everything as the majority class.

### 8.2. Threshold-Independent Metrics

ROC AUC (Receiver Operating Characteristic — Area Under Curve): Rather than judging the model outputs against a completely arbitrary terminal probabilistic threshold (e.g., decision limit = 0.5), the system plots the classification performance across all possible continuous thresholds.

The curve explicitly maps the mathematical True Positive Rate (Recall) against the False Positive Rate.

The total Area Under the Curve (AUC) is computed, establishing an absolute aggregate metric of the model's structural ability to distinguish linearly between pathological and healthy cell variations. (An AUC of 1.0 defines a theoretically perfect mathematical separator; an AUC of 0.5 defines randomized noise).

Note on Application: For the Deep Learning routes running Stratified 5-Fold validation, an overarching mean ROC AUC curve is linearly mapped and plotted across the 5 validation variances to visualize exact generalized performance stability and bounding error variations.

### 8.3. Contextual Matrix Evaluation

Categorical Confusion Matrices: Every pipeline ultimately yields a deterministic 2x2 prediction matrix generated directly via continuous probabilistic boundaries against the holdout splits. These natively generated matrices isolate exact counts grouping classifications exclusively into:

True Positives (TP): Sickle cell correctly analyzed as Sickle.

True Negatives (TN): Normal cell correctly analyzed as Normal.

False Positives (FP): Normal cell erroneously flagged as pathological.

False Negatives (FN): Sickle cell dangerously bypassed as normal. Visually plotting these matrices enables rapid, localized clinical intuition into exact network behavioral tendencies at the terminal phase.

## 9. Workflow / Pipeline Diagrams

To provide granular clarity on how the complex system architecture handles calculations iteratively, the overarching framework is broken down into four distinct, independent pipeline workflows. Each isolates the specific transformations occurring at that stage of the system.

### 9.1. Data Preparation Pipeline

This pipeline illustrates the physical ingestion of the Ugandan microscopic images and the mathematical bifurcation into geometric augmentation (for Deep Learning) and flattened PCA dimensionality reduction (for Traditional Machine Learning).

![Figure 2. Data preparation pipeline](/mnt/c/Users/sahod/Desktop/project/converted_report_assets/image2.png)
*Figure 2. Data preparation pipeline*

### Workflow Description:

Ingestion & Integrity Check: The raw dataset is directly loaded and immediately forced into a Stratified 5-Fold Split. This guarantees that all subsequent processing operations are strictly evaluated against varied, unbalanced data subsets without risking validation leakage.

Base Standardization: All images forcefully bypass arbitrary dimensions and are resized identically into a 64x64 grid and math-scaled to a 1/255 range.

Pipeline Bifurcation:

The Deep Learning Route requires massive variance to prevent memorization, so the pipeline injects Geometric Augmentation (aggressive flipping, zooming, and rotation).

The Traditional ML Route avoids spatial convolution, instead mathematically flattening the thousands of pixels. A Standard Scaler balances the numeric weight, before PCA compresses the noise strictly down to the top 100 mathematical components of variance.

### 9.2. Deep Learning Pipeline

This diagram isolates the dual-architecture approach of the deep neural networks. It explicitly branches between the rapid optimization of the Custom CNN Baseline and the complex, two-stage feature extraction / fine-tuning mechanics deployed by the pre-trained ImageNet backbones (ResNet, MobileNet, EfficientNet).

![Figure 3. Deep learning pipeline](/mnt/c/Users/sahod/Desktop/project/converted_report_assets/image3.png)
*Figure 3. Deep learning pipeline*

### Workflow Description:

Routing: Incoming augmented tensors are routed logically dependent entirely on architecture choice.

Baseline Flow (Custom CNN): Because the structure is natively built, the workflow skips freezing mechanics entirely. The spatial filters collapse into the 128->64 dense head and undergo immediate, direct End-to-End Adam Optimization.

Transfer Learning Flow: Massive ImageNet backbones must be tamed.

Stage 1: The workflow freezes the massive convolutional base physically. The tensor flows strictly to train the Unified Dense Head to learn how to map existing ImageNet filters cleanly to blood morphology.

Stage 2: The workflow systematically unfreezes the highest 20% of the backbone structure. A highly suppressed learning rate allows the model's top-level filters to delicately warp and fine-tune themselves specifically to sickle cell boundaries without catastrophic destruction. Both flows then terminate in binary categorization.

### 9.3. Traditional Machine Learning Pipeline

This streamlined branch isolates how the highly compressed 100-component PCA arrays are distributed into classical probability estimators, circumventing intense multi-epoch convolutional training for efficient algorithmic calculations.

![Figure 4. Traditional machine learning pipeline](/mnt/c/Users/sahod/Desktop/project/converted_report_assets/image4.png)
*Figure 4. Traditional machine learning pipeline*

### Workflow Description:

Feature Ingestion: The 100 extracted Principal Components flow identically across four disconnected estimator setups entirely concurrently.

Algorithmic Fitting:

The Support Vector Machine acts mathematically, mapping the 100-dimension vector into higher geometric dimensions via a Radial Basis Function (RBF) to draw a hyper-plane separating Sickle vs. Normal.

The Random Forest generates rapidly branching logic, synthesizing a consensus conclusion across 300 independent decision trees to battle outliers.

Logistic Regressions, Decision Trees, and Naive Bayes operate as highly interpretable statistical checks establishing speed and probabilistic confidence floors.

Synthesis: All discrete outputs are merged computationally for final clinical Sickle/Normal validation metric scoring versus the true labels.

### 9.4. Explainable AI (XAI) Pipeline

This final, critical pipeline details how both the raw input images and the completed output probabilities are fed iteratively into the interpretability algorithms. It shows the distinct methodologies between mapping internal gradients (Grad-CAM), testing synthetic boundary perturbations (LIME), and computing cooperative marginal distributions (SHAP).

![Figure 5. Explainable AI pipeline](/mnt/c/Users/sahod/Desktop/project/converted_report_assets/image5.png)
*Figure 5. Explainable AI pipeline*

### Workflow Description:

Intercept Point: The pipeline begins when a model outputs a final probability against a specific input image. The image alongside its predicted score is captured.

Grad-CAM Pathway: The system probes vertically backward into the model structure, dynamically targeting the ultimate Convolutional Block. It mathematically computes how strongly targeted gradients impact that exact spatial filter. A generated mathematical heatmap is directly overlaid natively across the original cell structure.

LIME Pathway: Bypassing internal structures, LIME chops the incoming image physically into chunks (Superpixels). The model prediction is forced against mathematically hidden permutations of these chunks. A highly transparent Surrogate Model learns precisely which un-hidden chunks consistently force a sickle prediction, displaying a geometric relevance mask.

SHAP Pathway: Evaluating strictly globally and natively, DeepExplainer and KernelExplainer establish mathematical baselines across multiple dataset expectations. Marginal Shapley values are calculated against every pixel or component linearly, structurally compiling output onto Waterfall flows, Bar importance charts, and absolute probability mapping overlays.

## 10. Deployment and Integration

### 10.1. Deployment Overview

The trained sickle cell classification models are deployed as an interactive web-based diagnostic tool using Streamlit, a Python-based rapid application framework. The deployment interface enables clinical or research users to upload a peripheral blood smear image and receive a real-time morphological classification (Sickle or Normal) along with a visual explainability output. The final model weights are persisted as .keras configuration files and loaded at application startup for efficient inference without retraining.

### 10.2. System Stack

The deployment stack consists of the following components. The frontend is built using Streamlit, providing an intuitive browser-based user interface that requires no installation on the client side. The backend inference engine is powered by TensorFlow/Keras, which loads the best-performing fine-tuned model (selected based on cross-validation AUC scores) to perform binary classification on the uploaded image. The explainability module runs Grad-CAM heatmap generation in real time on the same input image and overlays the activation map directly within the interface. The entire application is containerized using Docker to ensure consistent, reproducible execution across different operating environments including local machines, cloud instances, and hospital servers.

### 10.3. User Interaction Flow

The end-to-end user interaction proceeds as follows. A clinician or researcher uploads a microscopic blood smear image (JPG or PNG format) through the Streamlit interface. The application preprocesses the image to the required input dimensions and passes it through the loaded model to generate a prediction probability. The system displays the predicted class label (Sickle or Normal) along with a confidence score. Simultaneously, a Grad-CAM heatmap is generated and overlaid on the input image to visually highlight the cellular regions that most strongly influenced the classification decision. The user can optionally download the annotated output image for documentation or reporting purposes.

### 10.4. Model Persistence and Portability

The best model from each architecture (Custom CNN, EfficientNetB0, ResNet50, MobileNetV2) is saved as a .keras file upon training completion. These weight files are version-controlled and can be loaded independently for inference without requiring access to the original training pipeline. The modular design of the deployment system allows seamless swapping of model files, enabling future upgrades to improved architectures without modifying the interface or inference logic.
