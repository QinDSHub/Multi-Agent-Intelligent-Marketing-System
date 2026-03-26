<<<<<<< HEAD
# Main Optimizations of Version 2 for the Customer Churn Analysis<br>
This version focuses on data quality improvements, feature engineering, and a hybrid retrieval-based prediction strategy.<br>

## Feature Cleaning: repair_type<br>
The repair_type feature was further standardized and cleaned. It is now used for:<br>
&emsp;•	Data filtering<br>
&emsp;•	Text-based semantic representation<br>
________________________________________
## Removal of Internal Vehicles<br>
Internal vehicles were removed from the dataset to prevent potential bias in the analysis and modeling process.
________________________________________
## Filtering Non-Active Service Visits<br>
Only active service visits were retained. Records related to passive visits were removed.<br>
Examples of passive visits include:<br>
&emsp;•	Warranty/PDI claims<br>
&emsp;•	Accident repairs<br>
&emsp;•	Mandatory maintenance<br>
&emsp;•	Warranty-related services
________________________________________
## Additional Data Cleaning<br>
Further cleaning was applied to noisy or problematic records, such as removing invalid records (e.g., negative day differences) and imputed missing or abnormal values (e.g., mileage) using each user’s median daily metrics.
________________________________________
## Churn Labeling Strategy<br>
Users who have not actively visited the service center for three years were labeled as: churn = 100%<br>
This provides a clearer signal for churn identification and improving model calculating efficiency.
________________________________________
## Feature Engineering<br>
Both numerical features and text features were incorporated into the model.<br>
Numerical Features<br>
&emsp;•	Standardized using scaling, the following preprocessing strategy was mainly used:<br>
&emsp;&emsp;- Outlier detection: RobustScaler applied to columns containing extreme values<br>
&emsp;&emsp;- Skewness detection: PowerTransformer utilized for highly skewed distributions<br>
&emsp;&emsp;- Default handling: StandardScaler used for all remaining columns<br>
&emsp;•	The scaler was fit on the entire dataset to ensure consistent scaling across all samples<br>

Text Features<br>
&emsp;•	Converted into 1536-dimensional embeddings using the OpenAI text embedding model<br>
&emsp;•	These embeddings capture semantic similarity
________________________________________
## Feature Fusion<br>
Numerical and text embeddings were concatenated horizontally.<br>
Feature weights were applied:<br>
&emsp;•	Numerical features: 70%<br>
&emsp;•	Text features: 30%<br>
After concatenation, L2 normalization was applied and made sure fit on full dataset to ensure consistent vector scaling, and after that to split into train and valid.
________________________________________
## Why LLM Inference Was Not Used<br>
In the previous version, I transformed numerical features into binned textual features and fed them into an LLM. However, a very important and fundamental concept has been overlooked: text embedding models struggled to distinguish values such as “1-year car age” vs “11-year car age”, actually “1-year car age” should be much more similar with ‘2-year car age’ rather than ’11-year car age’.<br><br>
While LLMs are extremely powerful for text reasoning, this task is primarily driven by structured numerical signals, so relying on LLM-based inference did not provide meaningful benefits. This led to the current hybrid approach.
________________________________________
## Training and Validation Split<br>
•	The dataset was split into training and validation sets, the size of total dataset is 55129 samples, and 10% as valid dataset.<br>
•	Custom Training embeddings were stored in ChromaDB for vector similarity search
________________________________________
## Prediction Method (KNN-Style Retrieval)<br>
Instead of using a traditional classifier, this version adopts a vector similarity retrieval strategy.<br>
For each sample in the validation set:<br>
&emsp;•	Retrieve the Top-10 most similar users using cosine similarity<br>
&emsp;•	Apply a KNN-style majority voting strategy<br>
&emsp;•	Assign the majority label as the final prediction, the threshold is set as 0.4 will get the best AUC with 0.936
________________________________________
## Result<br>
Using this hybrid retrieval-based approach, the model achieved: AUC = 0.936
________________________________________
## Limitations and practical considerations<br>
While the approach performs well in many scenarios, several practical considerations are worth highlighting.<br>

First, the effectiveness of KNN-based similarity retrieval depends heavily on whether sufficiently similar users actually exist in the dataset. In certain edge cases, the nearest neighbors retrieved may still be weakly related to the target user. Directly transferring labels or behavioral signals from such neighbors may introduce noise rather than meaningful signal. A practical mitigation strategy is to introduce a similarity threshold, ensuring that only high-confidence neighbors contribute to downstream inference. In addition, multiple similarity or distance metrics can be incorporated to better monitor and validate the robustness of the retrieval process.<br>

Second, the choice of text embedding model should be aligned with the complexity and richness of the textual features. OpenAI embeddings generate Transformer-based representations with 1536 dimensions, which provide strong semantic capacity but may be unnecessarily heavy for simple or limited textual inputs. In those cases, lighter embedding models with lower-dimensional representations may offer a more efficient trade-off between semantic fidelity and computational cost.<br>

Finally, an interesting extension of thinking is that the models developed can naturally serve as Modular Capability Providers (MCPs) within agent or multi-agent architectures, enabling them to be orchestrated as reusable components in larger decision or automation pipelines.
________________________________________
## Key Insight<br>
Following the principle of Occam’s Razor, simplicity often outperforms unnecessary complexity. In practice, high-quality data cleaning, thoughtful feature engineering, and well-designed retrieval mechanisms can sometimes deliver stronger results than increasingly complex modeling stacks.<br>
________________________________________
## Project Status<br>
This project is currently still in the exploratory and development stage.<br>
The system has not yet been fully wrapped into a multi-agent or production-ready pipeline, so the full codebase is not published at this moment.<br>
The repository and implementation details will be shared once development is completed.
________________________________________
## Discussion<br>
Feedback, reviews, and discussions are very welcome.<br>
If you are working on similar problems involving hybrid features, vector retrieval, or churn prediction, feel free to share your thoughts or suggestions.
=======
# A Multi-Agent Intelligent Marketing System

This repository presents Version 3.0 of a multi-agent intelligent marketing system that implements a closed-loop, end-to-end automation pipeline for marketing intelligence and content operations. The system integrates retrieval-augmented generation (RAG), structured multi-agent orchestration, and human-in-the-loop (HITL) validation to support reliable insight extraction, content generation, automated publishing, and post-publication performance analysis.

Built upon LangChain and LangGraph, the proposed architecture emphasizes modularity, grounding, and extensibility, enabling controlled large language model (LLM) deployment in real-world marketing workflows. Compared with earlier versions, this iteration focuses on architectural optimization, explicit agent responsibility boundaries, and improved reproducibility.

---

## System Overview

The proposed system adopts a sequential multi-agent architecture, in which each agent is responsible for a well-defined stage in the marketing intelligence lifecycle. Agents communicate via structured outputs validated by Pydantic schemas, ensuring consistency and robustness across the pipeline.

The complete workflow forms a closed feedback loop, beginning with external and internal knowledge acquisition and ending with automated performance evaluation to support iterative strategy refinement.

---

## Agent Architecture and Pipeline
### **Search Agent**

The Search Agent retrieves professional and industry-grade textual sources to ground downstream reasoning and mitigate hallucination risks in LLM-generated outputs.

External information is collected using Google SerpAPI.

Retrieved documents serve as authoritative references for subsequent retrieval and generation tasks.

### **Search Document Loader Agent**

This agent extracts raw textual content from URLs provided by the Search Agent and applies standardized preprocessing procedures, including text cleaning, normalization, and noise removal.

### **Local Document Loader Agent**

To support domain-specific and proprietary knowledge integration, the Local Document Loader Agent enables users to upload PDF documents from local storage. These documents undergo the same preprocessing pipeline as externally sourced materials, ensuring uniform representation.

### **Retrieval-Augmented Generation (RAG) Agent**

The RAG Agent implements a Map–Reduce-style retrieval framework:

*** Map Phase ***

- Document chunking and embedding

- Vector storage using a temporary Chroma database

- Sources include both online documents and locally uploaded files

*** Reduce Phase ***

- Semantic retrieval and ranking of the most relevant document chunks in response to user queries

- This design grounds LLM reasoning in retrieved evidence and improves factual consistency.

### **Insights Agent**

The Insights Agent synthesizes retrieved information into a concise set of five high-level, actionable insights. These insights function as an intermediate semantic abstraction layer, decoupling knowledge retrieval from content generation.

### **Content Generation Agent**

The Content Generation Agent produces publication-ready marketing content using strictly structured outputs defined via Pydantic schemas. The output schema includes:

- Insight identifier linkage

- Content format specification (e.g., article, blog, case study)

- Headline and full textual body

- Call-to-action and target audience definition

- Recommended distribution channels

- Explicit source references

While the current implementation focuses on text generation, the architecture supports future extensions to multimodal generation (e.g., video scripts, visual assets) and multilingual content production via additional agents.

### **Human-in-the-Loop (HITL)**

To ensure quality control and ethical deployment, a Human-in-the-Loop mechanism is incorporated. Generated content undergoes manual review, validation, and optional revision before publication, ensuring alignment with organizational standards.

### **Auto Publish Agent**

The Auto Publish Agent schedules and disseminates approved content across social media or marketing platforms.

A Facebook API integration is provided as a reference implementation.

The design is platform-agnostic and supports extensibility.

### **Auto Analysis Report Agent**

Following publication, the Auto Analysis Report Agent generates analytical summaries evaluating engagement metrics, effectiveness, and return on investment (ROI). These reports enable data-driven optimization of future marketing strategies, closing the system feedback loop.

---

## Design Principles

The system is guided by the following design principles:

- Modularity: Each agent is independently extensible and replaceable

- Grounded Generation: RAG is employed to constrain LLM outputs to retrieved evidence

- Structured Communication: Pydantic schemas enforce data consistency across agents

- Human Oversight: HITL safeguards are integrated at critical decision points

Version 3.0 prioritizes architectural clarity and experimental reproducibility. Production-level optimizations such as asynchronous execution, caching, and persistent long-term memory are intentionally excluded to maintain research focus.

---

## Implementation Details
### **Technology Stack**

- Python 3.12.7

- chromadb 1.4.1

- langchain 1.2.8

- langchain-openai 1.1.7

- langgraph 1.0.7

- openai 2.17.0

---

## GenAI Assistance Disclosure

The system architecture, agent design, and implementation were independently conceived and developed by the author. Generative AI tools were employed in a supportive capacity with: GitHub Copilot was used to assist with code refactoring and optimization, while ChatGPT was utilized exclusively for academic-style language refinement and documentation editing. All technical decisions, system logic, and intellectual contributions remain solely attributable to the author.
>>>>>>> 15c3692ed2bbade317de31f5d7b2ac8727906f46
