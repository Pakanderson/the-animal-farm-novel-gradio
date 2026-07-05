# 📋 Animal Farm App Analyzer & Testing Guide (Gradio Edition)

Welcome to the **Animal Farm RAG Application Testing Suite**! This guide is designed for colleagues, presenters, and users new to George Orwell's *Animal Farm* to evaluate the performance, accuracy, and retrieval capabilities of our Gradio application.

---

## 🚀 Getting Started (Gradio)

To launch the Gradio web interface locally, run the following command in your terminal:

```bash
python app_gradio.py
```

> 💡 **Tip for Presenters:** Adjust the `top_k` slider panel directly inside the Gradio blocks while testing to see how changing chunk retrieval count actively alters the precision of the generated answers.

---

## 🎯 Purpose of This Document

Instead of providing the answers directly, use these structured test questions to probe the application. Observe how the underlying Language Model (LLM) and Retrieval-Augmented Generation (RAG) system fetch context and formulate summaries. 

> **How to Test:** Copy and paste these questions into the Gradio `gr.Chatbot` interface during your live evaluation or presentation. Note down how the app responds!

---

## 🏛️ Phase 1: Core Characters & Allegories

*These questions evaluate if the application accurately maps characters to their historical counterparts and underlying narrative motives.*

* 🔹 **Which pig became the farm's dictator?**
* 🔹 **Who represents Leon Trotsky in the book?**
* 🔹 **Which animal always loves sugar cubes?**
* 🔹 **Why did Snowball get chased away?**

---

## 📜 Phase 2: The Seven Commandments & Rules

*These questions evaluate if the vector database can pull specific chapter-level textual changes and identify how propaganda modifies rules over time.*

* 🔹 **List the 7 commandments.**
* 🔹 **Which commandment did the pigs change first?**
* 🔹 **Why did "no beds" get an upgrade?**
* 🔹 **What replaced all the original seven commandments?**
* 🔹 **Are some animals more equal than others?**

---

## 🚜 Phase 3: Plot Points & Historical Concepts

*These questions explore the broader systemic collapses, human vices, and allegorical parallels to Imperial Russia/the Soviet Union.*

* 🔹 **What happened to Boxer the loyal horse?**
* 🔹 **Why did the pigs start drinking beer?**
* 🔹 **Do the sheep have any original thoughts?**
* 🔹 **What does Manor Farm represent historically?**
* 🔹 **Can you spot Mr. Jones's biggest mistake?**

---

## 🧪 Evaluation Metrics for Reviewers

When interacting with the app using the questions above, please consider the following points for discussion:

* ✔️ **Context Accuracy:** Did the bot correctly locate the relevant sections of the novel?
* ✔️ **Tone Consistency:** Did the response maintain a helpful, informative, and educational stance?
* ✔️ **Length Constraints:** Are the answers concise, or did the app output excessively long essays?
