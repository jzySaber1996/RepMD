# README for RepMD

- Summary: This is the replicable package for RepMD.
- Paper Name: "All Changes May Have Invariant Principles: Improving Ever-Shifting Harmful Meme Detection via Design Concept Reproduction"
- Submission: ARR Jan, 2026

## Abstract

Harmful memes are ever-shifting in the Internet communities, which are difficult to analyze due to their type-shifting and temporal-evolving nature. Although these memes are shifting, we find that different memes may share invariant principles, i.e., the underlying design concept of malicious users, which can help us analyze why these memes are harmful. In this paper, we propose RepMD, an ever-shifting harmful meme detection method based on the design concept reproduction. We first refer to the attack tree to define the Design Concept Graph (DCG), which describes steps that people may take to design a harmful meme. Then, we derive the DCG from historical memes with design step reproduction and graph pruning. Finally, we use DCG to guide the Multimodal Large Language Model (MLLM) to detect harmful memes. The evaluation results show that RepMD achieves the highest accuracy with 81.1% and has slight accuracy decreases when generalized to type-shifting and temporal-evolving memes. Human evaluation shows that RepMD can improve the efficiency of human discovery on harmful memes, with 15-30 seconds per meme.


## Code and Dataset Available

_**Due to the confidentiality review requirements of specific political entities involved in the paper, we will gradually the disclosure of the complete code.**_

- **[Code](/code/):** Availble code for the replication (part of it remains close-source and need to be gone through the confidentiality review).
- **[Dataset](/dataset/):** Dataset with two sources (part of it remains close-source and need to be gone through the confidentiality review).