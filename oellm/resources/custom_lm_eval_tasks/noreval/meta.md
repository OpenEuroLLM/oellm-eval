# NorEval task metadata

Summary of the `metadata` blocks in the task configs (all at version 1.2).

| Task | Category | Language(s) | Primary metric | Random baseline |
|:---|:---|:---|:---|:---|
| ASK GEC | linguistic_knowledge | nob | errant_f05 | 0.0 |
| NCB | linguistic_knowledge | nob | acc | 0.5 |
| NoCoLA | linguistic_knowledge | nob | acc | 0.5 |
| Sámi MultiBLIMP | linguistic_knowledge | sme | acc | 0.5 |
| NorIdiom (nob) | linguistic_knowledge | nob | em_first | 0.0 |
| NorIdiom (nno) | linguistic_knowledge | nno | em_first | 0.0 |
| SLIDE | linguistic_knowledge | nob, nno, swe, dan | acc | 0.21289208633093526 |
| NorBelebele | language_understanding | nob | acc | 0.25 |
| NoReC (document) | language_understanding | nob | f1 | 0.4850709806 |
| NoReC (sentence) | language_understanding | nob | f1 | 0.4815172167 |
| NorOpenbookQA (nob) | language_understanding | nob | acc | 0.25 |
| NorOpenbookQA (nno) | language_understanding | nno | acc | 0.25 |
| NorQuAD | language_understanding | nob | f1 | 0.0 |
| NorCommonsenseQA (nob) | world_knowledge_and_reasoning | nob | acc | 0.2 |
| NorCommonsenseQA (nno) | world_knowledge_and_reasoning | nno | acc | 0.2 |
| NorOpenbookQA (nob, without facts) | world_knowledge_and_reasoning | nob | acc | 0.25 |
| NorOpenbookQA (nno, without facts) | world_knowledge_and_reasoning | nno | acc | 0.25 |
| NorTruthfulQA (nob, generative) | world_knowledge_and_reasoning | nob | rougeL_max | 0.0 |
| NorTruthfulQA (nno, generative) | world_knowledge_and_reasoning | nno | rougeL_max | 0.0 |
| NorTruthfulQA (nob, multiple choice) | world_knowledge_and_reasoning | nob | acc | 0.23170337745132827 |
| NorTruthfulQA (nno, multiple choice) | world_knowledge_and_reasoning | nno | acc | 0.23311814890762259 |
| NRK Quiz QA (nob) | world_knowledge_and_reasoning | nob | acc | 0.2836296296296296 |
| NRK Quiz QA (nno) | world_knowledge_and_reasoning | nno | acc | 0.27884711779448623 |
| NorRewrite Instruct| generation_and_summarization | nob | bleu | 0.0 |
| NorSummarize Instruct | generation_and_summarization | nob | bleu | 0.0 |
| NorSumm (nob) | generation_and_summarization | nob | rougeL_max | 0.0 |
| NorSumm (nno)| generation_and_summarization | nno | rougeL_max | 0.0 |
| Translation (nob → nno) | machine_translation | nob, nno | bleu | 0.0 |
| Translation (nno → nob) | machine_translation | nob, nno | bleu | 0.0 |
| Translation (eng → nob) | machine_translation | nob, eng | bleu | 0.0 |
| Translation (eng → nno) | machine_translation | nno, eng | bleu | 0.0 |
| Translation (nob → eng) | machine_translation | nob, eng | bleu | 0.0 |
| Translation (nno → eng) | machine_translation | nno, eng | bleu | 0.0 |
| Translation (nob → sme) | machine_translation | nob, sme | bleu | 0.0 |
| Translation (sme → nob) | machine_translation | nob, sme | bleu | 0.0 |
