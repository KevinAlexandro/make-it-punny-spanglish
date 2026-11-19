# Project Title

**Spanglish, But Make It Punny: Cross-Lingual Phonetic Homophone Mining and Structured Prompting for Spanish-English Code-Mixed Pun Generation**

## Project Keywords

Computational Humor, Code-Mixing, Spanglish, Pun Generation, Phonetic Homophones, Large Language Models, Structured Prompting

## Group Members

- Michaela Foo, mfa72@sfu.ca, #301431416 (CMPT 419)
- Kevin Gonzalez, kag18@sfu.ca, #301466855 (CMPT 983)

## Project Summary

Puns, known as a form of wordplay, provide humor by exploiting phonetic or semantic ambiguity, while code-mixing is a phenomenon when a speaker blends multiple languages within the same conversation. Combining puns with code-mixed text can be challenging due to its requirements to align the phonology and semantics of languages as well as make the pun humorous and meaningful.

Building on the frameworks of Asapu et al. (2025) and Sarrof (2025) for Hindi-English pun generation, this project adapts and evaluates a two-step pipeline that combines IPA sound-matching with LLM prompting for Spanish-English code-mixed puns. First, we use the ipa-dict dataset to extract phonetically similar common Spanish and English vocabulary. Such dataset has words in International Phonetic Alphabet (IPA) representation which would enable us to identify cross-lingual homophones and near-homophones via custom weighted Levenshtein edit distance. We will then apply a translation-based filter to remove cognates (words that mean the same and are spelled the same in different languages) and retain only semantically distinct pairs. Second, we feed the 100–150 curated Spanish-English word pairs into four LLM prompting strategies: a direct Baseline, Contextually Aligned, Question-Answer, and Subject-Masked generation. Finally, we evaluate the generated Spanglish outputs across human raters and an automated LLM judge on Pun Success, Funniness, and Acceptability to determine which structured prompting strategy best bridges cross-lingual wordplay.

> **Optional:** Use Epitran in case the IPA dictionaries do not return homophones as expected.

## Project Description

### What are the objectives of the Project?

1. **Build a Spanish-English pun pair dataset using Cross-Lingual Homophone Mining and Cognate Filtering**

   Construct a curated dataset of 100–150 Spanish-English pun pairs, (Pw ,Aw ), ensuring the Spanish pun word (Pw) and English alternate word (Aw) are phonetically similar and semantically distinct. Potential pairs will be identified using IPA pronunciations, then will be filtered through to remove cognates or other unsuitable pairs. Each final pair will include Spanish word, English word, pronunciations, meanings, and phonetic similarity score.

2. **Generate Spanglish Puns using Few-Shot Prompting:**

   Using the selected word pairs to generate puns with an LLM, adapting and applying the three structured few-shot LLM prompting methodologies from Asapu et al. (2025)—Contextually Aligned, Question-Answer, and Subject-Masked—against a baseline prompt. The word pairs, model, and settings will be kept the same to ensure that we will be able to fairly evaluate and compare between the methods.

3. **Evaluate Pun Quality and Error Analysis**

   Once we have a curated list of 100 chosen sentences, we will have humans evaluate the puns using the following measures: whether the sentence contains a real pun, how funny it is, and how natural and grammatical it sounds. Also identify common mistakes and errors that occurred during pun generation such as incorrectly used language, sentences that do not contain a pun, unnatural sounding sentences or words that are not phonetically similar.

### What activities will be undertaken by whom and by when to achieve these objectives?

#### Phase 1: Data Extraction & Phonetic Word-Pair Pipeline (Week 1 — By Nov 17)

- **Kevin & Michaela:** Extract high-frequency Spanish and English word lists, lemmas, and translations from doozan/spanish_data and open-dict-data/wikidict-es.
- **Kevin:** Implement the Python phonetic transcription and matching script using pre-existing IPA pronunciation dictionaries (ipa-dict) and edit distance (the minimum character edits needed to turn one word into another) to match similar-sounding Spanish and English words.
- **Michaela:** Manually inspect the candidate output list to verify definitions and finalize the clean dataset of 100–150 Spanish-English pun/alternate word tuples (Pw, Aw, Cw).

#### Phase 2: Prompt Adaptation & LLM Pun Generation (Week 2 — By Nov 24)

- **Michaela:** Adapt the prompt templates (Baseline, Contextually Aligned, Question-Answer, and Subject-Masked) from Asapu et al. (2025) to Spanglish, crafting 3–5 few-shot examples per method.
- **Kevin:** Run the batch LLM API generation script across the finalized word pairs for all four methods and compile the outputs into an evaluation sheet.

#### Phase 3: Evaluation & Final Analysis (Week 3 — By Dec 1)

- **Michaela:** Set up the rating spreadsheet/Google Form for 100 sampled outputs (25 pairs x 4 methods) and coordinate ratings on Pun Success (0/1), Funniness (1–5), and Acceptability (1–5).
- **Kevin & Michaela:** Compute quantitative metrics (Success %, Mean Funniness, Mean Acceptability, and inter-annotator agreement), categorize common LLM failure modes, and write the final report and presentation.

## Resources

- **Lexical & Dictionary Datasets:** doozan/spanish_data (GitHub: Spanish-to-English dictionary, word frequency lists, and lemma data); open-dict-data/wikidict-es and ipa-dict (GitHub: open-licensed bilingual Spanish-English reference and pre-computed IPA pronunciation dictionaries).
- **Reference Codebases:** Asapu et al. (2025) Code-Mixed Pun Generation Repository (github.com/Likhith-Asapu/Codemix-Pun-Generation); Sarrof (2025) Hinglish Pun Repository (github.com/yashYRS/HinglishPun).
- **Computational Tools & Software:** Python 3 (pandas, numpy, scipy, scikit-learn); Epitran (Mortensen et al., 2018) for grapheme-to-phoneme conversion to map Spanish and English orthography into IPA; Levenshtein / Custom Edit Distance for phonetic similarity scoring and cognate filtering; OpenAI API, Anthropic API and/or Google Gemini API (3.8 Flash) for few-shot candidate sentence generation; Google Forms / Spreadsheets for human evaluation.

## References

Asapu, L., Kodali, P., Dua, A., Kavitha, K. R., & Shrivastava, M. (2025). Bridging Laughter Across Languages: Generation of Hindi-English Code-mixed Puns. In Proceedings of the 1st Workshop on Computational Humor (CHum) (pp. 32–57).

Attardo, S. (2020). The Linguistics of Humor: An Introduction. Oxford University Press.

Attardo, S., & Raskin, V. (1991). Script theory revis(it)ed: Joke similarity and joke representation model. Humor - International Journal of Humor Research, 4(3-4), 293–348.

Castro, S., Chiruzzo, L., & Rosá, A. (2018). Overview of the HAHA task: Humor analysis based on human annotation at IberEval 2018. In IberEval@SEPLN (pp. 187–194).

He, H., Peng, N., & Liang, P. (2019). Pun generation with surprise. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT) (pp. 1734–1744).

Miller, T., Hempelmann, C., & Gurevych, I. (2017). SemEval-2017 Task 7: Detection and interpretation of English puns. In Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017) (pp. 58–68).

Mortensen, D. R., Dalmia, S., & Littell, P. (2018). Epitran: Precision G2P for many languages. In Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018).

Sarrof, Y. (2025). Homophonic Pun Generation in Code Mixed Hindi English. In Proceedings of the 1st Workshop on Computational Humor (CHum) (pp. 23–31).
